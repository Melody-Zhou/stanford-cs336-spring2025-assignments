import math
import torch


def flash_bwd_recompute_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    L: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Inputs:
      q,k,v,o,do: (B, Q/K, D)
      L: (B, Q)   where L = logsumexp(S, dim=-1)
    Returns:
      dq, dk, dv    
    """
    B, Q, D = q.shape
    K = k.shape[1]
    scale = 1.0 / math.sqrt(D)

    qf = q.float()
    kf = k.float()
    vf = v.float()
    of = o.float()
    dof = do.float()
    Lf = L.float()

    # S = QK^T * scale : (B, Q, K)
    S = torch.matmul(qf, kf.transpose(-1, -2)) * scale

    if is_causal:
        q_idx = torch.arange(Q, device=S.device)[:, None]
        k_idx = torch.arange(K, device=S.device)[None, :]
        casual = (q_idx >= k_idx)  # (Q, K)
        S = torch.where(casual[None, :, :], S, torch.full_like(S, -1.0e6))
    
    # P = exp(S - L) : (B, Q, K)
    P = torch.exp(S - Lf.unsqueeze(-1))

    # dV = P^T @ dO : (B, K, D)
    dv = torch.matmul(P.transpose(-1, -2), dof)

    # dP = dO @ V^T : (B, Q, K)
    dP = torch.matmul(dof, vf.transpose(-1, -2))

    # Dvec = sum(dO * O, dim=-1) : (B, Q)
    Dvec = (dof * of).sum(dim=-1)

    # dS = P * (dP - Dvec) * scale : (B, Q, K)
    dS = P * (dP - Dvec.unsqueeze(-1)) * scale

    # dQ = dS @ K : (B, Q, D)
    dq = torch.matmul(dS, kf)

    # dK = dS^T @ Q : (B, K, D)
    dk = torch.matmul(dS.transpose(-1, -2), qf)

    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)  


class FlashAttention2Pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
        # q: (B, Q, D), k/v: (B, K, D)
        if q.dim() < 3:
            raise ValueError("q must have shape (..., Q, D)")
        if k.dim() != q.dim() or v.dim() != q.dim():
            raise ValueError("q/k/v must have same rank")
        if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]:
            raise ValueError("leading dims of q/k/v must match")
        if k.shape[-2] != v.shape[-2] or k.shape[-1] != v.shape[-1]:
            raise ValueError("k and v must have sanme shape in last 2 dims")
        if q.shape[-1] != k.shape[-1]:
            raise ValueError("q and k must have same D")
        
        B, Q, D = q.shape
        K = k.shape[-2]        

        # ignore is_causal
        scale = 1.0 / math.sqrt(D)

        # choose tile sizes (>=16)
        Bq = 32
        Bk = 32

        # output buffers
        o = torch.empty((B, Q, D), device=q.device, dtype=q.dtype)
        L = torch.empty((B, Q), device=q.device, dtype=torch.float32)

        for i in range(0, Q, Bq):
            q_i = q[:, i : i + Bq, :]  # (B, Bq, D)

            # running stats for this query tile
            m = torch.full((B, Bq), -float("inf"), device=q.device, dtype=torch.float32)
            l = torch.zeros((B, Bq), device=q.device, dtype=torch.float32)
            o_acc = torch.zeros((B, Bq, D), device=q.device, dtype=torch.float32)

            for j in range(0, K, Bk):
                k_j = k[:, j : j + Bk, :]   # (B, Bk, D)
                v_j = v[:, j : j + Bk, :]   # (B, Bk, D)

                # S = q @ k^T * scale -> (B, Bq, Bk)
                S = torch.matmul(q_i, k_j.transpose(-1, -2)) * scale

                # online softmax update
                m_new = torch.maximum(m, S.max(dim=-1).values)   # (B, Bq)

                # exp(S - m_new)
                P_tilde = torch.exp(S - m_new.unsqueeze(-1))   # (B, Bq, Bk)

                # l_new = exp(m - m_new) * l + rowsum(P_tilde)
                l_new = torch.exp(m - m_new) * l + P_tilde.sum(dim=-1)   # (B, Bq)

                # o_acc = exp(m - m_new) * o_acc + P_tilde @ v_j
                o_acc = (torch.exp(m - m_new)).unsqueeze(-1) * o_acc + torch.matmul(P_tilde, v_j)

                m, l = m_new, l_new
            
            # finalize: O = o_acc / l
            o_i = o_acc / l.unsqueeze(-1)
            o[:, i : i + Bq, :] = o_i.to(dtype=q.dtype)

            # L = logsumexp(S_row) = m + log(l)
            L[:, i : i + Bq] = m + torch.log(l)

        # save tensor for later backward stage
        ctx.save_for_backward(L, q, k, v, o)
        ctx.is_causal = is_causal

        # return output
        return o
    
    @staticmethod
    def backward(ctx, do):
        (L, q, k, v, o) = ctx.saved_tensors
        dq, dk, dv = flash_bwd_recompute_impl(q, k, v, o, do, L, ctx.is_causal)
        return dq, dk, dv, None