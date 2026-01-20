import math
import torch
import triton
import triton.language as tl

from cs336_systems.flash_pytorch import flash_bwd_recompute_impl

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES: tl.constexpr,
    N_KEYS: tl.constexpr,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr
):
    # program ids
    pid_q = tl.program_id(0)  # query tile id
    pid_b = tl.program_id(1)  # batch id

    # offsets
    q_offsets = pid_q * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)  # [Bq]
    d_offsets = tl.arange(0, D)  # [D]

    # pointers for Q tile: (Bq, D)
    q_ptrs = Q_ptr + pid_b * stride_qb + q_offsets[:, None] * stride_qq + d_offsets[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(q_offsets[:, None] < N_QUERIES), other=0.0).to(tl.float32)

    # running state (on-chip)
    m = tl.full((Q_TILE_SIZE,), -float("inf"), tl.float32)  # [Bq]
    l = tl.zeros((Q_TILE_SIZE,), tl.float32)                # [Bq]
    acc = tl.zeros((Q_TILE_SIZE, D), tl.float32)            # [Bq, D]

    # loop over K tiles
    for kb in tl.static_range(0, N_KEYS, K_TILE_SIZE):
        k_offsets = kb + tl.arange(0, K_TILE_SIZE)  # [Bk]

        k_ptrs = K_ptr + pid_b * stride_kb + k_offsets[:, None] * stride_kk + d_offsets[None, :] * stride_kd
        v_ptrs = V_ptr + pid_b * stride_vb + k_offsets[:, None] * stride_vk + d_offsets[None, :] * stride_vd
    
        k = tl.load(k_ptrs, mask=(k_offsets[:, None] < N_KEYS), other=0.0).to(tl.float32)  # [Bk, D]
        v = tl.load(v_ptrs, mask=(k_offsets[:, None] < N_KEYS), other=0.0)                 # [Bk, D]

        # S = q @ k^T * scale -> [Bq, Bk]
        S = tl.dot(q, tl.trans(k)) * scale  # float32

        # causal mask: keep if q_idx >= k_idx else -1e-6
        if IS_CAUSAL:
            q_abs = q_offsets[:, None]  # [Bq, 1]
            k_abs = k_offsets[None, :]  # [1, Bk]
            causal = q_abs >= k_abs
            S = tl.where(causal, S, -1.0e6)
        
        # online softmax update
        m_new = tl.maximum(m, tl.max(S, axis=1))  # [Bq]
        p = tl.exp(S - m_new[:, None])            # [Bq, Bk]

        alpha = tl.exp(m - m_new)                 # [Bq]
        l_new = alpha * l + tl.sum(p, axis=1)     # [Bq]

        # acc = alpha * acc + p @ v
        # p needs to match v dtype before dot
        p = p.to(v.dtype)
        acc = alpha[:, None] * acc
        acc = tl.dot(p, v, acc=acc)

        m = m_new
        l = l_new
    
    # write O and L
    o = acc / l[:, None]
    o = o.to(tl.float32)

    o_ptrs = O_ptr + pid_b * stride_ob + q_offsets[:, None] * stride_oq + d_offsets[None, :] * stride_od
    tl.store(o_ptrs, o, mask=(q_offsets[:, None] < N_QUERIES))

    L_out = m + tl.log(l)  # [Bq]
    l_ptrs = L_ptr + pid_b * stride_lb + q_offsets * stride_lq
    tl.store(l_ptrs, L_out, mask=(q_offsets < N_QUERIES))


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
        # Expect (B, Q, D), (B, K, D), (B, K, D)
        if not q.is_cuda:
            raise RuntimeError("Triton implementation requires CUDA tensors")
        if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
            raise ValueError("Expected q/k/v to be 3D: (B, N, D)")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError("Batch size mismatch")
        if k.shape[1] != v.shape[1] or k.shape[2] != v.shape[2]:
            raise ValueError("k/v shape mismatch")
        if q.shape[2] != k.shape[2]:
            raise ValueError("q/k D mismatch")

        B, Q, D = q.shape
        K = k.shape[1]
        scale = 1.0 / math.sqrt(D)

        # tile sizes
        Bq = 32
        Bk = 32

        # outputs
        o = torch.empty((B, Q, D), device=q.device, dtype=q.dtype)
        L = torch.empty((B, Q), device=q.device, dtype=torch.float32)

        grid = (triton.cdiv(Q, Bq), B)

        flash_fwd_kernel[grid](
            q, k, v,
            o, L,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES=Q,
            N_KEYS=K,
            scale=scale,
            D=D,
            Q_TILE_SIZE=Bq,
            K_TILE_SIZE=Bk,
            IS_CAUSAL=is_causal,
            num_warps=4
        )

        # save for backward
        ctx.save_for_backward(L, q, k, v, o)
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx, do):
        (L, q, k, v, o) = ctx.saved_tensors
        dq, dk, dv = flash_bwd_recompute_impl(q, k, v, o, do, L, ctx.is_causal)
        return dq, dk, dv, None