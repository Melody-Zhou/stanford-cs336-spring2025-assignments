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
    IS_CAUSAL: tl.constexpr,
):
    # program ids
    pid_q = tl.program_id(0)  # query tile id
    pid_b = tl.program_id(1)  # batch id

    # block pointers for Q/K/V/O/L tiles
    # block pointers encapsulate base, shape, strides, and OOB handling;
    # advancing them avoids re-materializing raw pointer arithmetic in the loop
    Qb = Q_ptr + pid_b * stride_qb
    Kb = K_ptr + pid_b * stride_kb
    Vb = V_ptr + pid_b * stride_vb
    Ob = O_ptr + pid_b * stride_ob
    Lb = L_ptr + pid_b * stride_lb

    Q_bp = tl.make_block_ptr(
        base=Qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(pid_q * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_bp = tl.make_block_ptr(
        base=Kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_bp = tl.make_block_ptr(
        base=Vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_bp = tl.make_block_ptr(
        base=Ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(pid_q * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_bp = tl.make_block_ptr(
        base=Lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(pid_q * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # load the Q tile
    # keep the original dtype for the final store cast
    q_raw = tl.load(Q_bp, boundary_check=(0, 1), padding_option="zero")
    q = q_raw.to(tl.float32)

    # running state (on-chip)
    m = tl.full((Q_TILE_SIZE,), -float("inf"), tl.float32)  # [Bq]
    l = tl.zeros((Q_TILE_SIZE,), tl.float32)                # [Bq]
    acc = tl.zeros((Q_TILE_SIZE, D), tl.float32)            # [Bq, D]

    # iterate over K/V tile by advancing block pointers (instead of re-building raw pointers)
    K_it = K_bp
    V_it = V_bp

    # absolute query indices used only for causal masking
    q_abs = pid_q * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    if IS_CAUSAL:
        # Phase 1: Non-diagonal tiles (fully inside the causal mask)
        # Iterate over K tiles [0, pid_q * Q_TILE_SIZE]
        # These tiles are completely visible to the current Q tile, so no mask is needed.
        limit_nonding = min(N_KEYS, pid_q * Q_TILE_SIZE)
        for _ in range(0, limit_nonding, K_TILE_SIZE):
            k = tl.load(K_it, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
            v = tl.load(V_it, boundary_check=(0, 1), padding_option="zero")

            # S = q @ k^T * scale -> [Bq, Bk]
            S = tl.dot(q, tl.trans(k)) * scale  # float32

            # online softmax update (no causal mask check here)
            m_new = tl.maximum(m, tl.max(S, axis=1))  # [Bq]
            p = tl.exp(S - m_new[:, None])            # [Bq, Bk]
            alpha = tl.exp(m - m_new)                 # [Bq]
            l_new = alpha * l + tl.sum(p, axis=1)     # [Bq]

            # acc = alpha * acc + p @ v
            p = p.to(v.dtype)
            acc = alpha[:, None] * acc
            acc = tl.dot(p, v, acc=acc)
            m = m_new
            l = l_new            

            # advance K/V block pointers
            K_it = K_it.advance((K_TILE_SIZE, 0))
            V_it = V_it.advance((K_TILE_SIZE, 0))

        # Phase 2: Diagonal tile (partial causal mask)
        # Only process if the diagonal tile exists (i.e., within N_KEYS)
        if pid_q * Q_TILE_SIZE < N_KEYS:
            k = tl.load(K_it, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
            v = tl.load(V_it, boundary_check=(0, 1), padding_option="zero")
            S = tl.dot(q, tl.trans(k)) * scale

            # causal mask: keep if q_idx >= k_idx else -1e-6
            k_abs = pid_q * Q_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            S = tl.where(q_abs[:, None] >= k_abs[None, :], S, -1.0e6)

            # online softmax update
            m_new = tl.maximum(m, tl.max(S, axis=1))
            p = tl.exp(S - m_new[:, None])
            alpha = tl.exp(m - m_new)
            l_new = alpha * l + tl.sum(p, axis=1)

            # acc = alpha * acc + p @ v
            p = p.to(v.dtype)
            acc = alpha[:, None] * acc
            acc = tl.dot(p, v, acc=acc)
            m = m_new
            l = l_new
            # No need to advace further, as tiles > pid_q are fully masked out
    else:
        # Non-causal path: process all tiles
        for _ in range(0, N_KEYS, K_TILE_SIZE):
            # load one (K_TILE_SIZE, D) tile of K and V
            k = tl.load(K_it, boundary_check=(0, 1), padding_option="zero").to(tl.float32)  # [Bk, D]
            v = tl.load(V_it, boundary_check=(0, 1), padding_option="zero")                 # [Bk, D]

            # S = q @ k^T * scale -> [Bq, Bk]
            S = tl.dot(q, tl.trans(k)) * scale  # float32

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

            # advance K/V block pointers to the next tile along the sequence dimension
            K_it = K_it.advance((K_TILE_SIZE, 0))
            V_it = V_it.advance((K_TILE_SIZE, 0))

    # write O and L
    # store output in the original input dtype
    o = (acc / l[:, None]).to(q_raw.dtype)

    tl.store(O_bp, o, boundary_check=(0, 1))

    L_out = m + tl.log(l)  # [Bq]
    tl.store(L_bp, L_out, boundary_check=(0,))


@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, DO_ptr,
    L_ptr,
    DQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_dqb, stride_dqq, stride_dqd,
    N_QUERIES: tl.constexpr,
    N_KEYS: tl.constexpr,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_q = tl.program_id(0)  # q tile
    pid_b = tl.program_id(1)  # batch

    # offsets for causal and reductions
    q_idx = pid_q * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)   # [Bq]

    # base pointers
    Qb  = Q_ptr + pid_b * stride_qb
    Kb  = K_ptr  + pid_b * stride_kb
    Vb  = V_ptr  + pid_b * stride_vb
    Ob  = O_ptr  + pid_b * stride_ob
    DOb = DO_ptr + pid_b * stride_dob
    Lb  = L_ptr  + pid_b * stride_lb
    DQb = DQ_ptr + pid_b * stride_dqb

    # (Q, D) block pointers for q/o/do/dq/l
    Q_bp = tl.make_block_ptr(
        Qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(pid_q * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O_bp = tl.make_block_ptr(
        Ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(pid_q * Q_TILE_SIZE, 0), 
        block_shape=(Q_TILE_SIZE, D), 
        order=(1, 0),
    )
    DO_bp = tl.make_block_ptr(
        DOb, 
        shape=(N_QUERIES, D), 
        strides=(stride_doq, stride_dod),
        offsets=(pid_q * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    DQ_bp = tl.make_block_ptr(
        DQb, 
        shape=(N_QUERIES, D), 
        strides=(stride_dqq, stride_dqd),
        offsets=(pid_q * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_bp = tl.make_block_ptr(
        base=Lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(pid_q * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # load Q, dO, O, L
    # keep the original dtype for the final store cast
    q_raw = tl.load(Q_bp,  boundary_check=(0, 1), padding_option="zero")
    q  = q_raw.to(tl.float32)
    do = tl.load(DO_bp, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    o  = tl.load(O_bp,  boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    L  = tl.load(L_bp,  boundary_check=(0,),   padding_option="zero").to(tl.float32)

    D_row = tl.sum(do * o, axis=1)  # [Bq]
    dq_acc = tl.zeros((Q_TILE_SIZE, D), tl.float32)

    # (K, D) block pointers for sweeping K/V
    K_bp = tl.make_block_ptr(
        Kb, 
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_bp = tl.make_block_ptr(
        Vb,
        shape=(N_KEYS, D),
        strides=(stride_vk,stride_vd),
        offsets=(0, 0), 
        block_shape=(K_TILE_SIZE, D), 
        order=(1, 0),
    )

    K_it = K_bp
    V_it = V_bp

    if IS_CAUSAL:
        # Phase 1: Non-diagonal K tiles [0, pid_q * K_TILE_SIZE)
        limit_nondiag = min(N_KEYS, pid_q * Q_TILE_SIZE)
        for _ in range(0, limit_nondiag, K_TILE_SIZE):
            k = tl.load(K_it, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
            v = tl.load(V_it, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

            # S: [Bq, Bk]
            S = tl.dot(q, tl.trans(k)) * scale
            
            # No mask needed for k < q
            # P = exp(S - L)
            P = tl.exp(S - L[:, None])  # [Bq, Bk]
            
            # dP = dO @ V^T
            dP = tl.dot(do, tl.trans(v))  # [Bq, Bk]
            
            # dS = P * (dP - D_row)
            dS = P * (dP - D_row[:, None])  # [Bq, Bk]
            
            # dQ += dS @ K * scale
            dq_acc += tl.dot(dS, k) * scale
            K_it = K_it.advance((K_TILE_SIZE, 0))
            V_it = V_it.advance((K_TILE_SIZE, 0))

        # Phase 2: Diagonal K tile (kb = pid_q * K_TILE_SIZE)
        if pid_q * Q_TILE_SIZE < N_KEYS:
            k = tl.load(K_it, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
            v = tl.load(V_it, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
            S = tl.dot(q, tl.trans(k)) * scale

            # Apply mask for the diagonal
            k_idx = pid_q * Q_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            S = tl.where(q_idx[:, None] >= k_idx[None, :], S, -1.0e6)
            
            # P = exp(S - L)
            P = tl.exp(S - L[:, None])

            # dP = dO @ V^T
            dP = tl.dot(do, tl.trans(v))

            # dS = P * (dP - D_row)
            dS = P * (dP - D_row[:, None])

            # dQ += dS @ K * scale
            dq_acc += tl.dot(dS, k) * scale
    else:
        # Non-causal path
        for _ in range(0, N_KEYS, K_TILE_SIZE):
            k = tl.load(K_it, boundary_check=(0, 1), padding_option="zero").to(tl.float32)  # [Bk, D]
            v = tl.load(V_it, boundary_check=(0, 1), padding_option="zero").to(tl.float32)  # [Bk, D]

            # S: [Bq, Bk]
            S = tl.dot(q, tl.trans(k)) * scale

            # P = exp(S - L)
            P = tl.exp(S - L[:, None])  # [Bq, Bk]

            # dP = dO @ V^T
            dP = tl.dot(do, tl.trans(v))  # [Bq, Bk]

            # dS = P * (dP - D_row)
            dS = P * (dP - D_row[:, None])  # [Bq, Bk]

            # dQ += dS @ K * scale
            dq_acc += tl.dot(dS, k) * scale

            # advance K/V block pointers to the next tile along the sequence dimension
            K_it = K_it.advance((K_TILE_SIZE, 0))
            V_it = V_it.advance((K_TILE_SIZE, 0))

    tl.store(DQ_bp, dq_acc.to(q_raw.dtype), boundary_check=(0, 1))


@triton.jit
def flash_bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, DO_ptr,
    L_ptr,
    DK_ptr, DV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES: tl.constexpr,
    N_KEYS: tl.constexpr,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_k = tl.program_id(0)  # k tile
    pid_b = tl.program_id(1)

    # offsets for causal and reductions
    k_idx = pid_k * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)  # [Bk]

    # base pointers (batch)
    Qb  = Q_ptr  + pid_b * stride_qb
    Kb  = K_ptr  + pid_b * stride_kb
    Vb  = V_ptr  + pid_b * stride_vb
    Ob  = O_ptr  + pid_b * stride_ob
    DOb = DO_ptr + pid_b * stride_dob
    Lb  = L_ptr  + pid_b * stride_lb
    DKb = DK_ptr + pid_b * stride_dkb
    DVb = DV_ptr + pid_b * stride_dvb

    # (K, D) block pointers for this tile
    K_bp = tl.make_block_ptr(
        Kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(pid_k * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_bp = tl.make_block_ptr(
        Vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(pid_k * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    DK_bp = tl.make_block_ptr(
        DKb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(pid_k * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    DV_bp = tl.make_block_ptr(
        DVb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(pid_k * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )    

    # keep the original dtype for the final store cast
    k_raw = tl.load(K_bp, boundary_check=(0, 1), padding_option="zero")
    v_raw = tl.load(V_bp, boundary_check=(0, 1), padding_option="zero")
    k = k_raw.to(tl.float32)
    v = v_raw.to(tl.float32)

    dk_acc = tl.zeros((K_TILE_SIZE, D), tl.float32)
    dv_acc = tl.zeros((K_TILE_SIZE, D), tl.float32)

    # iter (Q, D) pointers for sweeping Q/O/DO/L
    Q_bp = tl.make_block_ptr(
        Qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O_bp = tl.make_block_ptr(
        Ob, 
        shape=(N_QUERIES, D), 
        strides=(stride_oq, stride_od),
        offsets=(0, 0), 
        block_shape=(Q_TILE_SIZE, D), 
        order=(1, 0),
    )
    DO_bp = tl.make_block_ptr(
        DOb,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_bp = tl.make_block_ptr(
        base=Lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Q_it  = Q_bp
    O_it  = O_bp
    DO_it = DO_bp
    L_it  = L_bp

    if IS_CAUSAL:
        # Optimization: Start from the diagonal Q tile (pid_k * Q_TILE_SIZE)
        # Previous Q tiles (i < k) are masked out and contribute 0 gradient to K[k], V[k]
        start_q_offset = pid_k * Q_TILE_SIZE

        # Advance pointers to the start Q tile
        Q_it  = Q_it.advance((start_q_offset, 0))
        O_it  = O_it.advance((start_q_offset, 0))
        DO_it = DO_it.advance((start_q_offset, 0))
        L_it  = L_it.advance((start_q_offset,))
        
        # Phase 1: Diagonal Q tile (needs mask)
        if start_q_offset < N_QUERIES:
            q  = tl.load(Q_it,  boundary_check=(0, 1), padding_option="zero").to(tl.float32)
            o  = tl.load(O_it,  boundary_check=(0, 1), padding_option="zero").to(tl.float32)
            do = tl.load(DO_it, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
            L  = tl.load(L_it,  boundary_check=(0,),   padding_option="zero").to(tl.float32)
            D_row = tl.sum(do * o, axis=1)   # [Bq]

            # S: [Bq, Bk]
            S = tl.dot(q, tl.trans(k)) * scale
            
            # Mask: q_idx >= k_idx
            q_idx = start_q_offset + tl.arange(0, Q_TILE_SIZE)
            S = tl.where(q_idx[:, None] >= k_idx[None, :], S, -1.0e6)
            
            # P = exp(S - L)
            P = tl.exp(S - L[:, None])  # [Bq, Bk]
            
            # dP = dO @ V^T
            dP = tl.dot(do, tl.trans(v))  # [Bq, Bk]
            
            # dS = P * (dP - D_row)
            dS = P * (dP - D_row[:, None])  # [Bq, Bk]
            
            # dV += P^T @ dO
            dv_acc += tl.dot(tl.trans(P), do)
            
            # dK += dS^T @ Q * scale
            dk_acc += tl.dot(tl.trans(dS), q) * scale
            
            # Advance to next tile
            Q_it  = Q_it.advance((Q_TILE_SIZE, 0))
            O_it  = O_it.advance((Q_TILE_SIZE, 0))
            DO_it = DO_it.advance((Q_TILE_SIZE, 0))
            L_it  = L_it.advance((Q_TILE_SIZE,))

        # Phase 2: Non-diagonal Q tiles (qb > start_q_offset, no mask)
        for _ in range(start_q_offset + Q_TILE_SIZE, N_QUERIES, Q_TILE_SIZE):
            q  = tl.load(Q_it,  boundary_check=(0, 1), padding_option="zero").to(tl.float32)
            o  = tl.load(O_it,  boundary_check=(0, 1), padding_option="zero").to(tl.float32)
            do = tl.load(DO_it, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
            L  = tl.load(L_it,  boundary_check=(0,),   padding_option="zero").to(tl.float32)
            D_row = tl.sum(do * o, axis=1)

            # S: [Bq, Bk]
            S = tl.dot(q, tl.trans(k)) * scale
            
            # No mask needed for q > k
            # P = exp(S - L)
            P = tl.exp(S - L[:, None])

            # dP = dO @ V^T
            dP = tl.dot(do, tl.trans(v))

            # dS = P * (dP - D_row)
            dS = P * (dP - D_row[:, None])

            # dV += P^T @ dO
            dv_acc += tl.dot(tl.trans(P), do)

            # dK += dS^T @ Q * scale
            dk_acc += tl.dot(tl.trans(dS), q) * scale

            # Advance to next tile
            Q_it  = Q_it.advance((Q_TILE_SIZE, 0))
            O_it  = O_it.advance((Q_TILE_SIZE, 0))
            DO_it = DO_it.advance((Q_TILE_SIZE, 0))
            L_it  = L_it.advance((Q_TILE_SIZE,))
    else:
        # Non-causal path: standard sweep
        for _ in range(0, N_QUERIES, Q_TILE_SIZE):
            q  = tl.load(Q_it,  boundary_check=(0, 1), padding_option="zero").to(tl.float32)  # [Bq, D]
            o  = tl.load(O_it,  boundary_check=(0, 1), padding_option="zero").to(tl.float32)  # [Bq, D]
            do = tl.load(DO_it, boundary_check=(0, 1), padding_option="zero").to(tl.float32)  # [Bq, D]
            L  = tl.load(L_it,  boundary_check=(0,),   padding_option="zero").to(tl.float32)  # [Bq]

            D_row = tl.sum(do * o, axis=1)   # [Bq]

            # S: [Bq, Bk]
            S = tl.dot(q, tl.trans(k)) * scale

            # P = exp(S - L)
            P = tl.exp(S - L[:, None])  # [Bq, Bk]

            # dP = dO @ V^T
            dP = tl.dot(do, tl.trans(v))  # [Bq, Bk]

            # dS = P * (dP - D_row)
            dS = P * (dP - D_row[:, None])  # [Bq, Bk]

            # dV += P^T @ dO
            dv_acc += tl.dot(tl.trans(P), do)

            # dK += dS^T @ Q * scale
            dk_acc += tl.dot(tl.trans(dS), q) * scale

            Q_it  = Q_it.advance((Q_TILE_SIZE, 0))
            O_it  = O_it.advance((Q_TILE_SIZE, 0))
            DO_it = DO_it.advance((Q_TILE_SIZE, 0))
            L_it  = L_it.advance((Q_TILE_SIZE,))

    tl.store(DK_bp, dk_acc.to(k_raw.dtype), boundary_check=(0, 1))
    tl.store(DV_bp, dv_acc.to(v_raw.dtype), boundary_check=(0, 1))


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

    # @staticmethod
    # def backward(ctx, do):
    #     (L, q, k, v, o) = ctx.saved_tensors
    #     dq, dk, dv = flash_bwd_recompute_impl(q, k, v, o, do, L, ctx.is_causal)
    #     return dq, dk, dv, None

    @staticmethod
    def backward(ctx, do):
        (L, q, k, v, o) = ctx.saved_tensors
        B, Q, D = q.shape
        K = k.shape[1]
        scale = 1.0 / math.sqrt(D)
        is_causal = ctx.is_causal

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        Bq = 32
        Bk = 32

        grid_dq = (triton.cdiv(Q, Bq), B)
        flash_bwd_dq_kernel[grid_dq](
            q, k, v, o, do,
            L,
            dq,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            L.stride(0), L.stride(1),
            dq.stride(0), dq.stride(1), dq.stride(2),
            N_QUERIES=Q,
            N_KEYS=K,
            scale=scale,
            D=D,
            Q_TILE_SIZE=Bq,
            K_TILE_SIZE=Bk,
            IS_CAUSAL=is_causal,
            num_warps=4,
        )

        grid_dkdv = (triton.cdiv(K, Bk), B)
        flash_bwd_dkdv_kernel[grid_dkdv](
            q, k, v, o, do,
            L,
            dk, dv,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            L.stride(0), L.stride(1),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            N_QUERIES=Q,
            N_KEYS=K,
            scale=scale,
            D=D,
            Q_TILE_SIZE=Bq,
            K_TILE_SIZE=Bk,
            IS_CAUSAL=is_causal,
            num_warps=4,
        )

        return dq, dk, dv, None    