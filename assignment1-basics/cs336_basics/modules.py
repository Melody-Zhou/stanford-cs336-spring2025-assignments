import math
import torch
from torch import nn

class Linear(nn.Module):
    """
    A bias-free Linear layer that matches torch.nn.Linear's interface
    (except it has no bias)
    Stores weight as W with shape (out_features, in_features)
    """

    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # Store W (NOT W^T): shape (d_out, d_in)
        self.weight = nn.Parameter(
            torch.empty((self.out_features, self.in_features), device=device, dtype=dtype)
        )

        # Init: N(0, 2/(d_in+d_out)), truncated to [-3σ,3σ]
        sigma = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_in) -> (..., d_out)
        # Because weight is (d_out, d_in), we need x @ weight.T.
        # Use einsum to make the intended dims explicit.
        return torch.einsum("... i, o i -> ... o", x, self.weight)
    
class Embedding(nn.Module):
    """
    A learnable embedding lookup table, equivalent to torch.nn.Embedding

    This module maps integer token IDs to continuous vectors of fixed
    dimensionality (embedding_dim). The embedding matrix is stored as a
    learnable parameter of shape (num_embeddings, embedding_dim).    
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)

        # Weight matrix with shape (vocab_size, d_model)
        self.weight = nn.Parameter(
            torch.empty((self.num_embeddings, self.embedding_dim), device=device, dtype=dtype)
        )

        # Init: N(0, 1), truncated to [-3, 3]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (...) int -> output: (..., d_model)
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    For an input vector a in R^{d_model}:
        RMS(a) = sqrt(mean(a^2) + eps)
        RMSNorm(a) = (a / RMS(a)) * g    
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = int(d_model)
        self.eps = float(eps)

        # Learnable gain parameter (g), shape (d_model,)
        self.weight = nn.Parameter(torch.ones((self.d_model,), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_fp32 = x.to(torch.float32)

        # Compute RMS over the last dimension: sqrt(mean(x^2) + eps)
        rms = torch.sqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # Normalize and apply gain; do match in float32 then cast back
        y = (x_fp32 / rms) * self.weight.to(torch.float32)

        return y.to(in_dtype)
    
def round_up_to_multiple(x: int, multiple: int) -> int:
    """Round x up to the nearest positive multiple of `multiple`."""
    if multiple <= 0:
        raise ValueError("multiple must be a positive integer")
    return int(((x + multiple - 1) // multiple) * multiple)

def default_d_ff(d_model: int, multiple_of: int = 64) -> int:
    """
    Compute the recommended SwiGLU hidden size.
    
    We use d_ff ~= (8/3) * d_model and then round up to a hardware-friendly multiple (typically 64).
    """
    raw = int(math.ceil((8.0 * d_model) / 3.0))
    return round_up_to_multiple(raw, multiple_of)

class SwiGLU(nn.Module):
    """
    Position-wise feed-forward network using the SwiGLU nonlinearity.

    The transformation is:
        FFN(x) = W2( SiLU(W1 x) ⊙ (W3 x) )

    where SiLU(z) = z * sigmoid(z), and ⊙ is elementwise multiplication.

    Shapes:
        input:  (..., d_model)
        W1, W3: (d_ff, d_model)   implemented as Linear(d_model -> d_ff)
        W2:     (d_model, d_ff)   implemented as Linear(d_ff -> d_model)
        output: (..., d_model)
    """

    def __init__(
        self, d_model: int, d_ff: int | None = None, *, multiple_of: int = 64,
        device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_ff = int(d_ff) if d_ff is not None else default_d_ff(self.d_model, multiple_of)

        # Two up-prpjections and one down-projection (no bias)
        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)

    @staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.w1(x)
        b = self.w3(x)
        gated = self.silu(a) * b
        return self.w2(gated)
    
class RoPE(nn.Module):
    """
    Rotary Positional Embeddings (RoPE).

    Applies a position-dependent rotation to the last dimension (d_k) of an input tensor.
    The rotation is applied pairwise on (x[..., 0], x[..., 1], x[..., 2], x[..., 3]), ...

    This module has no learnable parameters. It can precompute and cache cos/sin table.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even for RoPE, got d_k={d_k}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        
        self.theta = float(theta)
        self.d_k = int(d_k)
        self.max_seq_len = int(max_seq_len)

        # Precompute inverse frequencies for even indices:
        # inv_freq[j] = theta^(-2j/d_k), where j indexes pairs (0, 1, ..., d_k/2 - 1).
        pair_idx = torch.arange(0, self.d_k, 2, device=device, dtype=torch.float32)
        inv_freq = self.theta ** (-pair_idx / self.d_k)

        # Positions [0, 1, ..., max_seq_len-1]
        positions = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)

        # Angles: (max_seq_len, d_k/2)
        angles = positions[:, None] * inv_freq[None, :]

        cos = torch.cos(angles)  # (max_seq_len, d_k/2)
        sin = torch.sin(angles)  # (max_seq_len, d_k/2)

        # Cache as non-persistent buffers (not saved in state_dict)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (..., seq_len, d_k)
            token_positions: Tensor of shape (..., seq_len) with integer positions

        Returns:
            Tensor of shape (..., seq_len, d_k) after applying RoPE.
        """
        if x.size(-1) != self.d_k:
            raise ValueError(f"Expected x.size(-1)==d_k=={self.d_k}, got {x.size(-1)}")        
        
        # token_positions is used to slice cached cos/sin along the sequence of dimension.
        # Shapes after indexing: (..., seq_len, d_k/2)
        pos = token_positions.to(device=x.device)
        cos = self.cos.index_select(0, pos.reshape(-1)).reshape(*pos.shape, -1)
        sin = self.sin.index_select(0, pos.reshape(-1)).reshape(*pos.shape, -1)

        # Promote to float32 for numerical stability, then cast back
        x_fp32 = x.to(torch.float32)
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)

        x_even = x_fp32[..., ::2]  # (..., seq_len, d_k/2)
        x_odd = x_fp32[..., 1::2]  # (..., seq_len, d_k/2)

        # make cos/sin broadcastable for inputs like (B, H, S, d_k)
        while cos.dim() < x_even.dim():
            cos = cos.unsqueeze(cos.dim() - 2)
            sin = sin.unsqueeze(sin.dim() - 2)        

        # Apply 2D rotation for each pair.
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        # Interleave even/odd back to (..., seq_len, d_k)
        out = torch.stack((out_even, out_odd), dim=-1).flatten(-2)

        return out.to(dtype=x.dtype)
    
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Numerically stable softmax over a given dimension.

    This implementation subtracts the maximum value along `dim` before
    exponentiation to improve numerical stability.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension over which to apply softmax.

    Returns:
        torch.Tensor: Softmax output with the same shape/dtype/device as `x`.
    """
    # Subtract max for numerical stability (keepdim for correct broadcasting)
    x_max = torch.amax(x, dim=dim, keepdim=True)
    z = x - x_max

    exp_z = torch.exp(z)
    sum_exp = torch.sum(exp_z, dim=dim, keepdim=True)

    return exp_z / sum_exp

def scaled_dot_product_attention(
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.tensor, 
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Scaled dot-product attention.

    Args:
        query: Tensor of shape (..., seq_len, d_k)
        key:   Tensor of shape (..., seq_len, d_k)
        value: Tensor of shape (..., seq_len, d_v)
        mask:  Optional bool tensor of shape (seq_len, seq_len), where True means
               the position is allowed and False means it is masked out.

    Returns:
        Tensor of shape (..., seq_len, d_v)
    """
    if query.dim() < 2 or key.dim() < 2 or value.dim() < 2:
        raise ValueError("query/key/value must have shape (..., seq_len, d_*)")

    if query.shape[:-2] != key.shape[:-2] or query.shape[:-2] != value.shape[:-2]:
            raise ValueError("batch dimensions of query, key, value must match")

    d_k = query.shape[-1]
    if d_k != key.shape[-1]:
        raise ValueError("query and key must have the same d_k")
    
    # Compute attention logits in float32 for stability
    q = query.to(torch.float32)    
    k = key.to(torch.float32)
    v = value.to(torch.float32)

    scale = 1.0 / math.sqrt(d_k)

    # logits: (..., seq_len, seq_len)
    logits = torch.einsum("... s d, ... t d -> ... s t", q, k) * scale
    
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("mask must be a boolean tensor")

        # Broadcast mask to logits shape: (..., seq_len, seq_len)
        # True = keep, False = mask out.
        neg_inf = torch.finfo(torch.float32).min
        logits = torch.where(mask.to(device=logits.device), logits, neg_inf)        
    
    # probs: (..., seq_len, seq_len)
    probs = softmax(logits, dim=-1)

    if mask is not None:
        # Ensure exact zeros on masked positions (softmax(-inf) should already be 0,
        # but this makes behavior robust under extreme values).
        probs = probs * mask.to(device=probs.device, dtype=probs.dtype)
    
    # out: (..., seq_len, d_v)
    out = torch.einsum("... s t, ... t d -> ... s d", probs, v)

    # Cast back to the original value dtype
    return out.to(dtype=value.dtype)

class CausalMultiHeadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention (no RoPE).

    This module computes:
        Q = W_Q x, K = W_K x, V = W_V x
        heads = SDPA(Q_heads, K_heads, V_heads, causal_mask)
        out = W_O concat(heads)

    Shapes:
        x:   (..., seq_len, d_model)
        QKV: (..., seq_len, d_model)
        heads view: (..., num_heads, seq_len, head_dim)
        output: (..., seq_len, d_model)
    """
    def __init__(self, d_model: int, num_heads: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        
        self.head_dim = self.d_model // self.num_heads  # d_k = d_v = d_model / h

        # Separate projections (one matmul each). Combining into one is an optional optimization
        self.q_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.k_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.v_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.o_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Build a (seq_len, seq_len) causal mask where True means "allowed"
        """
        return torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (..., seq_len, d_model)

        Returns:
            Tensor of shape (..., seq_len, d_model)
        """
        if x.size(-1) != self.d_model:
            raise ValueError(f"Expected last dim {self.d_model}, got {x.size(-1)}")

        seq_len = x.size(-2)    
        device = x.device

        # Project to Q, K, V: (..., seq_len, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape into heads: (..., seq_len, num_heads, head_dim)
        # Then move heads into a batch-like dimension: (..., num_heads, seq_len, head_dim)
        new_shape = q.shape[:-1] + (self.num_heads, self.head_dim)
        q = q.view(new_shape).transpose(-3, -2)
        k = k.view(new_shape).transpose(-3, -2)
        v = v.view(new_shape).transpose(-3, -2)

        # Causal mask shared across heads and batches
        mask = self._causal_mask(seq_len, device=device)

        # SDPA: (..., num_heads, seq_len, head_dim)
        out = scaled_dot_product_attention(q, k, v, mask=mask)

        # Merge heads: (..., seq_len, d_model)
        out = out.transpose(-3, -2).contiguous().view(x.shape[:-1] + (self.d_model,))

        # Output projection: (..., seq_len, d_model)
        return self.o_proj(out)

class CausalMultiHeadSelfAttentionWithRoPE(nn.Module):
    """
    Causal multi-head self-attention with RoPE applied to Q and K (not V).

    This version uses a fused QKV projection:
        qkv = W_qkv x
        q, k, v = split(qkv)
    """
    def __init__(
        self, d_model: int, num_heads: int, theta: float, max_seq_len: int,
        device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)

        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.head_dim = self.d_model // self.num_heads

        # Separate projections (matches reference state_dict keys).
        self.q_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.k_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.v_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.output_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)

        # RoPE operates on per-head dimension
        self.rope = RoPE(theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len, device=device)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Build a (seq_len, seq_len) causal mask where True means 'allowed'."""
        return torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (..., seq_len, d_model)
            token_positions: Tensor of shape (..., seq_len)

        Returns:
            Tensor of shape (..., seq_len, d_model)
        """
        if x.size(-1) != self.d_model:
            raise ValueError(f"Expected last dim {self.d_model}, got {x.size(-1)}")

        seq_len = x.size(-2)    
        device = x.device

        # Project to Q, K, V: (..., seq_len, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape into heads: (..., seq_len, num_heads, head_dim)
        # Then transpose to (..., num_heads, seq_len, head_dim)
        new_shape = q.shape[:-1] + (self.num_heads, self.head_dim)
        q = q.view(new_shape).transpose(-3, -2)
        k = k.view(new_shape).transpose(-3, -2)
        v = v.view(new_shape).transpose(-3, -2)

        # Apply RoPE to Q and K for each head (heads are treated as batch-like dims)
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        # Causal mask shared across heads and batches
        mask = self._causal_mask(seq_len, device=device)

        # Attention: (..., num_heads, seq_len, head_dim)
        out = scaled_dot_product_attention(q, k, v, mask=mask)

        # Merge heads back: (..., seq_len, d_model)
        out = out.transpose(-3, -2).contiguous().view(x.shape[:-1] + (self.d_model,))

        return self.output_proj(out)
    
class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block.

    Structure (pre-norm):
        y = x + Attn(RMSNorm(x))
        z = y + FFN(RMSNorm(y))
    
    This block uses causal multi-head self-attention with RoPE
    """

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, *,
        max_seq_len: int, theta: float, eps: float = 1e-5,
        device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)

        self.ln1 = RMSNorm(self.d_model, eps=eps, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttentionWithRoPE(
            d_model=self.d_model,
            num_heads=self.num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype
        )
    
        self.ln2 = RMSNorm(self.d_model, eps=eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(self.d_model, d_ff=self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
            token_positions: Tensor of shape (batch, seq_len) or broadcastable to it

        Returns:
            Tensor of shape (batch, seq_len, d_model)
        """
        # Pre-norm attention + residual
        h = self.ln1(x)
        x = x + self.attn(h, token_positions)

        # Pre-norm FFN + residual
        h = self.ln2(x)
        x = x + self.ffn(h)

        return x