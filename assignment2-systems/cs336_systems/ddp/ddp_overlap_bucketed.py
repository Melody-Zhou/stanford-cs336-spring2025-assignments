from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist


@dataclass
class _ParamSlice:
    bucket_id: int
    offset: int
    numel: int
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device


class DDPBucketed(torch.nn.Module):
    """
    Bucketed + Overlap DDP:
      - Broadcast params/buffers from rank0 on init.
      - Bucket parameters (reverse order) by size <= bucket_size_mb
      - For each param, register post-accumulate hook:
          * copy grad into bucket flat buffer slice
          * replace param.grad with view into bucket buffer slice
          * when a bucket is fully ready, launch async all_reduce on bucket flat buffer
      - finish_gradient_synchronization(): wait all bucket handles and divide by world_size
    """

    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("Process group not initialized.")
        
        self.module = module
        self.world_size = dist.get_world_size()

        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        if self.bucket_size_bytes <= 0:
            raise ValueError("bucket_size_mb must be > 0")
        
        # Buckets: list of flat buffers + param ids in each bucket
        self._bucket_params: List[List[torch.nn.Parameter]] = []
        self._bucket_flats: List[Optional[torch.Tensor]] = []  # allocated lazily (first backward)
        self._bucket_handles: List[Optional[dist.Work]] = []
        self._bucket_pending: List[int] = []
        self._bucket_ready: List[List[bool]] = []  # per-bucket per-param ready flags
        self._pindex: Dict[int, Tuple[int, int]] = {}  # param_id -> (bucket_id, index_in_bucket)

        # Mapping id(param) -> slice info
        self._pslice: Dict[int, _ParamSlice] = {}

        # 1) Broadcast initial states
        self._broadcast_from_rank0()

        # 2) Build buckets (reverse order)
        self._build_buckets()

        # 3) Register hooks
        self._register_hooks()

        # 4) Initialize per-step state
        self.on_train_batch_start()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    @torch.no_grad()
    def _broadcast_from_rank0(self) -> None:
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
        for b in self.module.buffers():
            if b is not None and torch.is_tensor(b):
                dist.broadcast(b, src=0)
    
    def _build_buckets(self) -> None:
        params = [p for p in self.module.parameters() if p.requires_grad]

        # reverse order suggested by prompt
        params_rev = list(reversed(params))

        buckets: List[List[torch.nn.Parameter]] = []
        cur: List[torch.nn.Parameter] = []
        cur_bytes = 0

        def p_bytes(p: torch.nn.Parameter) -> int:
            # element_size may depend on dtype
            return p.numel() * p.element_size()

        for p in params_rev:
            pb = p_bytes(p)
            # If adding p would exceed bucket limit, start a new bucket (if current not empty)
            if cur and (cur_bytes + pb > self.bucket_size_bytes):
                buckets.append(cur)
                cur = []
                cur_bytes = 0
            cur.append(p)
            cur_bytes += pb
        
        if cur:
            buckets.append(cur)

        self._bucket_params = buckets
        self._pindex.clear()
        self._bucket_flats = [None for _ in buckets]
        self._bucket_handles = [None for _ in buckets]

        # Precompute slices (offset in elements, not bytes) per param
        for b_id, bps in enumerate(self._bucket_params):
            offset = 0
            for i, p in enumerate(bps):
                self._pindex[id(p)] = (b_id, i)
                pid = id(p)
                self._pslice[pid] = _ParamSlice(
                    bucket_id=b_id,
                    offset=offset,
                    numel=p.numel(),
                    shape=p.shape,
                    dtype=p.dtype,
                    device=p.device,
                )
                offset += p.numel()
    
    def _ensure_bucket_flat(self, bucket_id: int, ref_grad: torch.Tensor) -> None:
        """
        Allocate bucket flat buffer on first use, using grad's dtype/device
        """
        if self._bucket_flats[bucket_id] is not None:
            return
        # total numel in this bucket
        total = 0
        for p in self._bucket_params[bucket_id]:
            total += p.numel()
        self._bucket_flats[bucket_id] = torch.empty(
            (total,),
            device=ref_grad.device,
            dtype=ref_grad.dtype,
        )

    def _register_hooks(self) -> None:
        for p in self.module.parameters():
            if not p.requires_grad:
                continue

            pid = id(p)
            if pid not in self._pslice:
                continue

            def _make_hook(param: torch.nn.Parameter):
                param_id = id(param)

                def _hook(_):
                    g = param.grad
                    if g is None:
                        return
                    
                    sl = self._pslice[param_id]
                    b_id = sl.bucket_id

                    # Allocate bucket flat on first ready grad
                    self._ensure_bucket_flat(b_id, g)
                    flat = self._bucket_flats[b_id]
                    assert flat is not None

                    # View into flat slice
                    start = sl.offset
                    end = start + sl.numel
                    view = flat[start:end].view(sl.shape)

                    # Copy grad into bucket storage and redirect param.grad to this view
                    view.copy_(g)
                    param.grad = view  # optimizer will read from bucket storage

                    # Mark ready
                    _, idx_in_bucket = self._pindex[param_id]
                    if not self._bucket_ready[b_id][idx_in_bucket]:
                        self._bucket_ready[b_id][idx_in_bucket] = True
                        self._bucket_pending[b_id] -= 1
                    
                    # If bucket fully ready, launch async all-reduce now (overlap!)
                    if self._bucket_pending[b_id] == 0 and self._bucket_handles[b_id] is None:
                        h = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)
                        self._bucket_handles[b_id] = h
                
                return _hook

            # Preferred API
            if hasattr(p, "register_post_accumulate_grad_hook"):
                p.register_post_accumulate_grad_hook(_make_hook(p))
            else:
                # Fallback: grad hook receives grad tensor
                def _fallback_hook(grad: torch.Tensor, param=p):
                    # emulate post-accumulate behavior
                    if grad is None:
                        return grad
                    # temporarily set param.grad to grad then call the same logic
                    param.grad = grad
                    _make_hook(param)(None)
                    return param.grad
                p.register_hook(_fallback_hook)

    def on_train_batch_start(self) -> None:
        """
        Optional hook: reset per-step state.
        """
        self._bucket_handles = [None for _ in self._bucket_params]
        self._bucket_pending = [len(bps) for bps in self._bucket_params]
        self._bucket_ready = [[False for _ in bps] for bps in self._bucket_params]

    def on_after_backward(self) -> None:
        """
        Optional hook: ensure buckets don't get stuck if some params have None grads.
        """
        # Fill missing grads as zeros into bucket flats
        for b_id, bps in enumerate(self._bucket_params):
            if self._bucket_flats[b_id] is None:
                # allocate based on first param
                p0 = bps[0]
                self._bucket_flats[b_id] = torch.zeros(
                    (sum(p.numel() for p in bps),),
                    device=p0.device,
                    dtype=p0.dtype,
                )
            
            flat = self._bucket_flats[b_id]
            assert flat is not None

            for i, p in enumerate(bps):
                if self._bucket_ready[b_id][i]:
                    continue
                # If grad is None, set zeros view and mark ready
                sl = self._pslice[id(p)]
                start = sl.offset
                end = start + sl.numel
                view = flat[start:end].view(sl.shape)
                if p.grad is None:
                    view.zero_()
                    p.grad = view
                else:
                    view.copy_(p.grad)
                    p.grad = view
                self._bucket_ready[b_id][i] = True
                self._bucket_pending[b_id] -= 1
            
            # Launch if now ready and not launched 
            if self._bucket_pending[b_id] == 0 and self._bucket_handles[b_id] is None:
                self._bucket_handles[b_id] = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)
            
    def finish_gradient_synchronization(self) -> None:
        """
        Wait for all async all-reduce ops to be queued/completed, then average grads.
        Must be called after backward (or after on_after_backward) and before optimizer.step().
        """
        # Ensure buckets won't deadlock due to None grads (safe default)
        self.on_after_backward()

        # Wait all handles
        for h in self._bucket_handles:
            if h is not None:
                h.wait()

        # Average
        if self.world_size > 1:
            for flat in self._bucket_flats:
                if flat is not None:
                    flat.div_(self.world_size)

        self.on_train_batch_start()