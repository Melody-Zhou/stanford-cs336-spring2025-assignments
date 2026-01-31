from typing import List

import torch
import torch.distributed as dist


class DDPIndividualParameters(torch.nn.Module):
    """
    Overlap DDP (individual parameter gradients):
      - __init__: broadcast parameters from rank0, then register post-accumulate grad hooks
      - backward: as each parameter's grad becomes ready, launch async all_reduce(grad)
      - finish_gradient_synchronization: wait all handles, then average grads
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("Process group not initialized; call dist.init_process_group first.")
        
        self.module = module
        self.world_size = dist.get_world_size()

        # Handles returned by async all_reduce
        self._handles: List[dist.Work] = []
        # Grads that correspond to handles
        self._grads: List[torch.Tensor] = []

        # Make sure all ranks start from the same weights
        self._broadcast_from_rank0()

        # Register hooks to overlap comm with backprop compute
        self._register_grad_ready_hooks()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    @torch.no_grad()
    def _broadcast_from_rank0(self) -> None:
        # Broadcast parameters
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
        
        # Also broadcast buffers (safe default for general modules)
        for b in self.module.buffers():
            if b is not None and torch.is_tensor(b):
                dist.broadcast(b, src=0)

    def _register_grad_ready_hooks(self) -> None:
        """
        Use register_post_accumulate_grad_hook if available (preferred),
        otherwise fail back to register_hook.
        """
        for p in self.module.parameters():
            if not p.requires_grad:
                continue

            # Capture parameter with a factory to avoid late-binding closure bug
            def _make_hook(param: torch.Tensor):
                # Preferred API (fires after grad accumulation)
                if hasattr(param, "register_post_accumulate_grad_hook"):
                    def _hook(_):
                        g = param.grad
                        if g is None:
                            return
                        h = dist.all_reduce(g, op=dist.ReduceOp.SUM, async_op=True)
                        self._handles.append(h)
                        self._grads.append(g)
                    return _hook, "post_accumulate"
                else:
                    # Fallback: register_hook receives grad tensor
                    def _hook(grad: torch.Tensor):
                        h = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)
                        self._handles.append(h)
                        self._grads.append(grad)
                        return grad
                    return _hook, "hook"

            hook_fn, kind = _make_hook(p)
            if kind == "post_accumulate":
                p.register_post_accumulate_grad_hook(hook_fn)
            else:
                p.register_hook(hook_fn)

    def finish_gradient_synchronization(self) -> None:
        """
        Must be called after loss.backward() and before optimizer.step().
        Ensures all async all_reduce ops are completed/queued, then averages grads.
        """
        # Wait all outstanding async ops
        for h in self._handles:
            h.wait()

        # Average grads after reduction finished (avoid racing async ops)
        if self.world_size > 1:
            for g in self._grads:
                g.div_(self.world_size)
        
        # Clear for next iteration
        self._handles.clear()
        self._grads.clear()