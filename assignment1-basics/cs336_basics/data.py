import torch
import numpy as np
import numpy.typing as npt

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample language modeling batches from a 1D numpy array of token IDs.

    Args:
        dataset: 1D numpy array (or memmap) of integer token IDs.
        batch_size: Number of sequences to sample.
        context_length: Length of each input/target sequence.
        device: PyTorch device string, e.g. "cpu", "cuda:0", "mps".

    Returns:
        inputs:  LongTensor of shape (batch_size, context_length)
        targets: LongTensor of shape (batch_size, context_length)
    """

    if dataset.ndim != 1:
        raise ValueError(f"dataset must be 1D, got shape {dataset.shape}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")

    n = int(dataset.shape[0])
    if n < context_length + 1:
        raise ValueError(f"dataset too small: need at least context_length+1 tokens, got n={n}, context_length={context_length}")    
    
    # Sample start indices on cpu
    max_start = n - context_length - 1
    starts = np.random.randint(low=0, high=max_start+1, size=(batch_size,), dtype=np.int64)

    # Build index matrix of shape (batch_size, context_length + 1)
    offsets = np.arange(context_length + 1, dtype=np.int64)
    indices = starts[:, None] + offsets[None, :]

    # Gather a small contiguous block from the dataset
    block = dataset[indices]  # (B, S+1)

    # Split into inputs and targets
    inputs_np = block[:, :-1]
    targets_np = block[:, 1:]

    # Convert only the small batch to torch tensors and move to target device
    inputs = torch.from_numpy(inputs_np).to(device=device, dtype=torch.long)
    targets = torch.from_numpy(targets_np).to(device=device, dtype=torch.long)
    
    return inputs, targets