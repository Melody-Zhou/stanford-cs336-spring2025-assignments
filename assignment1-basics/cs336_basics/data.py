import torch
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
        (inputs, targets): both are torch.LongTensor of shape (batch_size, context_length),
        placed on the specified device.
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
    
    # Create a CPU tensor view of the dataset, Cast to a supported dtype for CPU advanced indexing
    x = torch.from_numpy(dataset).to(torch.long)

    # Sample start indices on cpu
    max_start = n - context_length - 1
    starts = torch.randint(low=0, high=max_start + 1, size=(batch_size,), device="cpu")

    # Build positions [batch_size, context_length] via broadcasting
    offsets = torch.arange(context_length, device="cpu")
    pos = starts.unsqueeze(1) + offsets.unsqueeze(0)  # (B, S)

    inputs = x[pos]
    targets = x[pos + 1]

    # Ensure dtype is int64 as token IDs, and move to the target device
    inputs = inputs.to(dtype=torch.long, device=device)
    targets = targets.to(dtype=torch.long, device=device)
    return inputs, targets