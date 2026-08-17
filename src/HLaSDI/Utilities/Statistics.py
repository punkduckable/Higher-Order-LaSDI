import  torch;



# -------------------------------------------------------------------------------------------------
# Initialize logger
# -------------------------------------------------------------------------------------------------

def tensor_statistics(prefix: str, values: torch.Tensor) -> dict[str, torch.Tensor]:
    """Summarize tensor values with scalar diagnostics.

    Arguments
    ---------
    prefix : str
        Metric-name prefix. Summary suffixes ``mean``, ``std``, ``min``, and ``max`` are appended
        to this prefix.
    values : torch.Tensor [...]
        Numeric values to summarize. Values are detached before conversion to Python floats so
        diagnostics never contribute to gradients.

    Returns
    -------
    metrics : dict[str, torch.Tensor]
        Detached scalar summary tensors computed over all entries of ``values``.
    """

    # Flatten tensor.
    flat = values.detach().to(dtype=torch.float32).reshape(-1)

    # If the tensor is empty, set all statistics to zero.
    if flat.numel() == 0:
        return {
            f"{prefix}/mean": values.new_zeros(()).detach(),
            f"{prefix}/std": values.new_zeros(()).detach(),
            f"{prefix}/min": values.new_zeros(()).detach(),
            f"{prefix}/max": values.new_zeros(()).detach(),
        }

    # Compute and package statistics.
    return {
        f"{prefix}/mean": flat.mean().detach(),
        f"{prefix}/std": flat.std(unbiased=False).detach(),
        f"{prefix}/min": flat.min().detach(),
        f"{prefix}/max": flat.max().detach(),
    }
