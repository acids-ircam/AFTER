"""Shared helpers used across after_scripts CLIs."""


def resolve_device(device_spec=None, gpu_index=None):
    """Return a torch device string.

    Priority:
      1. ``device_spec`` if provided ("cpu", "cuda", "cuda:N", "mps", "auto").
      2. ``gpu_index`` (legacy ``--gpu`` int): >=0 → "cuda:N", else "cpu".

    "auto" picks cuda > mps > cpu based on what torch reports as available.
    """
    if device_spec is not None and device_spec != "":
        if device_spec == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device_spec

    if gpu_index is not None and gpu_index >= 0:
        return f"cuda:{gpu_index}"
    return "cpu"
