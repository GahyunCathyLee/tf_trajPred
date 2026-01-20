#src/debug/debug.py
import torch
import torch.nn as nn

def _finite_stats(x: torch.Tensor) -> str:
    x_f = x[torch.isfinite(x)]
    if x_f.numel() == 0:
        return "all_nonfinite"
    return f"min={x_f.min().item():.3e} max={x_f.max().item():.3e}"

def _any_nonfinite(x: torch.Tensor) -> bool:
    return not torch.isfinite(x).all()

def _check_params_finite(model: nn.Module) -> bool:
    for p in model.parameters():
        if p is None:
            continue
        if _any_nonfinite(p.data):
            return False
    return True