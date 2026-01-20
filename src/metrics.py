#src/metrics.py
import torch

@torch.no_grad()
def ade(pred_abs: torch.Tensor, y_abs: torch.Tensor) -> torch.Tensor:
    # pred_abs, y_abs: (B,Tf,2)
    return torch.norm(pred_abs - y_abs, dim=-1).mean(dim=-1).mean()


@torch.no_grad()
def fde(pred_abs: torch.Tensor, y_abs: torch.Tensor) -> torch.Tensor:
    # pred_abs, y_abs: (B,Tf,2)
    return torch.norm(pred_abs[:, -1, :] - y_abs[:, -1, :], dim=-1).mean()


def delta_to_abs(pred_delta: torch.Tensor, x_last_abs: torch.Tensor) -> torch.Tensor:
    # pred_delta: (B,Tf,2), x_last_abs: (B,2)
    return torch.cumsum(pred_delta, dim=1) + x_last_abs[:, None, :]

def ade_per_sample(pred_abs: torch.Tensor, y_abs: torch.Tensor) -> torch.Tensor:
    """
    pred_abs, y_abs: (B, Tf, 2)
    -> (B,)
    """
    d = torch.norm(pred_abs - y_abs, dim=-1)  # (B,Tf)
    return d.mean(dim=-1)

def fde_per_sample(pred_abs: torch.Tensor, y_abs: torch.Tensor) -> torch.Tensor:
    """
    -> (B,)
    """
    d = torch.norm(pred_abs[:, -1, :] - y_abs[:, -1, :], dim=-1)  # (B,)
    return d