#src/losses.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def trajectory_loss(
    pred: torch.Tensor,
    y_abs: torch.Tensor,
    x_last_abs: torch.Tensor,
    predict_delta: bool,
    w_traj: float = 1.0,
    w_fde: float = 0.0,
) -> torch.Tensor:
    """
    Simple L2 loss in absolute space + optional FDE term.
    pred: (B,Tf,2)
    """
    l2 = torch.norm(pred - y_abs, dim=-1).mean()
    if w_fde > 0.0:
        f = torch.norm(pred[:, -1, :] - y_abs[:, -1, :], dim=-1).mean()
        return w_traj * l2 + w_fde * f
    return w_traj * l2

def _best_mode_by_minade(pred_abs_all: torch.Tensor, y_abs: torch.Tensor) -> torch.Tensor:
    """
    pred_abs_all: (B,M,Tf,2) absolute
    y_abs:        (B,Tf,2)
    return: best_idx (B,)
    """
    err = torch.norm(pred_abs_all - y_abs[:, None, :, :], dim=-1)  # (B,M,Tf)
    ade_bm = err.mean(dim=-1)                                      # (B,M)
    return ade_bm.argmin(dim=1)


def multimodal_loss(
    pred: torch.Tensor,
    y_abs: torch.Tensor,
    x_last_abs: torch.Tensor,
    predict_delta: bool,
    score_logits: Optional[torch.Tensor],
    w_traj: float = 1.0,
    w_fde: float = 0.0,
    w_cls: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    pred: (B,M,Tf,2)
    score_logits: (B,M) or None

    - Choose best mode by ADE (abs space)
    - Regression loss on best mode
    - Optional classification loss encouraging best mode (cross entropy on logits)
    """
    B, M, Tf, _ = pred.shape

    if predict_delta:
        pred_abs_all = torch.cumsum(pred, dim=2) + x_last_abs[:, None, None, :]
    else:
        pred_abs_all = pred

    # ADE per mode: (B,M)
    ade_m = torch.norm(pred_abs_all - y_abs[:, None, :, :], dim=-1).mean(dim=-1)
    best_idx = torch.argmin(ade_m, dim=1)  # (B,)

    best_pred_abs = pred_abs_all[torch.arange(B, device=pred.device), best_idx]  # (B,Tf,2)

    reg = torch.norm(best_pred_abs - y_abs, dim=-1).mean()
    if w_fde > 0.0:
        reg = reg + w_fde * torch.norm(best_pred_abs[:, -1, :] - y_abs[:, -1, :], dim=-1).mean()

    loss = w_traj * reg

    if (score_logits is not None) and (w_cls > 0.0):
        cls = nn.functional.cross_entropy(score_logits, best_idx)
        loss = loss + w_cls * cls

    return loss, best_idx