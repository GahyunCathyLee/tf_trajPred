# src/losses.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def trajectory_loss(
    pred: torch.Tensor,
    y_abs: torch.Tensor,
    w_traj: float = 1.0,
    w_fde: float = 0.0,
) -> torch.Tensor:
    """
    Optimized Trajectory Loss.
    """
    
    dist = torch.norm(pred - y_abs, dim=-1) # (B, Tf)
    l2 = dist.mean()
    
    loss = w_traj * l2
    
    if w_fde > 0.0:
        f = dist[:, -1].mean()
        loss += w_fde * f
        
    return loss

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
    Optimized Multimodal Loss.
    Removes redundant coordinate extraction and norm calculation.
    """
    B, M, Tf, _ = pred.shape

    if predict_delta:
        pred_abs_all = torch.cumsum(pred, dim=2) + x_last_abs[:, None, None, :]
    else:
        pred_abs_all = pred

    err_dist = torch.norm(pred_abs_all - y_abs[:, None, :, :], dim=-1) 

    ade_m = err_dist.mean(dim=-1)  # (B, M)
    best_idx = torch.argmin(ade_m, dim=1)  # (B,)
    
    gather_idx = best_idx.view(B, 1, 1).expand(B, 1, Tf)
    
    # (B, 1, Tf) -> (B, Tf)
    best_dist = torch.gather(err_dist, 1, gather_idx).squeeze(1)

    reg_loss = best_dist.mean() # ADE of best mode

    if w_fde > 0.0:
        fde_loss = best_dist[:, -1].mean() # FDE of best mode
        reg_loss = reg_loss + w_fde * fde_loss

    loss = w_traj * reg_loss

    # 5. Classification Loss
    if (score_logits is not None) and (w_cls > 0.0):
        cls_loss = F.cross_entropy(score_logits, best_idx)
        loss = loss + w_cls * cls_loss

    return loss, best_idx