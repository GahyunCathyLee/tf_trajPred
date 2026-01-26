# src/losses.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def trajectory_loss(
    pred: torch.Tensor,
    y_abs: torch.Tensor,
    w_ade: float = 1.0,
    w_fde: float = 0.0,
) -> torch.Tensor:
    """
    Optimized Trajectory Loss.
    """
    
    dist = torch.norm(pred - y_abs, dim=-1) # (B, Tf)
    l2 = dist.mean()
    
    loss = w_ade * l2
    
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
    w_ade: float = 1.0,
    w_fde: float = 0.0,
    w_cls: float = 0.0,
    w_rmse: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, M, Tf, _ = pred.shape

    if predict_delta:
        pred_abs_all = torch.cumsum(pred, dim=2) + x_last_abs[:, None, None, :]
    else:
        pred_abs_all = pred

    # 1. 공통 오차 계산 (L2 distance)
    # err_dist: (B, M, Tf)
    err_dist = torch.norm(pred_abs_all - y_abs[:, None, :, :], dim=-1) 

    # 2. Best Mode 선택 (기존처럼 ADE 기준으로 선택)
    ade_m = err_dist.mean(dim=-1)  # (B, M)
    best_idx = torch.argmin(ade_m, dim=1)  # (B,)
    
    gather_idx = best_idx.view(B, 1, 1).expand(B, 1, Tf)
    best_dist = torch.gather(err_dist, 1, gather_idx).squeeze(1) # (B, Tf)

    # 3. Loss components
    loss = 0.0

    # ADE Loss (L1-style)
    if w_ade > 0.0:
        loss += w_ade * best_dist.mean()

    # FDE Loss
    if w_fde > 0.0:
        loss += w_fde * best_dist[:, -1].mean()

    # RMSE Loss (MSE-style)
    if w_rmse > 0.0:
        # 제곱 오차의 평균에 루트를 씌워 RMSE 형태의 Loss를 만듭니다.
        mse_loss = torch.pow(best_dist, 2).mean()
        rmse_loss = torch.sqrt(mse_loss + 1e-6) # 수치 안정성을 위해 epsilon 추가
        loss += w_rmse * rmse_loss

    # 4. Classification Loss
    if (score_logits is not None) and (w_cls > 0.0):
        cls_loss = F.cross_entropy(score_logits, best_idx)
        loss += w_cls * cls_loss

    return loss, best_idx