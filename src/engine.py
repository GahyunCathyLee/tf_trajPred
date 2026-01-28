# src/engine.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast

import numpy as np
import pandas as pd
from collections import defaultdict

from pathlib import Path
from typing import Dict, Any, Optional

from tqdm import tqdm

import time
import math

from src.metrics import (
    ade, fde, ade_per_sample, fde_per_sample, delta_to_abs
)
from src.losses import trajectory_loss, multimodal_loss
from src.utils import _to_int, measure_latency_ms
from src.debug.debug import _any_nonfinite, _finite_stats, _check_params_finite

# -------------------------
# Eval / Train loops
# -------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    predict_delta: bool,
    w_ade: float,
    w_fde: float,
    w_rmse: float,
    w_cls: float,
    data_hz: float,
    labels_lut=None,
    save_event_path: Path | None = None,
    save_state_path: Path | None = None,
    epoch: int | None = None,
    measure_latency: bool = False,
    latency_iters: int = 200,
    latency_warmup: int = 30,
    latency_per_sample: bool = True,
    # [설명] Stratified CSV에 경로를 기록하기 위해 인자로 받습니다.
    cfg_path: Path | str = "unknown",
    ckpt_path: Path | str = "unknown",
) -> Dict[str, float]:
    model.eval()

    if data_hz <= 0:
        raise ValueError(f"data_hz must be > 0, got {data_hz}")

    # -------------------------
    # Overall accumulators
    # -------------------------
    sum_loss = 0.0
    sum_ade = 0.0
    sum_fde = 0.0
    sum_vel = 0.0
    sum_acc = 0.0
    sum_jerk = 0.0
    
    # [설명] RMSE의 정확한 계산을 위해 Squared Error 합을 누적합니다.
    sum_se_total = 0.0 
    n_points_total = 0

    n_samples = 0

    # Stratified accumulators
    # Index: 0:ADE, 1:FDE, 2:Vel, 3:Acc, 4:Jerk, 5:SE(Squared Error for RMSE), 6:Count
    ev_stats = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0])
    st_stats = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0])
    n_matched = 0
    n_total_samples = 0 

    hz = float(data_hz)

    # -------------------------
    # Latency measurement
    # -------------------------
    lat_results = {}
    if measure_latency:
        try:
            # Latency 측정을 위한 더미 데이터 가져오기
            first = next(iter(loader))
            x_ego_l = first["x_ego"].to(device, non_blocking=True)
            x_nb_l = first["x_nb"].to(device, non_blocking=True)
            nb_mask_l = first["nb_mask"].to(device, non_blocking=True)
            x_last_abs_l = first["x_last_abs"].to(device, non_blocking=True)
            B_lat = int(x_ego_l.shape[0])

            def _one_infer():
                with autocast(device_type="cuda", enabled=use_amp):
                    out = model(x_ego_l, x_nb_l, nb_mask_l)
                    if isinstance(out, (tuple, list)):
                        pred, scores = out
                    else:
                        pred, scores = out, None
                    
                    if pred.dim() == 4: # Multi-modal
                        if scores is not None:
                            best_idx = torch.argmax(scores, dim=1)
                        else:
                            best_idx = torch.zeros(pred.shape[0], device=pred.device, dtype=torch.long)
                        if predict_delta:
                            pred_abs_all = torch.cumsum(pred, dim=2) + x_last_abs_l[:, None, None, :]
                        else:
                            pred_abs_all = pred
                        _ = pred_abs_all[torch.arange(pred.shape[0], device=pred.device), best_idx]
                    else: # Single-modal
                        if predict_delta:
                            _ = torch.cumsum(pred, dim=1) + x_last_abs_l[:, None, :]
                        else:
                            _ = pred
                return None

            # warm-up and measure
            lat = measure_latency_ms(fn=_one_infer, device=device, iters=latency_iters, warmup=latency_warmup)
            
            avg_ms = lat.get('avg_ms', float('nan'))
            p50 = lat.get('p50_ms', float('nan'))
            p99 = lat.get('p99_ms', float('nan'))
            
            # [수정] Sample당 Latency 계산
            per_sample_avg = avg_ms / B_lat if B_lat > 0 else float('nan')
            
            print(f"[Lat] Batch({B_lat}): {avg_ms:.2f}ms | Sample: {per_sample_avg:.4f}ms | P50: {p50:.2f}ms | P99: {p99:.2f}ms")
            
            lat_results['latency_ms'] = per_sample_avg
        except StopIteration:
            print("[WARN] DataLoader is empty, skipping latency measure.")
            lat_results['latency_ms'] = float('nan')

    def _kin_err_per_sample(pred_kin: torch.Tensor, gt_kin: torch.Tensor) -> torch.Tensor:
        e = torch.norm(pred_kin - gt_kin, dim=-1)
        return e.mean(dim=1)

    def _finite_diff(x: torch.Tensor) -> torch.Tensor:
        B, Tf, D = x.shape
        if Tf <= 1:
            return torch.zeros_like(x)
        dx = x[:, 1:, :] - x[:, :-1, :]
        v = dx * hz
        v0 = v[:, :1, :]
        out = torch.cat([v0, v], dim=1)
        return out

    eval_horizons_sec = [1, 2, 3, 4, 5]
    rmse_accum = {t: 0.0 for t in eval_horizons_sec}
    rmse_counts = {t: 0 for t in eval_horizons_sec}

    def get_horizon_idx(sec, hz):
        return int(sec * hz) - 1

    pbar = tqdm(loader, desc="val", dynamic_ncols=True, leave=False)
    for batch in pbar:
        x_ego = batch["x_ego"].to(device, non_blocking=True)
        x_nb = batch["x_nb"].to(device, non_blocking=True)
        nb_mask = batch["nb_mask"].to(device, non_blocking=True)
        y_abs = batch["y"].to(device, non_blocking=True)
        x_last_abs = batch["x_last_abs"].to(device, non_blocking=True)
        y_vel = batch["y_vel"].to(device, non_blocking=True)
        y_acc = batch["y_acc"].to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=use_amp):
            out = model(x_ego, x_nb, nb_mask)
            if isinstance(out, (tuple, list)):
                pred, scores = out
            else:
                pred, scores = out, None

            if pred.dim() == 4:
                loss, best_idx = multimodal_loss(
                    pred=pred, y_abs=y_abs, x_last_abs=x_last_abs,
                    predict_delta=predict_delta, score_logits=scores,
                    w_ade=w_ade, w_fde=w_fde, w_rmse=w_rmse, w_cls=w_cls,
                )
                if predict_delta:
                    pred_abs_all = torch.cumsum(pred, dim=2) + x_last_abs[:, None, None, :]
                else:
                    pred_abs_all = pred
                pred_abs = pred_abs_all[torch.arange(pred.shape[0], device=pred.device), best_idx]
            else:
                loss = trajectory_loss(
                    pred=pred, y_abs=y_abs, x_last_abs=x_last_abs,
                    predict_delta=predict_delta, w_ade=w_ade, w_fde=w_fde, w_rmse=w_rmse
                )
                pred_abs = delta_to_abs(pred, x_last_abs) if predict_delta else pred

        B = int(pred_abs.shape[0])
        n_samples += B
        sum_loss += float(loss.item()) * B

        ade_s = ade_per_sample(pred_abs, y_abs)
        fde_s = fde_per_sample(pred_abs, y_abs)
        sum_ade += float(ade_s.sum().item())
        sum_fde += float(fde_s.sum().item())

        pred_vel = _finite_diff(pred_abs)
        pred_acc = _finite_diff(pred_vel)
        pred_jerk = _finite_diff(pred_acc)
        gt_jerk = _finite_diff(y_acc)

        vel_s = _kin_err_per_sample(pred_vel, y_vel)
        acc_s = _kin_err_per_sample(pred_acc, y_acc)
        jerk_s = _kin_err_per_sample(pred_jerk, gt_jerk)

        sum_vel += float(vel_s.sum().item())
        sum_acc += float(acc_s.sum().item())
        sum_jerk += float(jerk_s.sum().item())

        # [RMSE 계산 1] Overall RMSE를 위해 전체 포인트 제곱합 누적
        dist_all = torch.norm(pred_abs - y_abs, dim=-1) # (B, Tf)
        sum_se_total += (dist_all ** 2).sum().item()
        n_points_total += dist_all.numel()
        
        # [RMSE 계산 2] Stratified RMSE를 위해 샘플별 MSE 누적
        mse_s = (dist_all ** 2).mean(dim=1) # (B,)

        # [RMSE 계산 3] Horizon별 RMSE
        B_curr = int(pred_abs.shape[0])
        Tf_curr = int(pred_abs.shape[1])

        for sec in eval_horizons_sec:
            idx = get_horizon_idx(sec, hz)
            if 0 <= idx < Tf_curr:
                d_t = dist_all[:, idx]
                rmse_accum[sec] += (d_t ** 2).sum().item()
                rmse_counts[sec] += B_curr

        if labels_lut is not None:
            n_total_samples += B
            if "meta" in batch:
                metas = batch["meta"]
                Bb = min(len(metas), B)
                ade_np = ade_s.detach().cpu().numpy()
                fde_np = fde_s.detach().cpu().numpy()
                vel_np = vel_s.detach().cpu().numpy()
                acc_np = acc_s.detach().cpu().numpy()
                jerk_np = jerk_s.detach().cpu().numpy()
                mse_np = mse_s.detach().cpu().numpy()

                for i in range(Bb):
                    m = metas[i] or {}
                    rid = m.get("recordingId", None)
                    tid = m.get("trackId", None)
                    t0 = m.get("t0_frame", None)
                    if rid is None or tid is None or t0 is None:
                        continue
                    key = (_to_int(rid), _to_int(tid), _to_int(t0))
                    lab = labels_lut.get(key, None)
                    if lab is None:
                        continue
                    n_matched += 1
                    
                    ev = lab.get("event_label", None) or "unknown"
                    
                    # [수정] Lane Change 라벨 통합 로직
                    if ev == "simple_lane_change" or ev == "lane_change_other":
                        ev = "lane_change"

                    st = lab.get("state_label", None) or "unknown"

                    # Index: 0:ADE, 1:FDE, 2:Vel, 3:Acc, 4:Jerk, 5:SE(MSE), 6:Count
                    
                    ev_stats[ev][0] += float(ade_np[i])
                    ev_stats[ev][1] += float(fde_np[i])
                    ev_stats[ev][2] += float(vel_np[i])
                    ev_stats[ev][3] += float(acc_np[i])
                    ev_stats[ev][4] += float(jerk_np[i])
                    ev_stats[ev][5] += float(mse_np[i]) # 누적: Mean Squared Error per sample
                    ev_stats[ev][6] += 1

                    st_stats[st][0] += float(ade_np[i])
                    st_stats[st][1] += float(fde_np[i])
                    st_stats[st][2] += float(vel_np[i])
                    st_stats[st][3] += float(acc_np[i])
                    st_stats[st][4] += float(jerk_np[i])
                    st_stats[st][5] += float(mse_np[i])
                    st_stats[st][6] += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "ADE": f"{ade_s.mean().item():.3f}",
            "FDE": f"{fde_s.mean().item():.3f}",
            "vel": f"{vel_s.mean().item():.3f}",
        })

    # -------------------------
    # Final Metrics
    # -------------------------
    overall_loss = sum_loss / max(1, n_samples)
    overall_ade = sum_ade / max(1, n_samples)
    overall_fde = sum_fde / max(1, n_samples)
    overall_vel = sum_vel / max(1, n_samples)
    overall_acc = sum_acc / max(1, n_samples)
    overall_jerk = sum_jerk / max(1, n_samples)
    
    # [RMSE 최종 계산] sqrt(Total SE / Total Points)
    overall_rmse = math.sqrt(sum_se_total / max(1, n_points_total))

    matched_ratio = float("nan")
    if labels_lut is not None and n_total_samples > 0:
        matched_ratio = float(n_matched / max(1, n_total_samples))

    results = {
        "loss": float(overall_loss),
        "ade": float(overall_ade),
        "fde": float(overall_fde),
        "rmse": float(overall_rmse),
        "vel": float(overall_vel),
        "acc": float(overall_acc),
        "jerk": float(overall_jerk),
        "n_samples": int(n_samples),
        "matched": int(n_matched) if labels_lut is not None else -1,
        "matched_ratio": float(matched_ratio),
    }

    if measure_latency:
        results.update(lat_results)

    for sec in eval_horizons_sec:
        total_se = rmse_accum[sec]
        count = rmse_counts[sec]
        if count > 0:
            rmse_val = math.sqrt(total_se / count)
        else:
            rmse_val = float('nan')
        results[f"rmse_{sec}s"] = rmse_val

    # -------------------------
    # Stratified CSV saving
    # -------------------------
    def _per_label_row(stats_dict, label: str):
        # sa: sum_ade, sf: sum_fde, sv: sum_vel, sac: sum_acc, sj: sum_jerk, smse: sum_mse, c: count
        vals = stats_dict.get(label, [0.0]*7)
        sa, sf, sv, sac, sj, smse, c = vals
        if c <= 0:
            return (0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        c = int(c)
        rmse = math.sqrt(smse / c) # sqrt(Average MSE) = Global RMSE (동일 length 가정 시)
        return (c, float(sa / c), float(sf / c), rmse, float(sv / c), float(sac / c), float(sj / c))

    if labels_lut is not None and n_total_samples > 0 and epoch is not None:
        coverage = {
            "config": str(cfg_path),
            "ckpt": str(ckpt_path),
            "epoch": int(epoch),
            "matched": int(n_matched),
            "total": int(n_total_samples),
            "matched_ratio": float(n_matched / max(1, n_total_samples)),
        }
        
        # [수정] 통합된 Label List 사용
        EVENT_LABELS = ["none", "cut_in", "merging", "diverging", "lane_change", "unknown"]
        STATE_LABELS = ["free_flow", "dense", "car_following", "ramp_driving", "other", "unknown"]

        if save_event_path is not None:
            save_event_path.parent.mkdir(parents=True, exist_ok=True)
            row = dict(coverage)
            for lab in EVENT_LABELS:
                c, a, f, r, v, ac, j = _per_label_row(ev_stats, lab)
                row[f"{lab}_count"] = c
                row[f"{lab}_ADE"] = a
                row[f"{lab}_FDE"] = f
                row[f"{lab}_RMSE"] = r
                row[f"{lab}_vel"] = v
                row[f"{lab}_acc"] = ac
                row[f"{lab}_jerk"] = j
            df_row = pd.DataFrame([row])
            header = not save_event_path.exists()
            df_row.to_csv(save_event_path, mode="a", header=header, index=False)

        if save_state_path is not None:
            save_state_path.parent.mkdir(parents=True, exist_ok=True)
            row = dict(coverage)
            for lab in STATE_LABELS:
                c, a, f, r, v, ac, j = _per_label_row(st_stats, lab)
                row[f"{lab}_count"] = c
                row[f"{lab}_ADE"] = a
                row[f"{lab}_FDE"] = f
                row[f"{lab}_RMSE"] = r
                row[f"{lab}_vel"] = v
                row[f"{lab}_acc"] = ac
                row[f"{lab}_jerk"] = j
            df_row = pd.DataFrame([row])
            header = not save_state_path.exists()
            df_row.to_csv(save_state_path, mode="a", header=header, index=False)

    return results

# train_one_epoch은 변경사항이 없으므로 생략합니다 (기존과 동일하게 사용).
def train_one_epoch(
    model, loader, device, optimizer, scheduler, scaler,
    use_amp, predict_delta, grad_clip_norm,
    w_ade, w_fde, w_rmse, w_cls, global_step, log_every, epoch
):
    model.train()
    total_loss_t = torch.zeros((), device=device)
    total_ade_t  = torch.zeros((), device=device)
    total_fde_t  = torch.zeros((), device=device)
    n = 0

    pbar = tqdm(loader, desc=f"train ep{epoch}", dynamic_ncols=True, leave=False)

    for it, batch in enumerate(pbar):
        x_ego = batch["x_ego"].to(device, non_blocking=True)
        x_nb  = batch["x_nb"].to(device, non_blocking=True)
        nb_mask = batch["nb_mask"].to(device, non_blocking=True)
        y_abs = batch["y"].to(device, non_blocking=True)
        x_last_abs = batch["x_last_abs"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=use_amp):
            out = model(x_ego, x_nb, nb_mask)
            if isinstance(out, (tuple, list)):
                pred, scores = out
            else:
                pred, scores = out, None

            loss, best_idx = multimodal_loss(
                pred=pred, y_abs=y_abs, x_last_abs=x_last_abs,
                predict_delta=predict_delta, score_logits=scores,
                w_ade=w_ade, w_fde=w_fde, w_rmse=w_rmse, w_cls=w_cls
            )

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            pred_abs_all = pred
            pred_abs = pred_abs_all[torch.arange(pred.shape[0], device=pred.device), best_idx]

            a = ade(pred_abs, y_abs)
            f = fde(pred_abs, y_abs)

        total_loss_t += loss.detach()
        total_ade_t  += a.detach()
        total_fde_t  += f.detach()
        n += 1
        global_step += 1

        if it % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            if _any_nonfinite(x_ego) or _any_nonfinite(x_nb) or _any_nonfinite(y_abs):
                print(f"[BAD INPUT] ep={epoch} it={it}")
                raise RuntimeError("Non-finite in inputs")

            if _any_nonfinite(loss):
                print(f"[BAD LOSS] ep={epoch} it={it} loss={loss.item()}")
                raise RuntimeError("Non-finite loss")

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ADE=f"{a.item():.3f}",
                FDE=f"{f.item():.3f}",
                lr=f"{lr:.1e}",
            )

    return {
        "loss": float((total_loss_t / max(1, n)).item()),
        "ade":  float((total_ade_t  / max(1, n)).item()),
        "fde":  float((total_fde_t  / max(1, n)).item()),
        "global_step_end": global_step,
    }