import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast

import numpy as np
import pandas as pd
from collections import defaultdict

from pathlib import Path
from typing import Dict

from tqdm import tqdm

import time

import math

from src.metrics import (
    ade, fde, ade_per_sample, fde_per_sample
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
    w_traj: float,
    w_fde: float,
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
) -> Dict[str, float]:
    """
    Returns (overall):
      loss, ade, fde, vel, acc, jerk, n_samples,
      matched, matched_ratio   (when labels_lut provided; otherwise matched=-1)

    Also optionally appends per-epoch stratified CSV:
      - save_event_path: per-event label metrics
      - save_state_path: per-state label metrics

    Kinematic metrics:
      vel/acc/jerk are "ADE-style" time-averaged errors over the future horizon (no FDE).
    """
    model.eval()

    if data_hz <= 0:
        raise ValueError(f"data_hz must be > 0, got {data_hz}")

    # -------------------------
    # Overall accumulators (sample-weighted)
    # -------------------------
    sum_loss = 0.0
    sum_ade = 0.0
    sum_fde = 0.0
    sum_vel = 0.0
    sum_acc = 0.0
    sum_jerk = 0.0
    n_samples = 0

    # -------------------------
    # Stratified accumulators
    # label -> [sum_ade, sum_fde, sum_vel, sum_acc, sum_jerk, count]
    # -------------------------
    ev_stats = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0])
    st_stats = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0])
    n_matched = 0
    n_total_samples = 0  # denominator for matched_ratio

    # dt = 1/hz, and for finite-diff we use hz multiplier
    hz = float(data_hz)

    if measure_latency:
        # 1) get one batch (does NOT consume the main loop if we use iter(loader) fresh)
        first = next(iter(loader))

        x_ego_l = first["x_ego"].to(device, non_blocking=True)
        x_nb_l = first["x_nb"].to(device, non_blocking=True)
        nb_mask_l = first["nb_mask"].to(device, non_blocking=True)
        x_last_abs_l = first["x_last_abs"].to(device, non_blocking=True)

        B_lat = int(x_ego_l.shape[0])

        def _one_infer():
            # keep same autocast setting as eval
            with autocast(device_type="cuda", enabled=use_amp):
                out = model(x_ego_l, x_nb_l, nb_mask_l)

                # minimal postprocess so that multimodal outputs do a similar amount of work
                if isinstance(out, (tuple, list)):
                    pred, scores = out
                else:
                    pred, scores = out, None

                if pred.dim() == 4:
                    # pred: (B,K,Tf,2)
                    # choose one mode cheaply (avoid using y_abs/loss in latency)
                    if scores is not None:
                        best_idx = torch.argmax(scores, dim=1)
                    else:
                        best_idx = torch.zeros(pred.shape[0], device=pred.device, dtype=torch.long)

                    if predict_delta:
                        pred_abs_all = torch.cumsum(pred, dim=2) + x_last_abs_l[:, None, None, :]
                    else:
                        pred_abs_all = pred

                    _ = pred_abs_all[torch.arange(pred.shape[0], device=pred.device), best_idx]
                else:
                    # single-mode: just run through (no need to compute abs if you only care forward cost)
                    # but include delta->abs cost if your real inference requires it:
                    if predict_delta:
                        _ = torch.cumsum(pred, dim=1) + x_last_abs_l[:, None, :]
                    else:
                        _ = pred

            return None

        lat = measure_latency_ms(
            fn=_one_infer,
            device=device,
            iters=latency_iters,
            warmup=latency_warmup,
        )

        print()
        if latency_per_sample and B_lat > 0:
            div = float(B_lat)
            print(
                f"[LATENCY] avg={lat['avg_ms']/div:.3f} ms/sample "
                f"(p50={lat['p50_ms']/div:.3f}, p90={lat['p90_ms']/div:.3f}, p99={lat['p99_ms']/div:.3f}) "
                f"@batch={B_lat}, iters={int(lat['iters'])}"
            )
        else:
            print(
                f"[LATENCY] avg={lat['avg_ms']:.3f} ms/batch "
                f"(p50={lat['p50_ms']:.3f}, p90={lat['p90_ms']:.3f}, p99={lat['p99_ms']:.3f}) "
                f"@batch={B_lat}, iters={int(lat['iters'])}"
            )

    # -------------------------
    # helper: kinematic per-sample ADE-style error
    # -------------------------
    def _kin_err_per_sample(pred_kin: torch.Tensor, gt_kin: torch.Tensor) -> torch.Tensor:
        """
        pred_kin, gt_kin: (B, Tf, 2)
        returns: (B,) time-averaged L2 error over Tf
        """
        # (B, Tf)
        e = torch.norm(pred_kin - gt_kin, dim=-1)
        return e.mean(dim=1)

    def _finite_diff(x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Tf, 2)
        returns dx/dt approx on same length Tf:
          v[t] = (x[t] - x[t-1]) * hz  for t>=1
          v[0] = v[1] (or 0 if Tf==1)
        """
        B, Tf, D = x.shape
        if Tf <= 1:
            return torch.zeros_like(x)

        dx = x[:, 1:, :] - x[:, :-1, :]          # (B, Tf-1, 2)
        v = dx * hz                               # per-second
        v0 = v[:, :1, :]                          # (B,1,2)
        out = torch.cat([v0, v], dim=1)           # (B,Tf,2)
        return out

    # [수정 1] RMSE 평가를 위한 설정 (1초~5초)
    eval_horizons_sec = [1, 2, 3, 4, 5]
    rmse_accum = {t: 0.0 for t in eval_horizons_sec} # 오차 제곱합 누적용
    rmse_counts = {t: 0 for t in eval_horizons_sec}  # 샘플 개수 누적용

    # Frame Index 계산 함수 (0-based index)
    # 예: 25Hz일 때 1초 = 25번째 프레임 -> index 24
    def get_horizon_idx(sec, hz):
        return int(sec * hz) - 1

    pbar = tqdm(loader, desc="val", dynamic_ncols=True, leave=False)
    for batch in pbar:
        x_ego = batch["x_ego"].to(device, non_blocking=True)
        x_nb = batch["x_nb"].to(device, non_blocking=True)
        nb_mask = batch["nb_mask"].to(device, non_blocking=True)

        y_abs = batch["y"].to(device, non_blocking=True)          # (B,Tf,2)
        x_last_abs = batch["x_last_abs"].to(device, non_blocking=True)

        # ---- GT kinematics must exist (as you intended) ----
        if "y_vel" not in batch or "y_acc" not in batch:
            raise KeyError("Batch missing 'y_vel' or 'y_acc'. PtWindowDataset must provide them.")
        y_vel = batch["y_vel"].to(device, non_blocking=True)      # (B,Tf,2)
        y_acc = batch["y_acc"].to(device, non_blocking=True)      # (B,Tf,2)

        with autocast(device_type="cuda", enabled=use_amp):
            out = model(x_ego, x_nb, nb_mask)
            if isinstance(out, (tuple, list)):
                pred, scores = out
            else:
                pred, scores = out, None

            # -------- predict trajectories --------
            if pred.dim() == 4:
                # multimodal: pred (B,K,Tf,2)
                loss, best_idx = multimodal_loss(
                    pred=pred,
                    y_abs=y_abs,
                    x_last_abs=x_last_abs,
                    predict_delta=predict_delta,
                    score_logits=scores,
                    w_traj=w_traj,
                    w_fde=w_fde,
                    w_cls=w_cls,
                )
                if predict_delta:
                    pred_abs_all = torch.cumsum(pred, dim=2) + x_last_abs[:, None, None, :]
                else:
                    pred_abs_all = pred
                pred_abs = pred_abs_all[torch.arange(pred.shape[0], device=pred.device), best_idx]
            else:
                # single-mode: pred (B,Tf,2) or delta
                loss = trajectory_loss(
                    pred=pred,
                    y_abs=y_abs,
                    x_last_abs=x_last_abs,
                    predict_delta=predict_delta,
                    w_traj=w_traj,
                    w_fde=w_fde,
                )
                pred_abs = delta_to_abs(pred, x_last_abs) if predict_delta else pred

        B = int(pred_abs.shape[0])
        n_samples += B

        # loss is assumed batch-mean -> make sample-weighted
        sum_loss += float(loss.item()) * B

        # ---- displacement metrics (per-sample to avoid redundant compute) ----
        ade_s = ade_per_sample(pred_abs, y_abs)  # (B,)
        fde_s = fde_per_sample(pred_abs, y_abs)  # (B,)
        sum_ade += float(ade_s.sum().item())
        sum_fde += float(fde_s.sum().item())

        # ---- predicted kinematics from pred_abs ----
        pred_vel = _finite_diff(pred_abs)          # (B,Tf,2)
        pred_acc = _finite_diff(pred_vel)          # (B,Tf,2)
        pred_jerk = _finite_diff(pred_acc)         # (B,Tf,2)

        # ---- GT jerk from GT acc ----
        gt_jerk = _finite_diff(y_acc)

        # ---- kinematic ADE-style errors (per-sample) ----
        vel_s = _kin_err_per_sample(pred_vel, y_vel)       # (B,)
        acc_s = _kin_err_per_sample(pred_acc, y_acc)       # (B,)
        jerk_s = _kin_err_per_sample(pred_jerk, gt_jerk)   # (B,)

        sum_vel += float(vel_s.sum().item())
        sum_acc += float(acc_s.sum().item())
        sum_jerk += float(jerk_s.sum().item())

        # [수정 2] Horizon 별 Squared Error 누적 계산
        # 전체 시점의 거리 오차(Euclidean distance)를 먼저 구함: (B, Tf)
        dist_all = torch.norm(pred_abs - y_abs, dim=-1)

        B_curr = int(pred_abs.shape[0])
        Tf_curr = int(pred_abs.shape[1])

        for sec in eval_horizons_sec:
            idx = get_horizon_idx(sec, hz)
            # 예측 길이가 해당 시간(idx)보다 긴 경우에만 계산
            if 0 <= idx < Tf_curr:
                # 해당 시점의 오차 (B,)
                d_t = dist_all[:, idx]
                # 제곱합 누적
                rmse_accum[sec] += (d_t ** 2).sum().item()
                rmse_counts[sec] += B_curr

        # denominator for matched ratio
        if labels_lut is not None:
            n_total_samples += B

        # ---- stratified: only if meta exists ----
        if labels_lut is not None and ("meta" in batch):
            metas = batch["meta"]
            Bb = min(len(metas), B)

            ade_np = ade_s.detach().cpu().numpy()
            fde_np = fde_s.detach().cpu().numpy()
            vel_np = vel_s.detach().cpu().numpy()
            acc_np = acc_s.detach().cpu().numpy()
            jerk_np = jerk_s.detach().cpu().numpy()

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
                st = lab.get("state_label", None) or "unknown"

                # event accum
                ev_stats[ev][0] += float(ade_np[i])
                ev_stats[ev][1] += float(fde_np[i])
                ev_stats[ev][2] += float(vel_np[i])
                ev_stats[ev][3] += float(acc_np[i])
                ev_stats[ev][4] += float(jerk_np[i])
                ev_stats[ev][5] += 1

                # state accum
                st_stats[st][0] += float(ade_np[i])
                st_stats[st][1] += float(fde_np[i])
                st_stats[st][2] += float(vel_np[i])
                st_stats[st][3] += float(acc_np[i])
                st_stats[st][4] += float(jerk_np[i])
                st_stats[st][5] += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "ADE": f"{ade_s.mean().item():.3f}",
            "FDE": f"{fde_s.mean().item():.3f}",
            "vel": f"{vel_s.mean().item():.3f}",
            "acc": f"{acc_s.mean().item():.3f}",
            "jerk": f"{jerk_s.mean().item():.3f}",
        })

    # -------------------------
    # Overall means
    # -------------------------
    overall_loss = sum_loss / max(1, n_samples)
    overall_ade = sum_ade / max(1, n_samples)
    overall_fde = sum_fde / max(1, n_samples)
    overall_vel = sum_vel / max(1, n_samples)
    overall_acc = sum_acc / max(1, n_samples)
    overall_jerk = sum_jerk / max(1, n_samples)

    matched_ratio = float("nan")
    if labels_lut is not None and n_total_samples > 0:
        matched_ratio = float(n_matched / max(1, n_total_samples))

    # [수정 3] 최종 RMSE 계산
    results = {
        "loss": float(overall_loss),
        "ade": float(overall_ade),
        "fde": float(overall_fde),
        "vel": float(overall_vel),
        "acc": float(overall_acc),
        "jerk": float(overall_jerk),
        "n_samples": int(n_samples),
        "matched": int(n_matched) if labels_lut is not None else -1,
        "matched_ratio": float(matched_ratio),
    }

    # RMSE 결과를 results 딕셔너리에 추가
    for sec in eval_horizons_sec:
        total_se = rmse_accum[sec]
        count = rmse_counts[sec]
        if count > 0:
            rmse_val = math.sqrt(total_se / count)
        else:
            rmse_val = float('nan') # 해당 시간대 데이터가 없거나 예측 길이가 짧음
        results[f"rmse_{sec}s"] = rmse_val

    # -------------------------
    # Save per-label CSV (event/state)
    # - weighted_* 제거
    # - label별 평균만 저장
    # -------------------------
    def _per_label_row(stats_dict, label: str):
        sa, sf, sv, sac, sj, c = stats_dict.get(label, [0.0, 0.0, 0.0, 0.0, 0.0, 0])
        if c <= 0:
            return (0, np.nan, np.nan, np.nan, np.nan, np.nan)
        c = int(c)
        return (c, float(sa / c), float(sf / c), float(sv / c), float(sac / c), float(sj / c))

    if labels_lut is not None and n_total_samples > 0 and epoch is not None:
        coverage = {
            "epoch": int(epoch),
            "matched": int(n_matched),
            "total": int(n_total_samples),
            "matched_ratio": float(n_matched / max(1, n_total_samples)),
        }

        # 원하는 label 리스트는 네 프로젝트 정의와 맞추면 됨
        EVENT_LABELS = [
            "none", "cut_in", "merging", "diverging",
            "simple_lane_change", "lane_change_other", "unknown"
        ]
        STATE_LABELS = [
            "free_flow", "dense", "car_following",
            "ramp_driving", "other", "unknown"
        ]

        if save_event_path is not None:
            save_event_path.parent.mkdir(parents=True, exist_ok=True)
            row = dict(coverage)
            for lab in EVENT_LABELS:
                c, a, f, v, ac, j = _per_label_row(ev_stats, lab)
                row[f"{lab}_count"] = c
                row[f"{lab}_ADE"] = a
                row[f"{lab}_FDE"] = f
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
                c, a, f, v, ac, j = _per_label_row(st_stats, lab)
                row[f"{lab}_count"] = c
                row[f"{lab}_ADE"] = a
                row[f"{lab}_FDE"] = f
                row[f"{lab}_vel"] = v
                row[f"{lab}_acc"] = ac
                row[f"{lab}_jerk"] = j

            df_row = pd.DataFrame([row])
            header = not save_state_path.exists()
            df_row.to_csv(save_state_path, mode="a", header=header, index=False)

    return results


def train_one_epoch(
    model, loader, device, optimizer, scheduler, scaler,
    use_amp, predict_delta, grad_clip_norm,
    w_traj, w_fde, w_cls, global_step, log_every, epoch
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
                w_traj=w_traj, w_fde=w_fde, w_cls=w_cls
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

            if _any_nonfinite(x_ego) or _any_nonfinite(x_nb) or _any_nonfinite(y_abs) or _any_nonfinite(x_last_abs):
                print(f"[BAD INPUT] ep={epoch} it={it}")
                print("  x_ego:", _finite_stats(x_ego))
                print("  x_nb :", _finite_stats(x_nb))
                print("  y_abs:", _finite_stats(y_abs))
                print("  x_last_abs:", _finite_stats(x_last_abs))
                raise RuntimeError("Non-finite in inputs")

            if _any_nonfinite(loss):
                print(f"[BAD LOSS] ep={epoch} it={it} loss={loss.item()}")
                raise RuntimeError("Non-finite loss")

            if not _check_params_finite(model):
                print(f"[BAD PARAM] ep={epoch} it={it} params became non-finite AFTER step()")
                raise RuntimeError("Params exploded to NaN/Inf")

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