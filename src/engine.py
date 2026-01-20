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

from src.metrics import (
    ade,
    fde,
    ade_per_sample,
    fde_per_sample,
)
from src.losses import trajectory_loss, multimodal_loss
from src.utils import _to_int
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
    labels_lut=None,
    save_event_path: Path | None = None,
    save_state_path: Path | None = None,
    epoch: int | None = None,
) -> Dict[str, float]:
    model.eval()

    # sample-weighted overall accumulators
    sum_loss = 0.0
    sum_ade = 0.0
    sum_fde = 0.0
    n_samples = 0

    # stratified accumulators: label -> [sum_ade, sum_fde, count]
    ev_stats = defaultdict(lambda: [0.0, 0.0, 0])
    st_stats = defaultdict(lambda: [0.0, 0.0, 0])
    n_matched = 0
    n_total_samples = 0  # denominator for matched_ratio (count of evaluated samples)

    pbar = tqdm(loader, desc="val", dynamic_ncols=True, leave=False)
    for batch in pbar:
        x_ego = batch["x_ego"].to(device, non_blocking=True)
        x_nb = batch["x_nb"].to(device, non_blocking=True)
        nb_mask = batch["nb_mask"].to(device, non_blocking=True)
        y_abs = batch["y"].to(device, non_blocking=True)
        x_last_abs = batch["x_last_abs"].to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=use_amp):
            out = model(x_ego, x_nb, nb_mask)
            if isinstance(out, (tuple, list)):
                pred, scores = out
            else:
                pred, scores = out, None

            if pred.dim() == 4:
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

        # ---- overall: sample-wise ADE/FDE (no duplicate computation) ----
        ade_s_t = ade_per_sample(pred_abs, y_abs)  # [B]
        fde_s_t = fde_per_sample(pred_abs, y_abs)  # [B]

        sum_ade += float(ade_s_t.sum().item())
        sum_fde += float(fde_s_t.sum().item())

        # loss is assumed to be batch-mean; make it sample-weighted
        sum_loss += float(loss.item()) * B

        n_samples += B

        # denominator for matched ratio: count all evaluated samples when labels are enabled
        if labels_lut is not None:
            n_total_samples += B

        # ---- stratified: only if meta exists ----
        if labels_lut is not None and ("meta" in batch):
            ade_s = ade_s_t.detach().cpu().numpy()
            fde_s = fde_s_t.detach().cpu().numpy()
            metas = batch["meta"]

            # metas length should match B; if not, be safe
            Bb = min(len(metas), B)

            for i in range(Bb):
                m = metas[i] or {}
                rid = m.get("recordingId", None)
                tid = m.get("trackId", None)
                t0  = m.get("t0_frame", None)
                if rid is None or tid is None or t0 is None:
                    continue

                key = (_to_int(rid), _to_int(tid), _to_int(t0))
                lab = labels_lut.get(key, None)
                if lab is None:
                    continue

                n_matched += 1
                ev = lab.get("event_label", None) or "unknown"
                st = lab.get("state_label", None) or "unknown"

                ev_stats[ev][0] += float(ade_s[i])
                ev_stats[ev][1] += float(fde_s[i])
                ev_stats[ev][2] += 1

                st_stats[st][0] += float(ade_s[i])
                st_stats[st][1] += float(fde_s[i])
                st_stats[st][2] += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "ADE": f"{(ade_s_t.mean().item()):.3f}",
            "FDE": f"{(fde_s_t.mean().item()):.3f}",
        })

    # ---- helpers ----
    def _weighted_mean(stats_dict):
        tot = sum(v[2] for v in stats_dict.values())
        if tot <= 0:
            return (np.nan, np.nan, 0)
        sa = sum(v[0] for v in stats_dict.values())
        sf = sum(v[1] for v in stats_dict.values())
        return (sa / tot, sf / tot, tot)

    def _per_label(stats_dict, label):
        sa, sf, c = stats_dict.get(label, [0.0, 0.0, 0])
        if c <= 0:
            return (0, np.nan, np.nan)
        return (int(c), float(sa / c), float(sf / c))

    # ---- overall sample-weighted means ----
    overall_loss = sum_loss / max(1, n_samples)
    overall_ade  = sum_ade  / max(1, n_samples)
    overall_fde  = sum_fde  / max(1, n_samples)

    # ---- stratified summary for results.csv ----
    event_wade, event_wfde = (np.nan, np.nan)
    state_wade, state_wfde = (np.nan, np.nan)
    matched_ratio = np.nan

    if labels_lut is not None and n_total_samples > 0:
        matched_ratio = float(n_matched / max(1, n_total_samples))
        event_wade, event_wfde, _ = _weighted_mean(ev_stats)
        state_wade, state_wfde, _ = _weighted_mean(st_stats)

    # ---- save per-label CSV rows (event/state) ----
    if labels_lut is not None and n_total_samples > 0:
        coverage = {
            "matched": int(n_matched),
            "total": int(n_total_samples),
            "matched_ratio": float(n_matched / max(1, n_total_samples)),
        }

        if save_event_path is not None:
            save_event_path.parent.mkdir(parents=True, exist_ok=True)
            EVENT_LABELS = ["none", "cut_in", "merging", "diverging",
                            "simple_lane_change", "lane_change_other", "unknown"]

            wade, wfde, _ = _weighted_mean(ev_stats)
            row = dict(coverage)

            for lab in EVENT_LABELS:
                c, a, f = _per_label(ev_stats, lab)
                row[f"{lab}_count"] = c
                row[f"{lab}_ADE"] = a
                row[f"{lab}_FDE"] = f

            df_row = pd.DataFrame([row])
            header = not save_event_path.exists()
            df_row.to_csv(save_event_path, mode="a", header=header, index=False)

        if save_state_path is not None:
            save_state_path.parent.mkdir(parents=True, exist_ok=True)
            STATE_LABELS = ["free_flow", "dense", "car_following",
                            "ramp_driving", "other", "unknown"]

            wade, wfde, _ = _weighted_mean(st_stats)
            row = dict(coverage)

            for lab in STATE_LABELS:
                c, a, f = _per_label(st_stats, lab)
                row[f"{lab}_count"] = c
                row[f"{lab}_ADE"] = a
                row[f"{lab}_FDE"] = f

            df_row = pd.DataFrame([row])
            header = not save_state_path.exists()
            df_row.to_csv(save_state_path, mode="a", header=header, index=False)

    return {
        "loss": float(overall_loss),
        "ade": float(overall_ade),
        "fde": float(overall_fde),

        "n_samples": int(n_samples),

        "matched": int(n_matched) if labels_lut is not None else -1,
        "matched_ratio": float(matched_ratio) if np.isfinite(matched_ratio) else float("nan"),
    }


def train_one_epoch(
    model, loader, device, optimizer, scheduler, scaler,
    use_amp, predict_delta, grad_clip_norm,
    w_traj, w_fde, w_cls, global_step, log_every, epoch
):
    model.train()
    total_loss, total_ade, total_fde = 0.0, 0.0, 0.0
    n = 0

    pbar = tqdm(loader, desc=f"train ep{epoch}", dynamic_ncols=True, leave=False)

    for it, batch in enumerate(pbar):
        x_ego = batch["x_ego"].to(device, non_blocking=True)
        x_nb  = batch["x_nb"].to(device, non_blocking=True)
        nb_mask = batch["nb_mask"].to(device, non_blocking=True)
        y_abs = batch["y"].to(device, non_blocking=True)
        x_last_abs = batch["x_last_abs"].to(device, non_blocking=True)

        if _any_nonfinite(x_ego) or _any_nonfinite(x_nb) or _any_nonfinite(y_abs) or _any_nonfinite(x_last_abs):
            print(f"[BAD INPUT] ep={epoch} it={it}")
            print("  x_ego:", _finite_stats(x_ego))
            print("  x_nb :", _finite_stats(x_nb))
            print("  y_abs:", _finite_stats(y_abs))
            print("  x_last_abs:", _finite_stats(x_last_abs))
            raise RuntimeError("Non-finite in inputs")

        optimizer.zero_grad(set_to_none=True)

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
                    w_traj=w_traj, w_fde=w_fde, w_cls=w_cls
                )
            else:
                loss = trajectory_loss(
                    pred=pred, y_abs=y_abs, x_last_abs=x_last_abs,
                    predict_delta=predict_delta, w_traj=w_traj, w_fde=w_fde
                )
                best_idx = None

        if _any_nonfinite(loss):
            print(f"[BAD LOSS] ep={epoch} it={it} loss={loss.item()}")
            raise RuntimeError("Non-finite loss")

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

        if not _check_params_finite(model):
            print(f"[BAD PARAM] ep={epoch} it={it} params became non-finite AFTER step()")
            raise RuntimeError("Params exploded to NaN/Inf")

        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            if pred.dim() == 4:
                if predict_delta:
                    pred_abs_all = torch.cumsum(pred, dim=2) + x_last_abs[:, None, None, :]
                else:
                    pred_abs_all = pred
                pred_abs = pred_abs_all[torch.arange(pred.shape[0], device=pred.device), best_idx]
            else:
                pred_abs = delta_to_abs(pred, x_last_abs) if predict_delta else pred

            a = ade(pred_abs, y_abs)
            f = fde(pred_abs, y_abs)

        total_loss += float(loss.item())
        total_ade += float(a.item())
        total_fde += float(f.item())
        n += 1
        global_step += 1

        if it % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss.item():.4f}", ADE=f"{a.item():.3f}", FDE=f"{f.item():.3f}", lr=f"{lr:.1e}")

    return {"loss": total_loss/max(1,n), "ade": total_ade/max(1,n), "fde": total_fde/max(1,n), "global_step_end": global_step}