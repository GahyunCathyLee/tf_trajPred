# src/log.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from datetime import datetime

import pandas as pd


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def log_eval_to_csv(
    *,
    csv_out: Path,
    cfg: Dict[str, Any],
    cfg_path: Path,
    ckpt_path: Path,
    split: str,
    mode: str,
    tag: str,
    device: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    use_amp: bool,
    stats_path: Path,
    use_ego_static: bool,
    use_nb_static: bool,
    metrics: Dict[str, Any],
) -> None:
    """Append eval results to CSV (compatible with old results/results.csv style)."""
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    use_neighbors = bool(cfg.get("model", {}).get("use_neighbors", True))
    predict_delta = bool(cfg.get("model", {}).get("predict_delta", False))
    loss_cfg = cfg.get("loss", {})

    row: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "split": str(split),
        "mode": str(mode),
        "tag": str(tag),

        "config": str(cfg_path),
        "ckpt_path": str(ckpt_path),
        "stats_path": str(stats_path),

        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "seed": int(seed),
        "use_amp": bool(use_amp),

        "use_ego_static": bool(use_ego_static),
        "use_nb_static": bool(use_nb_static),
        "use_neighbors": bool(use_neighbors),

        "w_traj": _safe_float(loss_cfg.get("w_traj", 1.0)),
        "w_fde": _safe_float(loss_cfg.get("w_fde", 1.0)),
        "w_cls": _safe_float(loss_cfg.get("w_cls", 1.0)),

        "n_samples": int(metrics.get("n_samples", metrics.get("num_samples", -1))),
        "loss": _safe_float(metrics.get("loss")),
        "ade": _safe_float(metrics.get("ade")),
        "fde": _safe_float(metrics.get("fde")),
        # [NEW] Add RMSE logging
        "rmse": _safe_float(metrics.get("rmse")),
        "rmse_1s": _safe_float(metrics.get("rmse_1s")),
        "rmse_2s": _safe_float(metrics.get("rmse_2s")),
        "rmse_3s": _safe_float(metrics.get("rmse_3s")),
        "rmse_4s": _safe_float(metrics.get("rmse_4s")),
        "rmse_5s": _safe_float(metrics.get("rmse_5s")),
        
        "vel": _safe_float(metrics.get("vel")),
        "acc": _safe_float(metrics.get("acc")),
        "jerk": _safe_float(metrics.get("jerk")),

        "matched_ratio": _safe_float(metrics.get("matched_ratio")),
        
        # [NEW] Latency per sample
        "latency_ms": _safe_float(metrics.get("latency_ms")),
    }

    df = pd.DataFrame([row])
    header = not csv_out.exists()
    df.to_csv(csv_out, mode="a", header=header, index=False)