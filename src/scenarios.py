from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Subset # [추가] Subset 임포트

from pathlib import Path

def load_window_labels_csv(path: Path):
    """
    window_labels.csv:
      recordingId, trackId, t0_frame, ... , event_label, state_label
    -> dict[(recordingId,trackId,t0_frame)] = {"event_label":..., "state_label":...}
    """
    if not path.exists():
        return None

    df = pd.read_csv(path)
    required = {"recordingId", "trackId", "t0_frame"}
    if not required.issubset(set(df.columns)):
        print(f"[WARN] window_labels.csv missing keys: {required - set(df.columns)} -> stratified eval disabled")
        return None

    if "event_label" not in df.columns and "state_label" not in df.columns:
        print("[WARN] window_labels.csv has no event_label/state_label -> stratified eval disabled")
        return None

    cols = ["recordingId", "trackId", "t0_frame"]
    if "event_label" in df.columns:
        cols.append("event_label")
    if "state_label" in df.columns:
        cols.append("state_label")
    df = df[cols]

    lut = {}
    for r in df.itertuples(index=False):
        rid = int(getattr(r, "recordingId"))
        tid = int(getattr(r, "trackId"))
        t0  = int(getattr(r, "t0_frame"))
        lut[(rid, tid, t0)] = {
            "event_label": getattr(r, "event_label", None),
            "state_label": getattr(r, "state_label", None),
        }
    print(f"[INFO] Loaded window labels: {len(lut):,} from {path}")
    return lut

LabelMode = Literal["event", "state"]

def _get_key_from_item(item: Dict[str, Any]) -> Optional[Tuple[int,int,int]]:
    """
    item: dataset[i] 결과 dict
    expects item["meta"] with recordingId/trackId/t0_frame (torch scalar or int)
    """
    meta = item.get("meta", None)
    if meta is None:
        return None
    try:
        rid = int(meta["recordingId"])  # torch scalar OK
        tid = int(meta["trackId"])
        t0  = int(meta["t0_frame"])
        return (rid, tid, t0)
    except Exception:
        return None

def _extract_key_from_meta(meta: Dict[str, Any]):
    try:
        rid = int(meta["recordingId"])
        tid = int(meta["trackId"])
        t0  = int(meta["t0_frame"])
        return (rid, tid, t0)
    except Exception:
        return None

def _get_meta_fast(dataset, i: int):
    # 1) direct (PtWindowDataset 등)
    if hasattr(dataset, "get_meta"):
        return dataset.get_meta(i)

    # 2) Subset 처리 [추가된 부분]
    # Subset은 .dataset(부모)과 .indices(맵핑)를 가짐. 재귀적으로 호출.
    if isinstance(dataset, Subset):
        real_idx = dataset.indices[i]
        return _get_meta_fast(dataset.dataset, real_idx)

    # 3) ConcatDataset 처리
    if isinstance(dataset, ConcatDataset):
        if not hasattr(dataset, "_cached_meta_router"):
            cum = np.asarray(dataset.cumulative_sizes, dtype=np.int64)
            N = int(cum[-1])
            ds_id = np.searchsorted(cum, np.arange(N, dtype=np.int64), side="right")
            starts = np.concatenate(([0], cum[:-1]))
            local = np.arange(N, dtype=np.int64) - starts[ds_id]
            dataset._cached_meta_router = (ds_id, local)

        ds_id, local = dataset._cached_meta_router
        j = int(ds_id[i])
        li = int(local[i])
        sub = dataset.datasets[j]
        # 재귀 호출로 내부가 Subset이어도 처리 가능
        return _get_meta_fast(sub, li)

    # 4) fallback (느리지만 확실한 방법)
    item = dataset[i]
    return item.get("meta", None)

def build_sample_weights(
    dataset,
    labels_lut: Dict[Tuple[int,int,int], Dict[str, Any]],
    mode: LabelMode = "event",
    alpha: float = 0.5,
    unknown_weight: float = 0.0,
    clip_max: Optional[float] = None,
    log: bool = True
) -> torch.DoubleTensor:
    """
    Build per-index sampling weights for (WeightedRandomSampler).

    alpha:
      - 1.0 => inverse frequency (strong)
      - 0.5 => inverse sqrt frequency (recommended start)
    unknown_weight:
      - 0.0 => exclude keys not found / unknown labels
      - >0  => include them with small probability
    clip_max:
      - optional cap for extremely rare labels
    """
    labels = []

    # 전체 데이터셋 순회 (Subset인 경우 len은 Subset 크기)
    for i in range(len(dataset)):
        meta = _get_meta_fast(dataset, i)

        if meta is None:
            labels.append("unknown")
            continue

        key = _extract_key_from_meta(meta)
        if key is None:
            labels.append("unknown")
            continue

        rec = labels_lut.get(key, None)
        if rec is None:
            labels.append("unknown")
            continue

        lab = rec.get("event_label" if mode == "event" else "state_label", None)

        if mode == "event":
            s_lab = str(lab) if lab is not None else "unknown"
            if s_lab in ["simple_lane_change", "lane_change_other"]:
                lab = "lane_change"

        if lab is None:
            lab = "unknown"

        labels.append(str(lab))

    # label count
    known = [l for l in labels if l != "unknown"]
    uniq, cnt = np.unique(known, return_counts=True)
    count = {u: int(c) for u, c in zip(uniq, cnt)}

    # weights
    w = np.zeros(len(dataset), dtype=np.float64)
    for i, lab in enumerate(labels):
        if lab == "unknown":
            w[i] = float(unknown_weight)
        else:
            w[i] = 1.0 / (count[lab] ** float(alpha))

    if clip_max is not None:
        w = np.minimum(w, float(clip_max))

    if w.sum() <= 0:
        w[:] = 1.0

    w_t = torch.from_numpy(w).to(torch.double)

    # ---- LOGGING: raw label distribution + expected sampling distribution ----
    if log:
        total_known = sum(count.values())
        # raw proportions
        raw = sorted(count.items(), key=lambda x: (-x[1], x[0]))
        print(f"[SCENARIO-SAMPLING] mode={mode} alpha={alpha} known={total_known:,} unknown={len(labels)-total_known:,}")
        print("[SCENARIO-SAMPLING] raw label counts (top):")
        for lab, c in raw[:20]:
            print(f"  - {lab:>16s}: {c:8d} ({c/total_known*100:6.2f}%)")

        # expected proportions after weighting (approx)
        # expected mass per label ~ sum_i w_i over that label
        mass = {}
        for lab, c in count.items():
            mass[lab] = c * (1.0 / (c ** float(alpha)))  # c * c^{-alpha} = c^{1-alpha}
        mass_sum = sum(mass.values()) if mass else 1.0
        exp = sorted(mass.items(), key=lambda x: (-x[1], x[0]))
        print("[SCENARIO-SAMPLING] expected sampling share (approx, from weights):")
        for lab, m in exp[:20]:
            print(f"  - {lab:>16s}: {m/mass_sum*100:6.2f}%")

    return w_t