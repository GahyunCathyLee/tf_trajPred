#!/usr/bin/env python3
"""
exiD tracks.csv -> NPZ preprocessing for Transformer trajectory prediction (ego + 8 relation slots).

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import re


# =========================
# filtering / clipping knobs
# =========================
LANEWIDTH_DROP_TH = 3.0   # laneWidth < 3.0 -> drop window (history)
LEAD_TTC_CAP = 90.0       # seconds
LEAD_THW_CAP = 20.0       # seconds


NEIGHBOR_COLS_8 = [
    "leadId",
    "rearId",
    "leftLeadId",
    "leftAlongsideId",
    "leftRearId",
    "rightLeadId",
    "rightAlongsideId",
    "rightRearId",
]

# Vocabulary for class one-hot.
# Anything not in this list is mapped to "other".
CLASS_VOCAB = [
    "car",
    "truck",
    "van",
    "bus",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "other",
]
VRU_CLASSES = {"motorcycle", "bicycle", "pedestrian"}


def compute_downsample_step(source_hz: float, target_hz: float) -> int:
    step = int(round(float(source_hz) / float(target_hz)))
    if step <= 0:
        raise ValueError(f"Invalid downsample step: source_hz={source_hz}, target_hz={target_hz}")
    return step


def _safe_float(a: np.ndarray, default: float = 0.0) -> np.ndarray:
    """Replace NaN/inf with default."""
    a = a.astype(np.float32, copy=False)
    bad = ~np.isfinite(a)
    if np.any(bad):
        a = a.copy()
        a[bad] = default
    return a


def _normalize_ramp_type_series(s: pd.Series) -> np.ndarray:
    """
    Map ramp_type string -> int code:
      0 none
      1 onramp
      2 offramp
      3 unknown
    """
    x = s.fillna("").astype(str).str.strip().str.lower()
    x = x.replace(
        {
            "on_ramp": "onramp",
            "on-ramp": "onramp",
            "off_ramp": "offramp",
            "off-ramp": "offramp",
            "nan": "",
            "null": "",
        }
    )

    out = np.full(len(x), 3, dtype=np.int8)  # unknown default
    out[x == "none"] = 0
    out[x == "onramp"] = 1
    out[x == "offramp"] = 2
    # empty string stays unknown
    return out

def _parse_recording_id_from_tracks_stem(stem: str) -> int | None:
    # stem examples: "00_tracks", "17_tracks_with_ramp", etc.
    m = re.match(r"^(\d+)", stem)          # 留� �� �レ옄
    if not m:
        m = re.search(r"(\d+)", stem)      # fallback: �꾨Т �レ옄
    if not m:
        return None
    return int(m.group(1))

def _onehot4(code: np.ndarray) -> np.ndarray:
    """code in {0,1,2,3} -> onehot (N,4) float32."""
    oh = np.zeros((len(code), 4), dtype=np.float32)
    idx = code.astype(np.int64)
    ok = (idx >= 0) & (idx < 4)
    oh[np.arange(len(code))[ok], idx[ok]] = 1.0
    return oh


def _normalize_class_name(s: str) -> str:
    s = (s or "").strip().lower()
    if s == "" or s in {"nan", "null"}:
        return "other"
    # common variants
    if s in {"passenger car", "passengercar"}:
        s = "car"
    if s not in CLASS_VOCAB:
        s = "other"
    return s


def _build_class_map(meta_csv: Path) -> Tuple[Dict[int, str], Dict[str, int], np.ndarray]:
    """
    Returns:
      class_map: trackId -> normalized class string
      class_to_idx: class string -> index
      class_names: (C,) np.array
    """
    dfm = pd.read_csv(meta_csv, low_memory=False)
    if "trackId" not in dfm.columns or "class" not in dfm.columns:
        raise RuntimeError(f"{meta_csv} must contain columns: trackId, class")

    class_map: Dict[int, str] = {}
    for tid, cls in zip(dfm["trackId"].astype(int).tolist(), dfm["class"].astype(str).tolist()):
        class_map[int(tid)] = _normalize_class_name(cls)

    class_names = np.array(CLASS_VOCAB, dtype=object)
    class_to_idx = {c: i for i, c in enumerate(CLASS_VOCAB)}
    return class_map, class_to_idx, class_names


def _onehot_class(class_name: str, class_to_idx: Dict[str, int]) -> np.ndarray:
    C = len(class_to_idx)
    oh = np.zeros((C,), dtype=np.float32)
    idx = class_to_idx.get(class_name, class_to_idx["other"])
    oh[idx] = 1.0
    return oh


def build_vehicle_maps(
    track_ids: np.ndarray, frames: np.ndarray
) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, Dict[int, int]]]:
    """
    Build:
      uniq_ids: (V,)
      veh_rows: vid -> row indices (contiguous blocks if sorted by vid then frame)
      frame_to_pos: vid -> {frame -> position_in_veh_rows}
    Requires arrays are sorted by (trackId, frame).
    """
    uniq_ids, start_idx, counts = np.unique(track_ids, return_index=True, return_counts=True)
    veh_rows: Dict[int, np.ndarray] = {}
    frame_to_pos: Dict[int, Dict[int, int]] = {}

    for vid, st, ct in zip(uniq_ids.tolist(), start_idx.tolist(), counts.tolist()):
        idxs = np.arange(st, st + ct, dtype=np.int32)
        veh_rows[int(vid)] = idxs
        f = frames[idxs]
        frame_to_pos[int(vid)] = {int(fr): int(i) for i, fr in enumerate(f.tolist())}

    return uniq_ids.astype(np.int32), veh_rows, frame_to_pos


def make_windows_for_tracks_csv(
    tracks_csv: Path,
    out_path: Path,
    source_hz: float,
    target_hz: float,
    history_sec: float,
    future_sec: float,
    stride_sec: float,
    min_speed_mps: float = 0.0,
    drop_vru: bool = True,
    keep_only_vru_cases: bool = False,
    recording_id: int | None = None
) -> Tuple[int, Optional[str]]:
    try:
        df = pd.read_csv(tracks_csv, low_memory=False)
    except Exception as e:
        return 0, f"Failed to read {tracks_csv}: {e}"

    # meta csv next to tracks
    meta_csv = tracks_csv.with_name(tracks_csv.name.replace("_tracks.csv", "_tracksMeta.csv"))
    if not meta_csv.exists():
        return 0, f"Missing tracksMeta: {meta_csv.name}"

    try:
        class_map, class_to_idx, class_names = _build_class_map(meta_csv)
    except Exception as e:
        return 0, f"Failed to read/parse {meta_csv.name}: {e}"

    # ---- required columns check ----
    required = [
        "trackId",
        "frame",
        # main x/y for TP
        "xCenter",
        "yCenter",
        "xVelocity",
        "yVelocity",
        "xAcceleration",
        "yAcceleration",
        # ego lane/context
        "latLaneCenterOffset",
        "laneWidth",
        "laneChange",
        # ego safety
        "leadDHW",
        "leadDV",
        "leadTHW",
        "leadTTC",
        # ramp context
        "ramp_type",
        # size
        "width",
        "length",
    ] + NEIGHBOR_COLS_8

    missing = [c for c in required if c not in df.columns]
    if missing:
        return 0, f"Missing required columns: {missing}"

    # ---- sort & downsample ----
    df = df.sort_values(["trackId", "frame"], kind="mergesort").reset_index(drop=True)

    ds_step = compute_downsample_step(source_hz, target_hz)
    df = df[((df["frame"].astype(int) - 1) % ds_step) == 0].copy()
    if df.empty:
        return 0, "Downsample produced no rows."

    # ---- extract arrays ----
    track_ids = df["trackId"].astype(np.int32).to_numpy()
    frames = df["frame"].astype(np.int32).to_numpy()

    x = df["xCenter"].astype(np.float32).to_numpy()
    y = df["yCenter"].astype(np.float32).to_numpy()
    xv = df["lonVelocity"].astype(np.float32).to_numpy()
    yv = df["latVelocity"].astype(np.float32).to_numpy()
    xa = df["lonAcceleration"].astype(np.float32).to_numpy()
    ya = df["latAcceleration"].astype(np.float32).to_numpy()

    lat_center_off = df["latLaneCenterOffset"].astype(np.float32).to_numpy()
    lane_w = df["laneWidth"].astype(np.float32).to_numpy()
    lane_change = df["laneChange"].astype(np.float32).to_numpy()

    width_arr = _safe_float(df["width"].to_numpy(np.float32), 0.0)
    length_arr = _safe_float(df["length"].to_numpy(np.float32), 0.0)

    lead_dhw = _safe_float(df["leadDHW"].to_numpy(np.float32), 0.0)
    lead_dv = _safe_float(df["leadDV"].to_numpy(np.float32), 0.0)
    lead_thw = _safe_float(df["leadTHW"].to_numpy(np.float32), 0.0)
    lead_ttc = _safe_float(df["leadTTC"].to_numpy(np.float32), 0.0)

    # neighbor ids (take FIRST if '287;285')
    def _parse_id_cell(v) -> int:
        if pd.isna(v):
            return -1
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return -1
            return int(v)
        s = str(v).strip()
        if s == "" or s.lower() in {"nan", "null"}:
            return -1
        s = s.split(";")[0].split(",")[0].strip()
        try:
            return int(float(s))
        except Exception:
            return -1

    nb_ids_all = (
        df[NEIGHBOR_COLS_8]
        .apply(lambda col: col.map(_parse_id_cell))
        .astype(np.int32)
        .to_numpy()
    )

    # ---- ramp onehot etc ----
    ramp_code = _normalize_ramp_type_series(df["ramp_type"])
    ramp_oh = _onehot4(ramp_code)

    eps = 1e-6
    half_w = np.maximum(lane_w * 0.5, eps).astype(np.float32)
    norm_off = (lat_center_off / half_w).astype(np.float32)

    speed = np.sqrt(xv * xv + yv * yv) if min_speed_mps > 0.0 else None

    # ---- per-vehicle maps ----
    uniq_ids, veh_rows, frame_to_pos = build_vehicle_maps(track_ids, frames)

    # per-vehicle static lookup (from first row after downsample)
    veh_width: Dict[int, float] = {}
    veh_length: Dict[int, float] = {}
    veh_class: Dict[int, str] = {}
    for vid in uniq_ids.tolist():
        idx0 = int(veh_rows[int(vid)][0])
        veh_width[int(vid)] = float(width_arr[idx0])
        veh_length[int(vid)] = float(length_arr[idx0])
        veh_class[int(vid)] = class_map.get(int(vid), "other")

    def is_vru_trackid(tid: int) -> bool:
        return veh_class.get(int(tid), "other") in VRU_CLASSES

    T = int(round(history_sec * target_hz))
    Tf = int(round(future_sec * target_hz))
    stride = max(1, int(round(stride_sec * target_hz)))

    if T <= 0 or Tf <= 0:
        return 0, f"Invalid T/Tf: T={T}, Tf={Tf}."

    K = 8
    ego_dim = 18
    nb_dim = 6
    C = len(CLASS_VOCAB)
    static_dim = 2 + C

    win_len = T + Tf
    expected_gap = ds_step

    # ==========================================================
    # Recording-level coordinate shift (global min over all frames)
    # ==========================================================
    min_x = float(np.nanmin(x)) if x.size > 0 else 0.0
    min_y = float(np.nanmin(y)) if y.size > 0 else 0.0
    if not np.isfinite(min_x):
        min_x = 0.0
    if not np.isfinite(min_y):
        min_y = 0.0

    x_shift = (x - min_x).astype(np.float32)
    y_shift = (y - min_y).astype(np.float32)

    # ---- collect samples ----
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    NB_list: List[np.ndarray] = []
    NBmask_list: List[np.ndarray] = []
    EgoStatic_list: List[np.ndarray] = []
    NBStatic_list: List[np.ndarray] = []
    trackId_list: List[int] = []
    t0_list: List[int] = []
    YV_list: List[np.ndarray] = []   # (Tf,2) lon/lat velocity
    YA_list: List[np.ndarray] = []   # (Tf,2) lon/lat acceleration

    for vid in uniq_ids.tolist():
        idxs = veh_rows[int(vid)]
        f = frames[idxs]
        n = len(idxs)
        if n < win_len:
            continue

        ego_is_vru = is_vru_trackid(int(vid))

        for i in range(0, n - win_len + 1, stride):
            f0 = int(f[i])
            f_last = int(f[i + win_len - 1])
            if f_last - f0 != expected_gap * (win_len - 1):
                continue

            hist_rows = idxs[i : i + T]
            fut_rows  = idxs[i + T : i + win_len]

            if speed is not None and float(np.nanmax(speed[hist_rows])) < float(min_speed_mps):
                continue

            # ==========================================================
            # drop window if any laneWidth in HISTORY is < 3.0 or non-finite
            # ==========================================================
            lw_hist = lane_w[hist_rows]
            if (not np.all(np.isfinite(lw_hist))) or np.any(lw_hist < LANEWIDTH_DROP_TH):
                continue

            # window-level VRU detection (history)
            any_nb_vru = False
            if not ego_is_vru:
                for t in range(T):
                    row_ids = nb_ids_all[hist_rows[t]]
                    for nid in row_ids.tolist():
                        if nid >= 0 and is_vru_trackid(int(nid)):
                            any_nb_vru = True
                            break
                    if any_nb_vru:
                        break
            vru_involving = ego_is_vru or any_nb_vru

            # filtering policy
            if keep_only_vru_cases:
                if not vru_involving:
                    continue
            else:
                if drop_vru and vru_involving:
                    continue

            lead_exists = (nb_ids_all[hist_rows, 0] > -1).astype(np.float32)

            # take window slices
            dhw_w = lead_dhw[hist_rows].astype(np.float32, copy=False)
            dv_w  = lead_dv[hist_rows].astype(np.float32, copy=False)
            thw_w = lead_thw[hist_rows].astype(np.float32, copy=False)
            ttc_w = lead_ttc[hist_rows].astype(np.float32, copy=False)

            # sanitize: non-finite already handled by _safe_float, but keep defensive
            dhw_w = np.where(np.isfinite(dhw_w), dhw_w, 0.0).astype(np.float32)
            dv_w  = np.where(np.isfinite(dv_w),  dv_w,  0.0).astype(np.float32)
            thw_w = np.where(np.isfinite(thw_w), thw_w, 0.0).astype(np.float32)
            ttc_w = np.where(np.isfinite(ttc_w), ttc_w, 0.0).astype(np.float32)

            dhw_w = np.where(dhw_w > 0.0, dhw_w, 0.0).astype(np.float32)
            thw_w = np.where(thw_w > 0.0, thw_w, 0.0).astype(np.float32)
            ttc_w = np.where(ttc_w > 0.0, ttc_w, 0.0).astype(np.float32)

            # clip only when lead exists
            thw_w = np.where(lead_exists > 0.0, np.minimum(thw_w, LEAD_THW_CAP), 0.0).astype(np.float32)
            ttc_w = np.where(lead_exists > 0.0, np.minimum(ttc_w, LEAD_TTC_CAP), 0.0).astype(np.float32)
            dhw_w = np.where(lead_exists > 0.0, dhw_w, 0.0).astype(np.float32)
            dv_w  = np.where(lead_exists > 0.0, dv_w, 0.0).astype(np.float32)

            # ego history (T,18)  --- x/y shifted by recording-global min ---
            ego_hist = np.stack(
                [
                    x_shift[hist_rows],
                    y_shift[hist_rows],
                    xv[hist_rows],
                    yv[hist_rows],
                    xa[hist_rows],
                    ya[hist_rows],
                    lat_center_off[hist_rows],
                    lane_change[hist_rows],
                    norm_off[hist_rows],
                    dhw_w,          # leadDHW (gated)
                    dv_w,           # leadDV  (gated)
                    thw_w,          # leadTHW (gated + clip 20)
                    ttc_w,          # leadTTC (gated + clip 90)
                    lead_exists,    # hasLead/lead_exists (same index �좎�)
                    ramp_oh[hist_rows, 0],
                    ramp_oh[hist_rows, 1],
                    ramp_oh[hist_rows, 2],
                    ramp_oh[hist_rows, 3],
                ],
                axis=-1,
            ).astype(np.float32)

            # future label (Tf,2) --- shifted by recording-global min ---
            y_fut = np.stack(
                [x_shift[fut_rows], y_shift[fut_rows]],
                axis=-1
            ).astype(np.float32)

            # future GT lon/lat velocity & acceleration (Tf,2)
            y_fut_vel = np.stack(
                [xv[fut_rows], yv[fut_rows]],
                axis=-1
            ).astype(np.float32)

            y_fut_acc = np.stack(
                [xa[fut_rows], ya[fut_rows]],
                axis=-1
            ).astype(np.float32)

            # ego static: (2+C,)
            ego_w = veh_width.get(int(vid), 0.0)
            ego_l = veh_length.get(int(vid), 0.0)
            ego_cls = veh_class.get(int(vid), "other")
            ego_static = np.concatenate(
                [np.array([ego_w, ego_l], dtype=np.float32), _onehot_class(ego_cls, class_to_idx)],
                axis=0,
            ).astype(np.float32)

            # neighbors history (T,K,6) + mask (T,K)
            nb_hist = np.zeros((T, K, nb_dim), dtype=np.float32)
            nb_mask = np.zeros((T, K), dtype=bool)

            # neighbors static aligned with slot ids per time: (T,K,2+C)
            nb_static = np.zeros((T, K, static_dim), dtype=np.float32)

            hist_frames = frames[hist_rows]

            for t in range(T):
                fr = int(hist_frames[t])

                row_nb_ids = nb_ids_all[hist_rows[t]]  # (K,)
                for k in range(K):
                    nid = int(row_nb_ids[k])
                    if nid < 0:
                        continue

                    # static for this slot at this time
                    nw = veh_width.get(nid, 0.0)
                    nl = veh_length.get(nid, 0.0)
                    ncls = veh_class.get(nid, "other")
                    nb_static[t, k, :] = np.concatenate(
                        [np.array([nw, nl], dtype=np.float32), _onehot_class(ncls, class_to_idx)],
                        axis=0,
                    )

                    # dynamic requires same-frame lookup
                    d = frame_to_pos.get(nid)
                    if d is None:
                        continue
                    pos_in = d.get(fr)
                    if pos_in is None:
                        continue
                    nb_row = veh_rows[nid][pos_in]

                    nb_mask[t, k] = True

                    # neighbors use ego-relative coordinates (nb - ego) to match highD pipeline
                    ego_row = hist_rows[t]
                    nb_hist[t, k, 0] = float(x_shift[nb_row] - x_shift[ego_row])
                    nb_hist[t, k, 1] = float(y_shift[nb_row] - y_shift[ego_row])
                    nb_hist[t, k, 2] = float(xv[nb_row] - xv[ego_row])
                    nb_hist[t, k, 3] = float(yv[nb_row] - yv[ego_row])
                    nb_hist[t, k, 4] = float(xa[nb_row] - xa[ego_row])
                    nb_hist[t, k, 5] = float(ya[nb_row] - ya[ego_row])

            X_list.append(ego_hist)
            Y_list.append(y_fut)
            YV_list.append(y_fut_vel)
            YA_list.append(y_fut_acc)
            NB_list.append(nb_hist)
            NBmask_list.append(nb_mask)
            EgoStatic_list.append(ego_static)
            NBStatic_list.append(nb_static)
            trackId_list.append(int(vid))
            t0_list.append(int(frames[hist_rows[0]]))

    if len(X_list) == 0:
        return 0, "No valid samples produced."

    x_hist = np.stack(X_list, axis=0).astype(np.float32)
    y_fut = np.stack(Y_list, axis=0).astype(np.float32)
    y_fut_vel = np.stack(YV_list, axis=0).astype(np.float32)  # (N,Tf,2)
    y_fut_acc = np.stack(YA_list, axis=0).astype(np.float32)  # (N,Tf,2)
    nb_hist = np.stack(NB_list, axis=0).astype(np.float32)
    nb_mask = np.stack(NBmask_list, axis=0).astype(bool)
    ego_static = np.stack(EgoStatic_list, axis=0).astype(np.float32)
    nb_static = np.stack(NBStatic_list, axis=0).astype(np.float32)  # (N,T,K,static_dim)

    trackId_arr = np.asarray(trackId_list, dtype=np.int32)
    t0_arr = np.asarray(t0_list, dtype=np.int32)

    N = int(len(trackId_list))
    recordingId_arr = None
    if recording_id is not None:
        recordingId_arr = np.full((N,), int(recording_id), dtype=np.int32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        x_hist=x_hist,
        y_fut=y_fut,
        y_fut_vel=y_fut_vel, 
        y_fut_acc=y_fut_acc,
        nb_hist=nb_hist,
        nb_mask=nb_mask,
        ego_static=ego_static,
        nb_static=nb_static,
        trackId=trackId_arr,
        t0_frame=t0_arr,
        class_names=class_names,
        # meta
        recordingId=recordingId_arr if recordingId_arr is not None else np.zeros((N,), dtype=np.int32),
        T=np.array([T], dtype=np.int32),
        Tf=np.array([Tf], dtype=np.int32),
        K=np.array([K], dtype=np.int32),
        ego_dim=np.array([ego_dim], dtype=np.int32),
        nb_dim=np.array([nb_dim], dtype=np.int32),
        static_dim=np.array([static_dim], dtype=np.int32),
        target_hz=np.array([target_hz], dtype=np.float32),
        source_hz=np.array([source_hz], dtype=np.float32),
        ds_step=np.array([ds_step], dtype=np.int32),
        history_sec=np.array([history_sec], dtype=np.float32),
        future_sec=np.array([future_sec], dtype=np.float32),
        stride_sec=np.array([stride_sec], dtype=np.float32),
        drop_vru=np.array([1 if drop_vru else 0], dtype=np.int32),
        keep_only_vru_cases=np.array([1 if keep_only_vru_cases else 0], dtype=np.int32),
        origin_min_xy=np.array([min_x, min_y], dtype=np.float32),
        laneWidth_drop_th=np.array([LANEWIDTH_DROP_TH], dtype=np.float32),
        leadTTC_cap=np.array([LEAD_TTC_CAP], dtype=np.float32),
        leadTHW_cap=np.array([LEAD_THW_CAP], dtype=np.float32),
    )
    
    return int(x_hist.shape[0]), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks_dir", type=str, default="raw/", help="Directory containing *_tracks.csv (e.g., raw/)")
    ap.add_argument("--glob", type=str, default="*_tracks.csv", help="Glob pattern (default: *_tracks.csv)")
    ap.add_argument(
        "--out_root",
        type=str,
        default="data_npz",
        help="Root directory for generated npz folders",
    )

    ap.add_argument("--source_hz", type=float, default=25.0)
    ap.add_argument("--target_hz", type=float, default=5.0)
    ap.add_argument("--history_sec", type=float, default=2.0)
    ap.add_argument("--future_sec", type=float, default=5.0)
    ap.add_argument("--stride_sec", type=float, default=1.0)
    ap.add_argument("--min_speed_mps", type=float, default=0.0)

    ap.add_argument("--drop_vru", action="store_true", help="Drop windows involving VRU (default behavior).")
    ap.add_argument(
        "--keep_only_vru_cases",
        action="store_true",
        help="Keep ONLY windows involving VRU (for corner-case split).",
    )

    args = ap.parse_args()

    # ---- auto out_dir naming ----
    T_sec = int(round(args.history_sec))
    Tf_sec = int(round(args.future_sec))
    hz = int(round(args.target_hz))

    out_dir = (
        Path(args.out_root)
        / f"exid_T{T_sec}_Tf{Tf_sec}_hz{hz}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # default behavior: drop VRU unless keep_only_vru_cases enabled
    drop_vru = True if (not args.keep_only_vru_cases) else False
    if args.drop_vru:
        drop_vru = True
    if args.keep_only_vru_cases:
        drop_vru = False

    tracks_dir = Path(args.tracks_dir)

    files = sorted(tracks_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"No files found in {tracks_dir} with pattern {args.glob}")

    total = 0
    ok_files = 0
    for tracks_csv in files:
        out_path = out_dir / f"exid_{tracks_csv.stem.replace('_tracks','')}.npz"
        rid = _parse_recording_id_from_tracks_stem(tracks_csv.stem)
        n, err = make_windows_for_tracks_csv(
            tracks_csv=tracks_csv,
            out_path=out_path,
            source_hz=args.source_hz,
            target_hz=args.target_hz,
            history_sec=args.history_sec,
            future_sec=args.future_sec,
            stride_sec=args.stride_sec,
            min_speed_mps=args.min_speed_mps,
            drop_vru=drop_vru,
            keep_only_vru_cases=args.keep_only_vru_cases,
            recording_id=rid
        )
        if err is not None:
            print(f"[SKIP] {tracks_csv.name}  reason={err}")
            continue

        print(f"[OK] {tracks_csv.name} -> {out_path.name}  samples={n}")
        total += n
        ok_files += 1

    print(f"[DONE] ok_files={ok_files}/{len(files)} total_samples={total}")
    if args.keep_only_vru_cases:
        print("[INFO] keep_only_vru_cases enabled: produced VRU-involving subset only.")
    else:
        print(f"[INFO] drop_vru={drop_vru}")


if __name__ == "__main__":
    main()