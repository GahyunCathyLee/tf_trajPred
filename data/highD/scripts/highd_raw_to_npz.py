#!/usr/bin/env python3
"""
highD raw CSV -> NPZ in the SAME schema as the *modified* exiD preprocessor.

Target schema (per recording NPZ):
- x_hist   : (N,T,18) float32
    [x,y,xV,yV,xA,yA,
     latLaneCenterOffset,
     laneChange,
     norm_off,
     leadDHW, leadDV, leadTHW, leadTTC, lead_exists,
     ramp_onehot4(none/onramp/offramp/unknown) ]  # for highD: always "none" -> [1,0,0,0]
- y_fut    : (N,Tf,2) float32   [x,y]
- nb_hist  : (N,T,8,6) float32  ego-relative: (nb - ego) for [x,y,xV,yV,xA,yA]
- nb_mask  : (N,T,8) bool       True if neighbor exists
- ego_static : (N,2+C) float32  [width,length] + class onehot (C=8, CLASS_VOCAB same as exiD)
- nb_static  : (N,T,8,2+C) float32  per-timestep neighbor static (zeros if missing)
- recordingId : (N,) int32      mapped by --recording_offset (default 100: 01->101 ... 60->160)
- trackId     : (N,) int32
- t0_frame    : (N,) int32      original frame index (highD frame count, before downsample)

Notes / assumptions (highD v1):
- recordingMeta contains: frameRate, upperLaneMarkings, lowerLaneMarkings
- tracksMeta contains: id, width, height, class, drivingDirection
- tracks contains: frame, id, x, y, xVelocity, yVelocity, xAcceleration, yAcceleration,
                   laneId, precedingId, dhw, thw, ttc, and 8 neighbor-id columns.
- If some columns are missing, we fill with zeros and continue.

Coordinate convention:
- This script can optionally "normalize_upper_xy" (flip XY for drivingDirection==1) in the same
  manner as your existing raw_to_npz.py pipeline.
- Lane center/width computations are applied *after* the optional y-flip, using flipped markings.

Run example:
  python3 highD_raw_to_npz.py \
    --raw_dir /path/to/highD/data \
    --out_dir /path/to/out_npz \
    --target_hz 5 --history_sec 2 --future_sec 5 \
    --stride_sec 0.2 \
    --normalize_upper_xy \
    --recording_offset 100
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

NEIGHBOR_COLS_8 = [
    "precedingId",
    "followingId",
    "leftPrecedingId",
    "leftAlongsideId",
    "leftFollowingId",
    "rightPrecedingId",
    "rightAlongsideId",
    "rightFollowingId",
]

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


def compute_downsample_step(source_hz: float, target_hz: float) -> int:
    step = int(round(float(source_hz) / float(target_hz)))
    if step <= 0:
        raise ValueError(f"Invalid downsample step: source_hz={source_hz}, target_hz={target_hz}")
    return step


def parse_semicolon_floats(s: str) -> List[float]:
    if not isinstance(s, str):
        return []
    s = s.strip()
    if not s:
        return []
    out: List[float] = []
    for p in s.split(";"):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except ValueError:
            pass
    return out


def find_recording_ids(raw_dir: Path) -> List[str]:
    ids = []
    for p in raw_dir.glob("*_tracks.csv"):
        m = re.match(r"(\d+)_tracks\.csv$", p.name)
        if m:
            ids.append(m.group(1))
    return sorted(set(ids))


def map_highd_class_to_vocab(name) -> str:
    if isinstance(name, str):
        s = name.strip().lower()
        if s in {"car", "truck"}:
            return s
    return "other"


def class_onehot(name: str) -> np.ndarray:
    out = np.zeros((len(CLASS_VOCAB),), dtype=np.float32)
    name = map_highd_class_to_vocab(name)
    idx = CLASS_VOCAB.index(name) if name in CLASS_VOCAB else CLASS_VOCAB.index("other")
    out[idx] = 1.0
    return out


def _safe_float(x: np.ndarray, default: float = 0.0) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    bad = ~np.isfinite(x)
    if np.any(bad):
        x = x.copy()
        x[bad] = default
    return x


@dataclass
class Config:
    raw_dir: Path
    out_dir: Path
    target_hz: float
    history_sec: float
    future_sec: float
    stride_sec: float
    normalize_upper_xy: bool
    recording_offset: int
    min_speed_mps: float


def load_recording(raw_dir: Path, rec_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rec_meta_path = raw_dir / f"{rec_id}_recordingMeta.csv"
    tracks_meta_path = raw_dir / f"{rec_id}_tracksMeta.csv"
    tracks_path = raw_dir / f"{rec_id}_tracks.csv"
    recording_meta = pd.read_csv(rec_meta_path)
    tracks_meta = pd.read_csv(tracks_meta_path)
    tracks = pd.read_csv(tracks_path)
    return recording_meta, tracks_meta, tracks


def flip_constants(recording_meta: pd.DataFrame) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Return (C_y, frame_rate, upper_markings, lower_markings)"""
    if "frameRate" not in recording_meta.columns:
        raise ValueError("recordingMeta missing frameRate")
    frame_rate = float(recording_meta.loc[0, "frameRate"])

    upper = parse_semicolon_floats(str(recording_meta.loc[0, "upperLaneMarkings"])) if "upperLaneMarkings" in recording_meta.columns else []
    lower = parse_semicolon_floats(str(recording_meta.loc[0, "lowerLaneMarkings"])) if "lowerLaneMarkings" in recording_meta.columns else []

    upper_arr = np.array(upper, dtype=np.float32)
    lower_arr = np.array(lower, dtype=np.float32)

    # same constant used by your existing raw_to_npz.py: C_y = upper_last + lower_first
    if len(upper_arr) == 0 or len(lower_arr) == 0:
        C_y = 0.0
    else:
        C_y = float(upper_arr[-1] + lower_arr[0])

    return C_y, frame_rate, upper_arr, lower_arr


def maybe_flip_rows(
    x: np.ndarray, y: np.ndarray,
    xv: np.ndarray, yv: np.ndarray,
    xa: np.ndarray, ya: np.ndarray,
    lane_id: np.ndarray,
    driving_dir: np.ndarray,
    C_y: float,
    x_max: float,
    upper_lane_minmax: Optional[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Flip for drivingDirection==1:
      x' = x_max - x
      y' = C_y - y
      velocities/accelerations sign flipped for x/y
      laneId mirrored within upper-lanes: new_laneId = (min_lane_upper + max_lane_upper) - old_laneId
    """
    mask = (driving_dir == 1)
    if not np.any(mask):
        return x, y, xv, yv, xa, ya, lane_id

    x2 = x.copy()
    y2 = y.copy()
    xv2 = xv.copy()
    yv2 = yv.copy()
    xa2 = xa.copy()
    ya2 = ya.copy()
    lane2 = lane_id.copy()

    x2[mask] = x_max - x2[mask]
    y2[mask] = C_y - y2[mask]
    xv2[mask] = -xv2[mask]
    yv2[mask] = -yv2[mask]
    xa2[mask] = -xa2[mask]
    ya2[mask] = -ya2[mask]

    if upper_lane_minmax is not None:
        mn, mx = upper_lane_minmax
        # only mirror valid positive lane ids
        ok = mask & (lane2 > 0)
        lane2[ok] = (mn + mx) - lane2[ok]

    return x2, y2, xv2, yv2, xa2, ya2, lane2


def build_lane_tables_from_markings(markings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    markings: array of length (num_markings). For N lanes, num_markings should be N+1.
    Returns:
      centerlines: (N_lanes,) float32
      widths:      (N_lanes,) float32
    If markings are insufficient, returns empty arrays.
    """
    if markings is None or len(markings) < 2:
        return np.zeros((0,), np.float32), np.zeros((0,), np.float32)
    # lanes are between consecutive markings
    left = markings[:-1]
    right = markings[1:]
    widths = (right - left).astype(np.float32)
    center = ((right + left) * 0.5).astype(np.float32)
    return center, widths


def main_one_recording(cfg: Config, rec_id: str) -> None:
    recording_meta, tracks_meta, tracks = load_recording(cfg.raw_dir, rec_id)

    C_y, frame_rate, upper_markings, lower_markings = flip_constants(recording_meta)
    step = compute_downsample_step(frame_rate, cfg.target_hz)
    T = int(round(cfg.history_sec * cfg.target_hz))
    Tf = int(round(cfg.future_sec * cfg.target_hz))
    stride = max(1, int(round(cfg.stride_sec * cfg.target_hz)))

    # Required-ish columns, but be lenient.
    for c in ["frame", "id", "x", "y"]:
        if c not in tracks.columns:
            raise ValueError(f"{rec_id}: tracks missing required column {c}")

    # Ensure neighbor id cols exist (fill 0 if missing)
    for c in NEIGHBOR_COLS_8:
        if c not in tracks.columns:
            tracks[c] = 0

    # Ensure kinematics exist (fill 0 if missing)
    for c in ["xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]:
        if c not in tracks.columns:
            tracks[c] = 0.0

    # laneId exists? if not, fill 0
    if "laneId" not in tracks.columns:
        tracks["laneId"] = 0

    # safety cols
    for c in ["dhw", "thw", "ttc"]:
        if c not in tracks.columns:
            tracks[c] = 0.0

    # driving direction per vehicle id
    if "id" not in tracks_meta.columns or "drivingDirection" not in tracks_meta.columns:
        raise ValueError(f"{rec_id}: tracksMeta missing id/drivingDirection")

    vid_to_dd: Dict[int, int] = dict(zip(tracks_meta["id"].astype(int), tracks_meta["drivingDirection"].astype(int)))

    # vehicle dimensions / class
    for c in ["width", "height", "class"]:
        if c not in tracks_meta.columns:
            raise ValueError(f"{rec_id}: tracksMeta missing {c}")

    vid_to_w: Dict[int, float] = dict(zip(tracks_meta["id"].astype(int), tracks_meta["width"].astype(float)))
    vid_to_l: Dict[int, float] = dict(zip(tracks_meta["id"].astype(int), tracks_meta["height"].astype(float)))  # treat height as length
    vid_to_cls: Dict[int, str] = dict(zip(tracks_meta["id"].astype(int), tracks_meta["class"].astype(str)))

    # Build lane tables (center/width) for each direction.
    # If normalize_upper_xy is enabled, we must flip markings for upper direction too.
    # y' = C_y - y, so marking m becomes (C_y - m). This reverses order; we sort.
    upper_for_calc = upper_markings.copy()
    lower_for_calc = lower_markings.copy()
    if cfg.normalize_upper_xy and len(upper_for_calc) > 0:
        upper_for_calc = np.sort((C_y - upper_for_calc).astype(np.float32))
    # lower direction is NOT flipped in this convention, so keep as-is.


    upper_center, upper_width = build_lane_tables_from_markings(upper_for_calc)
    lower_center, lower_width = build_lane_tables_from_markings(lower_for_calc)

    # Precompute upper lane min/max for lane mirroring
    upper_lane_minmax = None
    if len(upper_center) > 0:
        upper_lane_minmax = (1, int(len(upper_center)))

    # Convert tracks to numpy arrays (fast access)
    frame = tracks["frame"].astype(np.int32).to_numpy()
    vid = tracks["id"].astype(np.int32).to_numpy()

    x = tracks["x"].astype(np.float32).to_numpy()
    y = tracks["y"].astype(np.float32).to_numpy()

    # highD tracks.csv stores the TOP-LEFT corner of each bounding box in (x, y).
    # Convert to center coordinates to match exiD (xCenter, yCenter).
    w_row = np.array([vid_to_w.get(int(v), 0.0) for v in vid], dtype=np.float32)
    h_row = np.array([vid_to_l.get(int(v), 0.0) for v in vid], dtype=np.float32)  # tracksMeta 'height' treated as length
    x = x + 0.5 * w_row
    y = y + 0.5 * h_row
    xv = tracks["xVelocity"].astype(np.float32).to_numpy()
    yv = tracks["yVelocity"].astype(np.float32).to_numpy()
    xa = tracks["xAcceleration"].astype(np.float32).to_numpy()
    ya = tracks["yAcceleration"].astype(np.float32).to_numpy()
    lane_id = tracks["laneId"].astype(np.int16).to_numpy()

    dd = np.array([vid_to_dd.get(int(v), 0) for v in vid], dtype=np.int8)

    # Compute x_max for flipping (use max x in recording)
    x_max = float(np.nanmax(x)) if len(x) else 0.0

    if cfg.normalize_upper_xy:
        x, y, xv, yv, xa, ya, lane_id = maybe_flip_rows(
            x, y, xv, yv, xa, ya, lane_id, dd, C_y, x_max, upper_lane_minmax
        )


    # --- Coordinate min-shift (recording-level) to match exiD ---
    # Apply AFTER bbox top-left->center conversion and AFTER optional direction flipping.
    if x.size == 0:
        x_min, y_min = 0.0, 0.0
    else:
        x_min = float(np.nanmin(x))
        y_min = float(np.nanmin(y))

    x = (x - x_min).astype(np.float32, copy=False)
    y = (y - y_min).astype(np.float32, copy=False)

    # Shift lane centerlines too so latLaneCenterOffset remains consistent under the same shift.
    if len(upper_center) > 0:
        upper_center = (upper_center - y_min).astype(np.float32, copy=False)
    if len(lower_center) > 0:
        lower_center = (lower_center - y_min).astype(np.float32, copy=False)

    # Build per-vehicle: frames -> row-index map (for O(1) lookup at same frame)
    # We'll also store per-vehicle sorted row indices for window sampling.
    per_vid_rows: Dict[int, np.ndarray] = {}
    per_vid_frame_to_row: Dict[int, Dict[int, int]] = {}

    # group by id
    # (pandas groupby is okay here; number of vehicles per rec is not crazy)
    tracks_idx = np.arange(len(tracks), dtype=np.int32)
    for v, idxs in tracks.groupby("id").indices.items():
        idxs = np.array(idxs, dtype=np.int32)
        # sort by frame
        order = np.argsort(frame[idxs])
        idxs = idxs[order]
        per_vid_rows[int(v)] = idxs
        per_vid_frame_to_row[int(v)] = {int(fr): int(r) for fr, r in zip(frame[idxs], idxs)}

    # laneChange per row (1 at first frame where laneId changes for that vehicle)
    lane_change = np.zeros((len(tracks),), dtype=np.float32)
    for v, idxs in per_vid_rows.items():
        if len(idxs) < 2:
            continue
        l = lane_id[idxs].astype(np.int32)
        # change at position i where l[i] != l[i-1]
        ch = (l[1:] != l[:-1])
        if np.any(ch):
            # mark the first frame where changed
            lane_change[idxs[1:][ch]] = 1.0

    # Neighbor ids arrays
    nb_ids_all = np.stack([tracks[c].astype(np.int32).to_numpy() for c in NEIGHBOR_COLS_8], axis=1)  # (M,8)

    dhw = tracks["dhw"].astype(np.float32).to_numpy()
    thw = tracks["thw"].astype(np.float32).to_numpy()
    ttc = tracks["ttc"].astype(np.float32).to_numpy()

    # ---------------------------
    # Sanitize safety metrics (highD)
    # - sanitize arrays first (invalid -> -1, cap large values),
    #   then later force them to 0 where lead_exists==0 so that -1 means
    #   "invalid measurement given a lead exists".
    # ---------------------------
    # TTC: <=1 or negative or non-finite -> -1, >=90 -> 90
    ttc_bad = (~np.isfinite(ttc)) | (ttc <= 1.0) | (ttc < 0.0)
    ttc = np.where(ttc_bad, -1.0, ttc).astype(np.float32)
    ttc = np.clip(ttc, -1.0, 90.0).astype(np.float32)

    # THW: <=0.5 or non-finite -> -1, >=20 -> 20
    thw_bad = (~np.isfinite(thw)) | (thw <= 0.5)
    thw = np.where(thw_bad, -1.0, thw).astype(np.float32)
    thw = np.clip(thw, -1.0, 20.0).astype(np.float32)

    # DHW: <=10 or non-finite -> -1, >=150 -> 150
    dhw_bad = (~np.isfinite(dhw)) | (dhw <= 10.0)
    dhw = np.where(dhw_bad, -1.0, dhw).astype(np.float32)
    dhw = np.clip(dhw, -1.0, 150.0).astype(np.float32)

    # We will generate windows: for each vehicle, choose t0 frames with stride on downsampled grid.
    x_hist_list = []
    y_fut_list = []
    y_fut_vel_list = [] 
    y_fut_acc_list = []
    nb_hist_list = []
    nb_mask_list = []
    ego_static_list = []
    nb_static_list = []
    recid_list = []
    trackid_list = []
    t0_list = []

    # Ramp type: always none => onehot [1,0,0,0]
    ramp_oh = np.array([1, 0, 0, 0], dtype=np.float32)

    C = len(CLASS_VOCAB)
    K = 8

    for v, idxs in per_vid_rows.items():
        frs = frame[idxs]
        if len(frs) < (T + Tf) * step:
            continue

        # candidate t0 indices in this vehicle track index list (idxs positions)
        # We'll pick based on frame values aligned to step: use actual frames, then sample by step.
        # Simplest: iterate over positions in idxs with increment = stride*step, but ensure contiguous frames exist.
        # We'll choose t0 positions on downsampled frames: require all frames at (t0 + k*step) exist.
        # Precompute a set for fast presence
        fr_set = set(map(int, frs.tolist()))
        # pick t0 frames starting from first possible
        start_min = int(frs[0] + (T - 1) * step)
        end_max = int(frs[-1] - Tf * step)
        if start_min > end_max:
            continue

        # We pick t0 frames spaced by stride*step.
        # Align to step grid roughly by snapping to closest existing frame >= start_min that is on modulo step from frs[0]
        t0_frame = start_min
        while t0_frame <= end_max:
            # Collect history frames ending at t0_frame inclusive: t0 - (T-1)*step ... t0
            hist_frames = [t0_frame - (T - 1 - i) * step for i in range(T)]
            fut_frames = [t0_frame + (i + 1) * step for i in range(Tf)]

            if not all((hf in fr_set) for hf in hist_frames):
                t0_frame += stride * step
                continue
            if not all((ff in fr_set) for ff in fut_frames):
                t0_frame += stride * step
                continue

            # Build ego hist/fut arrays
            ego_rows = [per_vid_frame_to_row[v][hf] for hf in hist_frames]
            fut_rows = [per_vid_frame_to_row[v][ff] for ff in fut_frames]

            ego_x = x[ego_rows]
            ego_y = y[ego_rows]
            ego_xv = xv[ego_rows]
            ego_yv = yv[ego_rows]
            ego_xa = xa[ego_rows]
            ego_ya = ya[ego_rows]

            # lane features (per timestep)
            ego_lane = lane_id[ego_rows].astype(np.int32)
            ego_dd = np.array([vid_to_dd.get(v, 0)] * T, dtype=np.int32)

            # compute center/width per timestep based on direction + laneId
            lat_off = np.zeros((T,), dtype=np.float32)
            lane_w = np.zeros((T,), dtype=np.float32)
            for i in range(T):
                lid = int(ego_lane[i])
                if lid <= 0:
                    continue
                if int(ego_dd[i]) == 1 and len(upper_center) >= lid:
                    lat_off[i] = float(ego_y[i] - upper_center[lid - 1])
                    lane_w[i] = float(upper_width[lid - 1])
                elif int(ego_dd[i]) == 2 and len(lower_center) >= lid:
                    lat_off[i] = float(ego_y[i] - lower_center[lid - 1])
                    lane_w[i] = float(lower_width[lid - 1])
                else:
                    # unknown mapping -> keep zeros
                    pass

            half_w = lane_w * 0.5
            norm_off = np.zeros((T,), dtype=np.float32)
            ok = half_w > 1e-6
            norm_off[ok] = lat_off[ok] / half_w[ok]

            # lane change (history)
            ego_lc = lane_change[ego_rows].astype(np.float32)

            # lead features: use precedingId at each history frame
            lead_id = nb_ids_all[ego_rows, 0]  # precedingId slot
            lead_exists = (lead_id > 0).astype(np.float32)

            lead_dhw_raw = dhw[ego_rows].astype(np.float32)
            lead_thw_raw = thw[ego_rows].astype(np.float32)
            lead_ttc_raw = ttc[ego_rows].astype(np.float32)
            # If no lead at a timestep, force lead-related values to 0 (not -1).
            # This keeps -1 reserved for 'invalid measurement' only when lead_exists==1.
            lead_dhw = np.where(lead_exists > 0, lead_dhw_raw, 0.0).astype(np.float32)
            lead_thw = np.where(lead_exists > 0, lead_thw_raw, 0.0).astype(np.float32)
            lead_ttc = np.where(lead_exists > 0, lead_ttc_raw, 0.0).astype(np.float32)

            # leadDV = lead_xV - ego_xV if lead exists and lead row at same frame exists
            lead_dv = np.zeros((T,), dtype=np.float32)
            for i, (hf, lid) in enumerate(zip(hist_frames, lead_id.tolist())):
                if lid <= 0:
                    continue
                row_map = per_vid_frame_to_row.get(int(lid))
                if row_map is None:
                    continue
                r = row_map.get(int(hf))
                if r is None:
                    continue
                lead_dv[i] = float(xv[r] - ego_xv[i])

            # Ensure leadDV is 0 where lead does not exist (keeps semantics consistent).
            lead_dv = np.where(lead_exists > 0, lead_dv, 0.0).astype(np.float32)

            # Build ego_hist (T,18)
            ego_hist = np.stack([ego_x, ego_y, ego_xv, ego_yv, ego_xa, ego_ya], axis=1)  # (T,6)
            ego_hist = np.concatenate(
                [
                    ego_hist,
                    lat_off[:, None],
                    ego_lc[:, None],
                    norm_off[:, None],
                    lead_dhw[:, None],
                    lead_dv[:, None],
                    lead_thw[:, None],
                    lead_ttc[:, None],
                    lead_exists[:, None],
                    np.repeat(ramp_oh[None, :], T, axis=0),
                ],
                axis=1,
            ).astype(np.float32)
            assert ego_hist.shape[1] == 18

            # future (Tf,2)
            fut_xy = np.stack([x[fut_rows], y[fut_rows]], axis=1).astype(np.float32)
            fut_vel = np.stack([xv[fut_rows], yv[fut_rows]], axis=1).astype(np.float32)
            fut_acc = np.stack([xa[fut_rows], ya[fut_rows]], axis=1).astype(np.float32)

            # Neighbor history (T,8,6) ego-relative
            nb_hist = np.zeros((T, 8, 6), dtype=np.float32)
            nb_mask = np.zeros((T, 8), dtype=bool)

            # Neighbor static (T,8,2+C)
            nb_static = np.zeros((T, 8, 2 + C), dtype=np.float32)

            for ti, hf in enumerate(hist_frames):
                ego_vec = np.array([ego_x[ti], ego_y[ti], ego_xv[ti], ego_yv[ti], ego_xa[ti], ego_ya[ti]], dtype=np.float32)
                ids8 = nb_ids_all[ego_rows[ti]]  # (8,)
                for ki in range(8):
                    nid = int(ids8[ki])
                    if nid <= 0:
                        continue
                    row_map = per_vid_frame_to_row.get(nid)
                    if row_map is None:
                        continue
                    r = row_map.get(int(hf))
                    if r is None:
                        continue

                    nb_vec = np.array([x[r], y[r], xv[r], yv[r], xa[r], ya[r]], dtype=np.float32)
                    nb_hist[ti, ki] = nb_vec - ego_vec
                    nb_mask[ti, ki] = True

                    # static
                    w = float(vid_to_w.get(nid, 0.0))
                    l = float(vid_to_l.get(nid, 0.0))
                    oh = class_onehot(vid_to_cls.get(nid, "other"))
                    nb_static[ti, ki, 0] = w
                    nb_static[ti, ki, 1] = l
                    nb_static[ti, ki, 2:] = oh

            # ego static (2+C)
            ego_w = float(vid_to_w.get(v, 0.0))
            ego_l = float(vid_to_l.get(v, 0.0))
            ego_oh = class_onehot(vid_to_cls.get(v, "other"))
            ego_static = np.zeros((2 + C,), dtype=np.float32)
            ego_static[0] = ego_w
            ego_static[1] = ego_l
            ego_static[2:] = ego_oh

            # filter: minimum speed (use mean speed over history)
            sp = np.sqrt(ego_xv**2 + ego_yv**2)
            if float(np.nanmean(sp)) < cfg.min_speed_mps:
                t0_frame += stride * step
                continue

            # push
            x_hist_list.append(ego_hist)
            y_fut_list.append(fut_xy)
            y_fut_vel_list.append(fut_vel)
            y_fut_acc_list.append(fut_acc) 
            nb_hist_list.append(nb_hist)
            nb_mask_list.append(nb_mask)
            ego_static_list.append(ego_static)
            nb_static_list.append(nb_static)

            recid_list.append(int(rec_id) + int(cfg.recording_offset))
            trackid_list.append(int(v))
            t0_list.append(int(t0_frame))

            t0_frame += stride * step

    if len(x_hist_list) == 0:
        print(f"[WARN] {rec_id}: no samples produced.")
        return

    x_hist_arr = _safe_float(np.stack(x_hist_list, axis=0))
    y_fut_arr = _safe_float(np.stack(y_fut_list, axis=0))
    y_fut_vel_arr = _safe_float(np.stack(y_fut_vel_list, axis=0))  # (N,Tf,2)
    y_fut_acc_arr = _safe_float(np.stack(y_fut_acc_list, axis=0))  # (N,Tf,2)
    nb_hist_arr = _safe_float(np.stack(nb_hist_list, axis=0))
    nb_mask_arr = np.stack(nb_mask_list, axis=0)
    ego_static_arr = _safe_float(np.stack(ego_static_list, axis=0))
    nb_static_arr = _safe_float(np.stack(nb_static_list, axis=0))

    recordingId_arr = np.array(recid_list, dtype=np.int32)
    trackId_arr = np.array(trackid_list, dtype=np.int32)
    t0_frame_arr = np.array(t0_list, dtype=np.int32)

    out_path = cfg.out_dir / f"highd_{int(rec_id):02d}.npz"
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        x_hist=x_hist_arr,
        y_fut=y_fut_arr,
        y_fut_vel=y_fut_vel_arr,
        y_fut_acc=y_fut_acc_arr,
        nb_hist=nb_hist_arr,
        nb_mask=nb_mask_arr,
        ego_static=ego_static_arr,
        nb_static=nb_static_arr,
        recordingId=recordingId_arr,
        trackId=trackId_arr,
        t0_frame=t0_frame_arr,
        # --- Meta keys (match exiD) ---
        class_names=np.array(CLASS_VOCAB, dtype=object),
        T=np.array([T], dtype=np.int32),
        Tf=np.array([Tf], dtype=np.int32),
        K=np.array([K], dtype=np.int32),
        ego_dim=np.array([18], dtype=np.int32),
        nb_dim=np.array([6], dtype=np.int32),
        static_dim=np.array([2 + len(CLASS_VOCAB)], dtype=np.int32),
        origin_min_xy=np.array([x_min, y_min], dtype=np.float32),
    )
    print(f"[OK] {rec_id} -> {out_path.name}  samples={len(x_hist_arr)}")


def parse_args() -> Config:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="raw/", help="Directory containing highD *_tracks.csv, *_tracksMeta.csv, *_recordingMeta.csv")
    ap.add_argument("--out_dir", type=str, default="data_npz", help="Output directory for npz files")
    ap.add_argument("--target_hz", type=float, default=5.0)
    ap.add_argument("--history_sec", type=float, default=2.0)
    ap.add_argument("--future_sec", type=float, default=5.0)
    ap.add_argument("--stride_sec", type=float, default=1.0, help="Sampling stride in seconds between consecutive windows (on target_hz grid)")
    ap.add_argument("--normalize_upper_xy", action="store_true", help="Flip (x,y,vel,acc,laneId) for drivingDirection==1 to unify directions")
    ap.add_argument("--recording_offset", type=int, default=100, help="Map highD 01..60 -> 101..160 by default")
    ap.add_argument("--min_speed_mps", type=float, default=0.0, help="Drop samples whose mean speed over history is below this")
    args = ap.parse_args()

    # ---- auto out_dir naming ----
    T_sec = int(round(args.history_sec))
    Tf_sec = int(round(args.future_sec))
    hz = int(round(args.target_hz))

    out_dir = (
        Path(args.out_dir)
        / f"highd_T{T_sec}_Tf{Tf_sec}_hz{hz}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(out_dir),
        target_hz=args.target_hz,
        history_sec=args.history_sec,
        future_sec=args.future_sec,
        stride_sec=args.stride_sec,
        normalize_upper_xy=args.normalize_upper_xy,
        recording_offset=args.recording_offset,
        min_speed_mps=args.min_speed_mps,
    )


def main():
    cfg = parse_args()
    rec_ids = find_recording_ids(cfg.raw_dir)
    if not rec_ids:
        raise SystemExit(f"No *_tracks.csv found under {cfg.raw_dir}")

    for rec_id in rec_ids:
        main_one_recording(cfg, rec_id)


if __name__ == "__main__":
    main()
