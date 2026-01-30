#!/usr/bin/env python3
"""
exiD tracks.csv -> NPZ preprocessing with Adjacency Filtering & Enhanced Features.

Key Features maintained from uploaded file:
1. Filters neighbors to include ONLY those in physically adjacent lanelets 
   (using lanelet_adj_allmaps.pkl).
2. Robust CSV parsing (handling ';' in neighbor columns).
3. Optimized map building.

New Features implemented:
1. ego_hist (T, 13): Excludes lead safety features.
2. ego_safety (T, 5): [leadDHW, leadDV, leadTHW, leadTTC, lead_exists].
3. nb_hist (T, K, 9): 
   - 0-5: dx, dy, dvx, dvy, dax, day
   - 6: lc_state (Slot-based logic: -3 to +3)
   - 7: dx_time (dx / |v_ego|)
   - 8: gate (1 if within time bounds)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import pickle

# =========================
# Constants & Config
# =========================
LANEWIDTH_DROP_TH = 3.0   # laneWidth < 3.0 -> drop window (history)
LEAD_TTC_CAP = 90.0       # seconds
LEAD_THW_CAP = 20.0       # seconds

# Index 0, 1 are Same Lane (Always Keep)
# Index 2~7 are Side Lanes (Check Adjacency)
NEIGHBOR_COLS_8 = [
    "leadId",           # 0
    "rearId",           # 1
    "leftLeadId",       # 2
    "leftAlongsideId",  # 3
    "leftRearId",       # 4
    "rightLeadId",      # 5
    "rightAlongsideId", # 6
    "rightRearId",      # 7
]

CLASS_VOCAB = [
    "car", "truck", "van", "bus", "motorcycle", "bicycle", "pedestrian", "other",
]
VRU_CLASSES = {"motorcycle", "bicycle", "pedestrian"}


def compute_downsample_step(source_hz: float, target_hz: float) -> int:
    step = int(round(float(source_hz) / float(target_hz)))
    if step <= 0:
        step = 1
    return step


def _safe_float(a: np.ndarray, default: float = 0.0) -> np.ndarray:
    """Replace NaN/inf with default."""
    a = a.astype(np.float32, copy=False)
    bad = ~np.isfinite(a)
    if np.any(bad):
        a = a.copy()
        a[bad] = default
    return a


def _onehot4(code: np.ndarray) -> np.ndarray:
    """code in {0,1,2,3} -> onehot (N,4) float32."""
    oh = np.zeros((len(code), 4), dtype=np.float32)
    idx = np.clip(code.astype(np.int64), 0, 3)
    oh[np.arange(len(code)), idx] = 1.0
    return oh


def _normalize_class_name(s: str) -> str:
    s = (s or "").strip().lower()
    if s == "" or s in {"nan", "null"}:
        return "other"
    # Map common variations if necessary, or check vocab
    if s not in CLASS_VOCAB:
        # Simple heuristics for exiD/highD
        if "truck" in s: return "truck"
        return "other"
    return s


def _build_class_map(meta_csv: Path) -> Tuple[Dict[int, str], Dict[str, int], np.ndarray]:
    dfm = pd.read_csv(meta_csv, low_memory=False)
    if "trackId" not in dfm.columns or "class" not in dfm.columns:
        # Fallback if specific columns missing
        return {}, {c: i for i, c in enumerate(CLASS_VOCAB)}, np.array(CLASS_VOCAB)

    class_map: Dict[int, str] = {}
    for tid, cls in zip(dfm["trackId"].astype(int).tolist(), dfm["class"].astype(str).tolist()):
        class_map[int(tid)] = _normalize_class_name(cls)

    class_names = np.array(CLASS_VOCAB, dtype=object)
    class_to_idx = {c: i for i, c in enumerate(CLASS_VOCAB)}
    return class_map, class_to_idx, class_names


def _onehot_class(class_name: str, class_to_idx: Dict[str, int]) -> np.ndarray:
    C = len(class_to_idx)
    oh = np.zeros((C,), dtype=np.float32)
    idx = class_to_idx.get(class_name, class_to_idx.get("other", 0))
    oh[idx] = 1.0
    return oh


def load_lanelet_adj(path: Path) -> Dict[int, Dict[int, Set[int]]]:
    """
    Load adjacency pickle.
    Returns: map_id -> { lanelet_id -> set(adjacent_lanelet_ids) }
    """
    if not path.exists():
        print(f"[WARN] Adjacency pickle not found: {path}. Filtering will be skipped.")
        return {}
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj.get("adj_by_map", obj)
    except Exception as e:
        print(f"[WARN] Failed to load adjacency pickle: {e}")
        return {}


def build_vehicle_maps(
    track_ids: np.ndarray, frames: np.ndarray
) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[int, Dict[int, int]]]:
    """
    Build efficient lookups using numpy.
    """
    uniq_ids, start_idx, counts = np.unique(track_ids, return_index=True, return_counts=True)
    veh_rows: Dict[int, np.ndarray] = {}
    frame_to_pos: Dict[int, Dict[int, int]] = {}

    for vid, st, ct in zip(uniq_ids.tolist(), start_idx.tolist(), counts.tolist()):
        idxs = np.arange(st, st + ct, dtype=np.int32)
        veh_rows[int(vid)] = idxs
        f = frames[idxs]
        # Dictionary comprehension is still Python-loop based but unavoidable for O(1) frame lookup per ID
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
    adjacent_only: bool = False,
    adj_db: Optional[Dict] = None,
    # New params for gating/LC
    t_front: float = 3.0,
    t_back: float = 1.5,
    vy_eps: float = 0.2,
    eps_gate: float = 0.1,
) -> Tuple[int, Optional[str]]:

    # 1. Read Tracks
    try:
        df = pd.read_csv(tracks_csv, low_memory=False)
    except Exception as e:
        return 0, f"Failed to read {tracks_csv}: {e}"

    # 2. Read Meta
    meta_csv = tracks_csv.with_name(tracks_csv.name.replace("_tracks.csv", "_tracksMeta.csv"))
    rec_meta_csv = tracks_csv.with_name(tracks_csv.name.replace("_tracks.csv", "_recordingMeta.csv"))
    
    # Get Map ID
    map_id = -1
    if rec_meta_csv.exists():
        try:
            rdf = pd.read_csv(rec_meta_csv, low_memory=False)
            if "locationId" in rdf.columns:
                map_id = int(rdf["locationId"].iloc[0])
        except Exception:
            pass
            
    try:
        class_map, class_to_idx, class_names = _build_class_map(meta_csv)
    except Exception:
        # Fallback if meta missing
        class_map, class_to_idx, class_names = {}, {c: i for i, c in enumerate(CLASS_VOCAB)}, np.array(CLASS_VOCAB)

    # 3. Check Columns & Robust Parsing
    # Ensure standard columns exist
    required_base = [
        "trackId", "frame", "xCenter", "yCenter",
        "lonVelocity", "latVelocity", "lonAcceleration", "latAcceleration",
        "latLaneCenterOffset", "laneWidth", "laneChange",
        "leadDHW", "leadDV", "leadTHW", "leadTTC",
        "ramp_type", "width", "length", "laneletId"
    ]
    
    # Check what's missing
    missing = [c for c in required_base if c not in df.columns]
    if missing and adjacent_only and "laneletId" in missing:
         return 0, f"Missing columns for adjacency: {missing}"

    # 4. Sort & Downsample
    df = df.sort_values(["trackId", "frame"], kind="mergesort").reset_index(drop=True)
    ds_step = compute_downsample_step(source_hz, target_hz)
    df = df[((df["frame"].astype(int) - 1) % ds_step) == 0].copy()
    if df.empty:
        return 0, "Downsample produced no rows."

    rec_ids = df["recordingId"].dropna().unique()
    rec_id = int(rec_ids[0]) if len(rec_ids) == 1 else -1

    # 5. Extract Arrays
    track_ids = df["trackId"].astype(np.int32).to_numpy()
    frames = df["frame"].astype(np.int32).to_numpy()
    lanelet_ids = df["laneletId"].fillna(-1).astype(np.int32).to_numpy() if "laneletId" in df.columns else np.full(len(df), -1, dtype=np.int32)

    x = _safe_float(df["xCenter"].to_numpy())
    y = _safe_float(df["yCenter"].to_numpy())
    xv = _safe_float(df["lonVelocity"].to_numpy())
    yv = _safe_float(df["latVelocity"].to_numpy())
    xa = _safe_float(df["lonAcceleration"].to_numpy())
    ya = _safe_float(df["latAcceleration"].to_numpy())

    lat_center_off = _safe_float(df["latLaneCenterOffset"].to_numpy())
    lane_w = _safe_float(df["laneWidth"].to_numpy())
    lane_change = _safe_float(df["laneChange"].to_numpy())

    width_arr = _safe_float(df["width"].to_numpy(), 0.0)
    length_arr = _safe_float(df["length"].to_numpy(), 0.0)

    # Lead Safety Columns (from CSV)
    lead_dhw = _safe_float(df["leadDHW"].to_numpy(), 0.0)
    lead_dv = _safe_float(df["leadDV"].to_numpy(), 0.0)
    lead_thw = _safe_float(df["leadTHW"].to_numpy(), 0.0)
    lead_ttc = _safe_float(df["leadTTC"].to_numpy(), 0.0)

    # Robust Neighbor Parsing (Handle "id;dist" format in raw exiD)
    nb_ids_cols = []
    for col in NEIGHBOR_COLS_8:
        if col in df.columns:
            s = df[col].astype(str).str.strip()
            s = s.str.split(';').str[0] # Take ID part only
            s = pd.to_numeric(s, errors='coerce').fillna(-1).astype(np.int32)
            nb_ids_cols.append(s.to_numpy())
        else:
            nb_ids_cols.append(np.full(len(df), -1, dtype=np.int32))
    
    nb_ids_all = np.stack(nb_ids_cols, axis=1) # (N, 8)

    # Ramp Onehot
    RAMP_MAP = {"none": 0, "onramp": 1, "offramp": 2, "not_found": 3}
    if "ramp_type" in df.columns:
        ramp_code = df["ramp_type"].map(RAMP_MAP).fillna(3).astype(np.int8).to_numpy()
    else:
        ramp_code = np.zeros(len(df), dtype=np.int8)
    ramp_oh = _onehot4(ramp_code)

    # Norm Offset
    eps = 1e-6
    half_w = np.maximum(lane_w * 0.5, eps).astype(np.float32)
    norm_off = (lat_center_off / half_w).astype(np.float32)

    speed = np.sqrt(xv * xv + yv * yv)

    # 6. Build Maps (Optimized)
    uniq_ids, veh_rows, frame_to_pos = build_vehicle_maps(track_ids, frames)

    # Static Lookup
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

    # Adjacency Set
    current_adj_set: Dict[int, Set[int]] = {}
    if adjacent_only and adj_db and map_id in adj_db:
        current_adj_set = adj_db[map_id]

    # Time Parameters
    T = int(round(history_sec * target_hz))
    Tf = int(round(future_sec * target_hz))
    stride = max(1, int(round(stride_sec * target_hz)))
    win_len = T + Tf
    expected_gap = ds_step

    # Shift to origin
    min_x = float(np.nanmin(x)) if x.size > 0 else 0.0
    min_y = float(np.nanmin(y)) if y.size > 0 else 0.0
    if not np.isfinite(min_x): min_x = 0.0
    if not np.isfinite(min_y): min_y = 0.0

    x_shift = (x - min_x).astype(np.float32)
    y_shift = (y - min_y).astype(np.float32)

    # ---- Collect Samples ----
    X_list, Y_list = [], []
    YV_list, YA_list = [], []
    NB_list, NBmask_list = [], []
    EgoStatic_list, EgoSafety_list, NBStatic_list = [], [], []
    trackId_list, t0_list = [], []

    K = 8
    ego_dim = 13  # Reduced dim (removed lead safety)
    nb_dim = 9    # Expanded dim (added LC state, gate)
    safety_dim = 5 # Separated lead safety
    static_dim = 2 + len(class_to_idx)

    for vid in uniq_ids.tolist():
        idxs = veh_rows[int(vid)]
        f = frames[idxs]
        n = len(idxs)
        if n < win_len:
            continue

        ego_is_vru = is_vru_trackid(int(vid))

        for i in range(0, n - win_len + 1, stride):
            # Check frame continuity
            f0 = int(f[i])
            f_last = int(f[i + win_len - 1])
            if f_last - f0 != expected_gap * (win_len - 1):
                continue

            hist_rows = idxs[i : i + T]
            fut_rows  = idxs[i + T : i + win_len]

            # Filter: Min Speed
            if min_speed_mps > 0.0 and float(np.nanmax(speed[hist_rows])) < float(min_speed_mps):
                continue

            # Filter: Lane Width
            if np.any(lane_w[hist_rows] < LANEWIDTH_DROP_TH):
                continue

            # VRU Logic
            any_nb_vru = False
            if (drop_vru or keep_only_vru_cases) and not ego_is_vru:
                 all_nbs = nb_ids_all[hist_rows].flatten()
                 for nid in all_nbs:
                     if nid >= 0 and is_vru_trackid(int(nid)):
                         any_nb_vru = True
                         break
            
            vru_involving = ego_is_vru or any_nb_vru
            if keep_only_vru_cases:
                if not vru_involving: continue
            elif drop_vru:
                if vru_involving: continue

            # --- Sample Construction ---
            
            # Lead Safety Construction
            lead_exists_mask = (nb_ids_all[hist_rows, 0] > -1)
            dhw_w = np.maximum(0.0, lead_dhw[hist_rows])
            dv_w  = lead_dv[hist_rows]
            thw_w = np.maximum(0.0, lead_thw[hist_rows])
            ttc_w = np.maximum(0.0, lead_ttc[hist_rows])

            # Apply Safety Caps
            thw_w = np.where(lead_exists_mask, np.minimum(thw_w, LEAD_THW_CAP), 0.0)
            ttc_w = np.where(lead_exists_mask, np.minimum(ttc_w, LEAD_TTC_CAP), 0.0)
            dhw_w = np.where(lead_exists_mask, dhw_w, 0.0)
            dv_w  = np.where(lead_exists_mask, dv_w, 0.0)
            
            # (T, 5) Safety Tensor
            ego_safety = np.stack([
                dhw_w, dv_w, thw_w, ttc_w, lead_exists_mask.astype(np.float32)
            ], axis=-1).astype(np.float32)

            # Ego History (T, 13) - Excluding safety
            ego_hist = np.stack([
                x_shift[hist_rows], y_shift[hist_rows],
                xv[hist_rows], yv[hist_rows],
                xa[hist_rows], ya[hist_rows],
                lat_center_off[hist_rows],
                lane_change[hist_rows],
                norm_off[hist_rows],
                ramp_oh[hist_rows, 0], ramp_oh[hist_rows, 1],
                ramp_oh[hist_rows, 2], ramp_oh[hist_rows, 3],
            ], axis=-1).astype(np.float32)

            # Future
            y_fut = np.stack([x_shift[fut_rows], y_shift[fut_rows]], axis=-1)
            y_fut_vel = np.stack([xv[fut_rows], yv[fut_rows]], axis=-1)
            y_fut_acc = np.stack([xa[fut_rows], ya[fut_rows]], axis=-1)

            # Ego Static
            ego_static = np.concatenate([
                [veh_width.get(int(vid), 0.0), veh_length.get(int(vid), 0.0)],
                _onehot_class(veh_class.get(int(vid), "other"), class_to_idx)
            ], axis=0).astype(np.float32)

            # --- Neighbor Processing (T, K, 9) ---
            nb_hist = np.zeros((T, K, nb_dim), dtype=np.float32)
            nb_mask = np.zeros((T, K), dtype=bool)
            nb_static = np.zeros((T, K, static_dim), dtype=np.float32)

            hist_frames = frames[hist_rows]

            for t in range(T):
                fr = int(hist_frames[t])
                ego_r = hist_rows[t]
                ego_lane = lanelet_ids[ego_r]
                
                row_nb_ids = nb_ids_all[ego_r]  # (K,)

                for k in range(K):
                    nid = int(row_nb_ids[k])
                    if nid < 0:
                        continue
                    
                    # 1. Look up neighbor index
                    d_pos = frame_to_pos.get(nid)
                    if d_pos is None: continue
                    pos_in = d_pos.get(fr)
                    if pos_in is None: continue
                    nb_row = veh_rows[nid][pos_in]

                    # 2. ADJACENCY FILTERING
                    if adjacent_only and k >= 2:
                        nb_lane = lanelet_ids[nb_row]
                        if ego_lane == -1 or nb_lane == -1: continue
                        if ego_lane not in current_adj_set: continue
                        if nb_lane not in current_adj_set[ego_lane]: continue

                    # 3. Features
                    nb_mask[t, k] = True

                    # Static
                    nb_static[t, k, 0] = veh_width.get(nid, 0.0)
                    nb_static[t, k, 1] = veh_length.get(nid, 0.0)
                    nb_static[t, k, 2:] = _onehot_class(veh_class.get(nid, "other"), class_to_idx)

                    # Dynamic Kinematics
                    dx = float(x_shift[nb_row] - x_shift[ego_r])
                    nb_hist[t, k, 0] = dx
                    nb_hist[t, k, 1] = float(y_shift[nb_row] - y_shift[ego_r])
                    nb_hist[t, k, 2] = float(xv[nb_row] - xv[ego_r])
                    nb_hist[t, k, 3] = float(yv[nb_row] - yv[ego_r])
                    nb_hist[t, k, 4] = float(xa[nb_row] - xa[ego_r])
                    nb_hist[t, k, 5] = float(ya[nb_row] - ya[ego_r])

                    # LC State Logic
                    lc_state = 0.0
                    vyn = float(yv[nb_row])
                    
                    if k >= 2: # Side neighbors
                        if k < 5: # Left Group (2,3,4)
                            if vyn > vy_eps: lc_state = -1.0    # Moving Right (Towards)
                            elif vyn < -vy_eps: lc_state = -3.0 # Moving Left (Away)
                            else: lc_state = -2.0
                        else:     # Right Group (5,6,7)
                            if vyn < -vy_eps: lc_state = 1.0    # Moving Left (Towards)
                            elif vyn > vy_eps: lc_state = 3.0   # Moving Right (Away)
                            else: lc_state = 2.0
                    
                    nb_hist[t, k, 6] = lc_state

                    # Gating Logic
                    v_ego = float(xv[ego_r])
                    dx_time = dx / (abs(v_ego) + eps_gate)
                    gate = 1.0 if (-t_back < dx_time < t_front) else 0.0

                    nb_hist[t, k, 7] = dx_time
                    nb_hist[t, k, 8] = gate

            X_list.append(ego_hist)
            Y_list.append(y_fut)
            YV_list.append(y_fut_vel)
            YA_list.append(y_fut_acc)
            NB_list.append(nb_hist)
            NBmask_list.append(nb_mask)
            EgoStatic_list.append(ego_static)
            EgoSafety_list.append(ego_safety)
            NBStatic_list.append(nb_static)
            trackId_list.append(int(vid))
            t0_list.append(int(frames[hist_rows[0]]))

    if len(X_list) == 0:
        return 0, "No valid samples produced."

    # Stack
    x_hist = np.stack(X_list, axis=0)
    y_fut = np.stack(Y_list, axis=0).astype(np.float32)
    y_fut_vel = np.stack(YV_list, axis=0).astype(np.float32)
    y_fut_acc = np.stack(YA_list, axis=0).astype(np.float32)
    nb_hist = np.stack(NB_list, axis=0)
    nb_mask = np.stack(NBmask_list, axis=0)
    ego_static = np.stack(EgoStatic_list, axis=0)
    ego_safety = np.stack(EgoSafety_list, axis=0)
    nb_static = np.stack(NBStatic_list, axis=0)
    
    trackId_arr = np.asarray(trackId_list, dtype=np.int32)
    t0_arr = np.asarray(t0_list, dtype=np.int32)
    recordingId_arr = np.full((len(trackId_list),), rec_id, dtype=np.int32)

    # Save
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
        ego_safety=ego_safety,
        nb_static=nb_static,
        trackId=trackId_arr,
        t0_frame=t0_arr,
        class_names=class_names,
        recordingId=recordingId_arr,
        T=np.array([T], dtype=np.int32),
        Tf=np.array([Tf], dtype=np.int32),
        K=np.array([K], dtype=np.int32),
        ego_dim=np.array([ego_dim], dtype=np.int32),
        nb_dim=np.array([nb_dim], dtype=np.int32),
        safety_dim=np.array([safety_dim], dtype=np.int32),
        static_dim=np.array([static_dim], dtype=np.int32),
        target_hz=np.array([target_hz], dtype=np.float32),
        source_hz=np.array([source_hz], dtype=np.float32),
        ds_step=np.array([ds_step], dtype=np.int32),
        history_sec=np.array([history_sec], dtype=np.float32),
        future_sec=np.array([future_sec], dtype=np.float32),
        stride_sec=np.array([stride_sec], dtype=np.float32),
        t_front=np.array([t_front], dtype=np.float32),
        t_back=np.array([t_back], dtype=np.float32),
        vy_eps=np.array([vy_eps], dtype=np.float32),
        eps_gate=np.array([eps_gate], dtype=np.float32),
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
    ap.add_argument("--tracks_dir", type=str, default="raw/", help="Directory containing *_tracks.csv")
    ap.add_argument("--glob", type=str, default="*_tracks.csv", help="Glob pattern")
    ap.add_argument("--out_root", type=str, default="data_npz", help="Output root directory")

    ap.add_argument("--source_hz", type=int, default=25)
    ap.add_argument("--target_hz", type=int, default=3)
    ap.add_argument("--history_sec", type=int, default=2)
    ap.add_argument("--future_sec", type=int, default=5)
    ap.add_argument("--stride_sec", type=int, default=1)
    ap.add_argument("--min_speed_mps", type=float, default=0.0)

    ap.add_argument("--drop_vru", action="store_true", help="Drop windows involving VRU.")
    ap.add_argument("--keep_only_vru_cases", action="store_true", help="Keep ONLY windows involving VRU.")
    
    ap.add_argument("--adjacent_only", action="store_true", help="Filter neighbors: include only those in physically adjacent lanelets.")
    ap.add_argument("--lanelet_adj_pkl", type=str, default="maps/lanelet_adj_allmaps.pkl", help="Path to adjacency pickle.")

    # New Params for Gating/LC
    ap.add_argument("--t_front", type=float, default=1.6, help="Time gate front")
    ap.add_argument("--t_back", type=float, default=1.4, help="Time gate back")
    ap.add_argument("--vy_eps", type=float, default=0.2, help="Lateral velocity epsilon for LC state")
    ap.add_argument("--eps_gate", type=float, default=0.1, help="Small epsilon for dx_time division")

    args = ap.parse_args()

    # Determine VRU policy
    drop_vru = True if (not args.keep_only_vru_cases) else False
    if args.drop_vru: drop_vru = True
    if args.keep_only_vru_cases: drop_vru = False

    # Load Adjacency DB once
    adj_db = {}
    if args.adjacent_only:
        adj_db = load_lanelet_adj(Path(args.lanelet_adj_pkl))
        if not adj_db:
            print("[WARN] Adjacency filter enabled but DB empty or not found. Neighbors will NOT be filtered correctly.")

    T_sec = int(round(args.history_sec))
    Tf_sec = int(round(args.future_sec))
    hz = int(round(args.target_hz))

    out_dir = Path(args.out_root) / f"exid_T{T_sec}_Tf{Tf_sec}_hz{hz}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks_dir = Path(args.tracks_dir)
    files = sorted(tracks_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"No files found in {tracks_dir} with pattern {args.glob}")

    total = 0
    ok_files = 0
    for tracks_csv in files:
        out_path = out_dir / f"exid_{tracks_csv.stem.replace('_tracks','')}.npz"
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
            adjacent_only=args.adjacent_only,
            adj_db=adj_db,
            t_front=args.t_front,
            t_back=args.t_back,
            vy_eps=args.vy_eps,
            eps_gate=args.eps_gate,
        )
        if err is not None:
            print(f"[SKIP] {tracks_csv.name}  reason={err}")
            continue

        print(f"[OK] {tracks_csv.name} -> {out_path.name}  samples={n}")
        total += n
        ok_files += 1

    print(f"[DONE] ok_files={ok_files}/{len(files)} total_samples={total}")

if __name__ == "__main__":
    main()