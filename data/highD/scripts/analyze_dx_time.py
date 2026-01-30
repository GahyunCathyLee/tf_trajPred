#!/usr/bin/env python3
"""
analyze_dx_time.py

Compute dx_time distributions from exiD *_tracks.csv and propose t_front/t_back
based on percentiles, optionally under "interaction" filters.

dx_time definition (matches your preprocessor):
  dx_time = dx / (abs(v_ego) + eps_gate)
  where dx = x_nb - x_ego (same frame), v_ego = xVelocity of ego at that frame.

Outputs:
- CSV summary (counts, percentiles) per slot group / per slot
- Optional histograms (png)

Usage examples:
  python analyze_dx_time.py --tracks_dir ./raw \
    --out_dir ./dx_time_stats --eps_gate 0.1 --min_speed 1.0 \
    --max_abs_dx 150 --p_front 0.95 --p_back 0.95 --make_plots

  # Focus on "interaction-like" cases:
  python analyze_dx_time.py --tracks_dir ./raw --out_dir ./dx_time_stats \
    --min_speed 1.0 --max_abs_dx 120 --interaction_mode lc \
    --vy_thresh 0.2 --make_plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Neighbor slot definitions
# (same as your preprocessor)
# -----------------------------
SLOT_NAMES = [
    "lead", "rear",
    "leftLead", "leftAlong", "leftRear",
    "rightLead", "rightAlong", "rightRear",
]
K = 8

# Column name candidates (exiD exports sometimes vary)
SIDE_COL_CANDIDATES = {
    "leftLead":   ["leftPrecedingId", "leftLeadId"],
    "leftAlong":  ["leftAlongsideId"],
    "leftRear":   ["leftFollowingId", "leftRearId"],
    "rightLead":  ["rightPrecedingId", "rightLeadId"],
    "rightAlong": ["rightAlongsideId"],
    "rightRear":  ["rightFollowingId", "rightRearId"],
}

REQUIRED_COLS = [
    "id", "frame",
    "x", "y",
    "xVelocity", "yVelocity",
]

# -----------------------------
# Helpers
# -----------------------------
def _safe_float(a: np.ndarray, default: float = 0.0) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    bad = ~np.isfinite(a)
    if np.any(bad):
        a = a.copy()
        a[bad] = default
    return a

def _get_col(df: pd.DataFrame, candidates: List[str], default: int = -1) -> np.ndarray:
    """
    Robustly read an ID column that may contain:
      - int/float
      - string like "287;285" (multiple ids)
      - empty / nan

    Policy: take the FIRST valid id.
    """
    for c in candidates:
        if c not in df.columns:
            continue

        s = df[c].fillna(default)

        # Convert to string to handle cases like "287;285"
        s_str = s.astype(str)

        # Take first token before ';' if exists
        first_tok = s_str.str.split(";", n=1, expand=True)[0].str.strip()

        # Convert to numeric safely
        out = pd.to_numeric(first_tok, errors="coerce").fillna(default).astype(np.int32).to_numpy()
        return out

    return np.full((len(df),), default, dtype=np.int32)

def _col_or_default(df: pd.DataFrame, name: str, default: int = -1) -> np.ndarray:
    if name in df.columns:
        return df[name].fillna(default).to_numpy(dtype=np.int32)
    return np.full((len(df),), default, dtype=np.int32)

def build_vehicle_row_index(track_ids: np.ndarray) -> Dict[int, np.ndarray]:
    veh_rows: Dict[int, List[int]] = {}
    for i, tid in enumerate(track_ids.tolist()):
        veh_rows.setdefault(int(tid), []).append(i)
    return {k: np.asarray(v, dtype=np.int32) for k, v in veh_rows.items()}

def build_frame_pos_maps(frames: np.ndarray, veh_rows: Dict[int, np.ndarray]) -> Dict[int, Dict[int, int]]:
    frame_to_pos: Dict[int, Dict[int, int]] = {}
    for tid, idxs in veh_rows.items():
        frs = frames[idxs]
        d: Dict[int, int] = {}
        for pos, fr in enumerate(frs.tolist()):
            d[int(fr)] = pos
        frame_to_pos[int(tid)] = d
    return frame_to_pos

def ensure_side_neighbor_cols(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    out = {}
    for key, cands in SIDE_COL_CANDIDATES.items():
        out[key] = _get_col(df, cands, default=-1)
    return out

@dataclass
class PercentileSpec:
    p05: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float

def calc_percentiles(x: np.ndarray) -> PercentileSpec:
    q = np.nanpercentile(x, [5,10,25,50,75,90,95,99]).tolist()
    return PercentileSpec(*map(float, q))

def save_hist_png(values: np.ndarray, out_png: Path, title: str, bins: int = 200,
                  xlim: Optional[Tuple[float,float]] = None) -> None:
    import matplotlib.pyplot as plt

    v = values[np.isfinite(values)]
    if len(v) == 0:
        return

    plt.figure()
    plt.hist(v, bins=bins)
    plt.title(title)
    plt.xlabel("dx_time (s)")
    plt.ylabel("count")
    if xlim is not None:
        plt.xlim(*xlim)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()

# -----------------------------
# Interaction filters
# -----------------------------
def interaction_mask(
    mode: str,
    slot_k: int,
    dx: float,
    dy: float,
    dvx: float,
    dvy: float,
    v_ego: float,
    vy_nb: float,
    lane_change_ego: float,
    vy_thresh: float,
    ttc_thresh: float,
) -> bool:
    """
    Decide whether this (ego, neighbor, frame, slot) is 'interaction-like'.

    mode:
      - "none": keep all
      - "lc":   lane-change involved (ego laneChange != 0 OR neighbor lateral speed large)
      - "ttc":  closing scenarios with TTC < ttc_thresh (only meaningful for lead/rear; dx and dvx sign matters)
      - "lc_or_ttc": union of lc and ttc
    """
    if mode == "none":
        return True

    lc_like = (abs(lane_change_ego) > 0.0) or (abs(vy_nb) > vy_thresh)

    if mode == "lc":
        return lc_like

    # TTC-ish: define TTC for longitudinal closing.
    # For a lead (dx>0): closing if dvx < 0 (neighbor slower than ego in ego-relative? careful: dvx = v_nb - v_ego)
    # ego approaches lead if (v_ego - v_nb) > 0 => dvx < 0
    # TTC = dx / (v_ego - v_nb) = dx / (-dvx) when dvx < 0
    # For rear (dx<0): rear approaches ego if (v_nb - v_ego) > 0 => dvx > 0, TTC = (-dx) / dvx
    ttc_like = False
    if slot_k == 0:  # lead
        if dx > 0 and dvx < 0:
            ttc = dx / max(-dvx, 1e-3)
            ttc_like = (ttc > 0) and (ttc < ttc_thresh)
    elif slot_k == 1:  # rear
        if dx < 0 and dvx > 0:
            ttc = (-dx) / max(dvx, 1e-3)
            ttc_like = (ttc > 0) and (ttc < ttc_thresh)

    if mode == "ttc":
        return ttc_like
    if mode == "lc_or_ttc":
        return lc_like or ttc_like

    raise ValueError(f"Unknown interaction_mode={mode}")

# -----------------------------
# Main extraction
# -----------------------------
def analyze_one_tracks_csv(
    tracks_csv: Path,
    eps_gate: float,
    min_speed: float,
    max_abs_dx: float,
    interaction_mode: str,
    vy_thresh: float,
    ttc_thresh: float,
    sample_every_n_rows: int,
) -> Dict[str, List[float]]:
    """
    Returns dict bucket_name -> list of dx_time values.
    Buckets include:
      - "all"
      - per-slot "slot:lead", ...
      - groups "same_lane" (k 0,1), "left" (2,3,4), "right"(5,6,7), "side"(2..7)
    """

    df = pd.read_csv(tracks_csv, low_memory=False)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    # Optional downsampling for speed (row-level); still keeps per-frame mapping correct only if we keep all rows.
    # IMPORTANT: for correctness, we should NOT drop rows arbitrarily because frame->pos lookup uses full tracks.
    # So we keep full df in memory, and only sample ego-rows in the loop.
    track_ids = df["id"].to_numpy(dtype=np.int32)
    frames = df["frame"].to_numpy(dtype=np.int32)

    x = _safe_float(df["x"].to_numpy())
    y = _safe_float(df["y"].to_numpy())
    xv = _safe_float(df["xVelocity"].to_numpy())
    yv = _safe_float(df["yVelocity"].to_numpy())

    # exiD-compatible lead / rear mapping
    lead_id = _get_col(
        df,
        ["precedingId", "leadId"],
        default=-1
    )

    rear_id = _get_col(
        df,
        ["followingId", "rearId"],
        default=-1
    )
    side = ensure_side_neighbor_cols(df)

    # laneChange column exists in your preprocessor; use if present
    lane_change_ego = df["laneChange"].fillna(0.0).to_numpy(dtype=np.float32) if "laneChange" in df.columns \
        else np.zeros((len(df),), dtype=np.float32)

    # Pack neighbor ids per row
    nb_ids_all = np.stack([
        lead_id, rear_id,
        side["leftLead"], side["leftAlong"], side["leftRear"],
        side["rightLead"], side["rightAlong"], side["rightRear"]
    ], axis=1).astype(np.int32)

    veh_rows = build_vehicle_row_index(track_ids)
    frame_to_pos = build_frame_pos_maps(frames, veh_rows)

    # Shift origin for numeric stability (not required for dx, but consistent)
    min_x = float(np.min(x))
    min_y = float(np.min(y))
    x_shift = x - min_x
    y_shift = y - min_y

    buckets: Dict[str, List[float]] = {
        "all": [],
        "same_lane": [],
        "side": [],
        "left": [],
        "right": [],
    }
    for name in SLOT_NAMES:
        buckets[f"slot:{name}"] = []

    # Iterate ego tracks
    for ego_tid, idxs in veh_rows.items():
        idxs = idxs.astype(np.int32, copy=False)
        n = len(idxs)
        if n == 0:
            continue

        # sample ego rows for speed (but keep mapping full)
        ego_positions = np.arange(0, n, sample_every_n_rows, dtype=np.int32)
        for p in ego_positions:
            ego_r = int(idxs[p])
            v_ego = float(xv[ego_r])
            if abs(v_ego) < min_speed:
                continue

            row_nb_ids = nb_ids_all[ego_r]  # (K,)
            fr = int(frames[ego_r])

            for k in range(K):
                nid = int(row_nb_ids[k])
                if nid < 0:
                    continue

                d_pos = frame_to_pos.get(nid)
                if d_pos is None:
                    continue
                pos_in = d_pos.get(fr)
                if pos_in is None:
                    continue
                nb_row = int(veh_rows[nid][pos_in])

                dx = float(x_shift[nb_row] - x_shift[ego_r])
                if max_abs_dx > 0 and abs(dx) > max_abs_dx:
                    continue

                dy = float(y_shift[nb_row] - y_shift[ego_r])
                dvx = float(xv[nb_row] - xv[ego_r])
                dvy = float(yv[nb_row] - yv[ego_r])
                vy_nb = float(yv[nb_row])
                lc_ego = float(lane_change_ego[ego_r])

                if not interaction_mask(
                    mode=interaction_mode,
                    slot_k=k,
                    dx=dx, dy=dy, dvx=dvx, dvy=dvy,
                    v_ego=v_ego, vy_nb=vy_nb,
                    lane_change_ego=lc_ego,
                    vy_thresh=vy_thresh,
                    ttc_thresh=ttc_thresh,
                ):
                    continue

                dx_time = dx / (abs(v_ego) + eps_gate)
                if not np.isfinite(dx_time):
                    continue

                buckets["all"].append(dx_time)
                buckets[f"slot:{SLOT_NAMES[k]}"].append(dx_time)

                if k < 2:
                    buckets["same_lane"].append(dx_time)
                else:
                    buckets["side"].append(dx_time)
                    if 2 <= k <= 4:
                        buckets["left"].append(dx_time)
                    elif 5 <= k <= 7:
                        buckets["right"].append(dx_time)

    return buckets

def summarize_buckets(
    buckets: Dict[str, List[float]],
    p_front: float,
    p_back: float,
) -> pd.DataFrame:
    """
    Produce a summary DataFrame:
      count, mean/std, percentiles, suggested t_front / t_back
    Suggested:
      t_front = quantile(values where dx_time>0, p_front)
      t_back  = -quantile(values where dx_time<0, 1-p_back)  (so that P(dx_time > -t_back) ~= p_back)
    """
    rows = []
    for key, vals in buckets.items():
        v = np.asarray(vals, dtype=np.float32)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue

        ps = calc_percentiles(v)

        v_pos = v[v > 0]
        v_neg = v[v < 0]

        t_front = float(np.nan) if len(v_pos) == 0 else float(np.nanpercentile(v_pos, p_front * 100.0))
        # for negative side: want P(dx_time > -t_back) = p_back
        # equivalently, (-t_back) is (1 - p_back) percentile of negative distribution
        t_back = float(np.nan)
        if len(v_neg) > 0:
            q = float(np.nanpercentile(v_neg, (1.0 - p_back) * 100.0))  # q is negative
            t_back = float(-q)

        rows.append({
            "bucket": key,
            "count": int(len(v)),
            "mean": float(np.nanmean(v)),
            "std": float(np.nanstd(v)),
            "p05": ps.p05, "p10": ps.p10, "p25": ps.p25, "p50": ps.p50,
            "p75": ps.p75, "p90": ps.p90, "p95": ps.p95, "p99": ps.p99,
            f"t_front@p{int(p_front*100)}(pos)": t_front,
            f"t_back@p{int(p_back*100)}(neg)": t_back,
        })

    return pd.DataFrame(rows).sort_values(["bucket"]).reset_index(drop=True)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks_dir", type=str, default="raw/", help="Directory containing *_tracks.csv")
    ap.add_argument("--out_dir", type=str, default="out/", help="Output directory for CSV/plots")

    ap.add_argument("--eps_gate", type=float, default=0.1, help="eps added to |v_ego|")
    ap.add_argument("--min_speed", type=float, default=1.0, help="drop rows where |v_ego| < this (m/s)")
    ap.add_argument("--max_abs_dx", type=float, default=150.0, help="drop neighbors with |dx| > this (m). set <=0 to disable")

    ap.add_argument("--interaction_mode", type=str, default="none",
                    choices=["none", "lc", "ttc", "lc_or_ttc"],
                    help="filter to interaction-like cases")
    ap.add_argument("--vy_thresh", type=float, default=0.2, help="for lc mode: |vy_nb| threshold (m/s)")
    ap.add_argument("--ttc_thresh", type=float, default=6.0, help="for ttc mode: TTC threshold (s)")

    ap.add_argument("--p_front", type=float, default=0.95, help="percentile for t_front (positive dx_time)")
    ap.add_argument("--p_back", type=float, default=0.95, help="coverage for negative side used to derive t_back")

    ap.add_argument("--sample_every_n_rows", type=int, default=1,
                    help="speed option: only evaluate every Nth ego row within each track (keeps correctness)")
    ap.add_argument("--make_plots", action="store_true", help="save histograms to out_dir/plots")
    ap.add_argument("--plot_xlim", type=float, nargs=2, default=[-8, 12], help="xlim for hist plots")
    ap.add_argument("--bins", type=int, default=200, help="hist bins")

    args = ap.parse_args()

    tracks_dir = Path(args.tracks_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(tracks_dir.glob("*_tracks.csv"))
    if len(csvs) == 0:
        raise SystemExit(f"No *_tracks.csv found under {tracks_dir}")

    # Aggregate across all files
    agg: Dict[str, List[float]] = {}

    for p in csvs:
        print(f"[LOAD] {p.name}")
        buckets = analyze_one_tracks_csv(
            tracks_csv=p,
            eps_gate=args.eps_gate,
            min_speed=args.min_speed,
            max_abs_dx=args.max_abs_dx,
            interaction_mode=args.interaction_mode,
            vy_thresh=args.vy_thresh,
            ttc_thresh=args.ttc_thresh,
            sample_every_n_rows=max(1, int(args.sample_every_n_rows)),
        )
        for k, vals in buckets.items():
            agg.setdefault(k, []).extend(vals)

    df_sum = summarize_buckets(agg, p_front=args.p_front, p_back=args.p_back)

    tag = f"{args.interaction_mode}_minV{args.min_speed}_dx{args.max_abs_dx}_eps{args.eps_gate}"
    out_csv = out_dir / f"dx_time_summary_{tag}.csv"
    df_sum.to_csv(out_csv, index=False)
    print(f"[SAVE] {out_csv}")

    # Save some overall suggestions plainly
    overall = df_sum[df_sum["bucket"] == "all"]
    if len(overall) == 1:
        rec_front = overall.iloc[0][f"t_front@p{int(args.p_front*100)}(pos)"]
        rec_back  = overall.iloc[0][f"t_back@p{int(args.p_back*100)}(neg)"]
        print(f"[SUGGEST] overall t_front≈{rec_front:.3f}s  t_back≈{rec_back:.3f}s "
              f"(mode={args.interaction_mode})")

    if args.make_plots:
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        xlim = (float(args.plot_xlim[0]), float(args.plot_xlim[1]))

        for key, vals in agg.items():
            v = np.asarray(vals, dtype=np.float32)
            if len(v) < 100:
                continue
            out_png = plots_dir / f"hist_{key.replace(':','_')}_{tag}.png"
            save_hist_png(v, out_png, title=f"{key}  ({tag})", bins=int(args.bins), xlim=xlim)

        print(f"[SAVE] histograms -> {plots_dir}")

if __name__ == "__main__":
    main()