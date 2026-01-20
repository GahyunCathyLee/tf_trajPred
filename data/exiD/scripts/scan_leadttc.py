#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks_dir", type=str, default="raw", help="directory containing *_tracks.csv")
    ap.add_argument("--glob", type=str, default="*_tracks.csv")
    ap.add_argument("--lane_width_min", type=float, default=1.0, help="apply laneWidth >= this (set <=0 to disable)")
    ap.add_argument("--hist_bins", type=int, default=200)
    ap.add_argument("--hist_max", type=float, default=60.0, help="hist range max for visualization (seconds)")
    args = ap.parse_args()

    tracks_dir = Path(args.tracks_dir).resolve()
    paths = sorted(tracks_dir.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"No files: {tracks_dir}/{args.glob}")

    all_vals = []

    for p in paths:
        df = pd.read_csv(p, low_memory=False)

        if args.lane_width_min > 0:
            if "laneWidth" not in df.columns:
                print(f"[WARN] {p.name}: no laneWidth column, lane filter disabled for this file")
            else:
                lane_w = df["laneWidth"].to_numpy(dtype=np.float64)
                mask = np.isfinite(lane_w) & (lane_w >= args.lane_width_min)

        vals = lane_w[mask]
        if vals.size > 0:
            all_vals.append(vals)

        print(f"[OK] {p.name}: kept {vals.size} laneWidth frames")

    if not all_vals:
        print("No laneWidth values found with the given filters.")
        return

    x = np.concatenate(all_vals, axis=0)

    # summary
    print("\n==== laneWidth Distribution (filtered) ====")
    print(f"count: {x.size}")
    print(f"min/max: {x.min():.6g} / {x.max():.6g}")
    print(f"mean/std: {x.mean():.6g} / {x.std():.6g}")
    for q in [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.5, 99.9, 100]:
        print(f"p{q:>5}: {np.percentile(x, q):.6g}")

if __name__ == "__main__":
    main()
