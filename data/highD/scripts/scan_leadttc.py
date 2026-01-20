#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks_dir", type=str, default="raw", help="directory containing *_tracksMeta.csv")
    ap.add_argument("--glob", type=str, default="*_tracksMeta.csv")
    ap.add_argument("--col", type=str, default="minDHW", help="column name for min TTC in tracksMeta")
    args = ap.parse_args()

    tracks_dir = Path(args.tracks_dir).resolve()
    paths = sorted(tracks_dir.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"No files: {tracks_dir}/{args.glob}")

    all_vals = []

    for p in paths:
        df = pd.read_csv(p, low_memory=False)

        if args.col not in df.columns:
            print(f"[WARN] {p.name}: no '{args.col}' column, skipped")
            continue

        x = df[args.col].to_numpy(dtype=np.float64)

        # -1 means invalid -> drop
        mask = np.isfinite(x) & (x != -1.0)
        vals = x[mask]

        if vals.size > 0:
            all_vals.append(vals)

        dropped = x.size - vals.size
        print(f"[OK] {p.name}: kept {vals.size} {args.col} rows (dropped {dropped})")

    if not all_vals:
        print(f"No valid '{args.col}' values found (after dropping -1 / non-finite).")
        return

    x = np.concatenate(all_vals, axis=0)

    print(f"\n==== {args.col} Distribution (tracksMeta, filtered) ====")
    print(f"count: {x.size}")
    print(f"min/max: {x.min():.6g} / {x.max():.6g}")
    print(f"mean/std: {x.mean():.6g} / {x.std():.6g}")
    for q in [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.5, 99.9, 100]:
        print(f"p{q:>5}: {np.percentile(x, q):.6g}")

if __name__ == "__main__":
    main()