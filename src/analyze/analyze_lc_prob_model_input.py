#!/usr/bin/env python3
"""
Find Optimal vy Threshold using Smoothed Precision Curve
- Goal: Find threshold x where P(LC | v > x) is high (e.g., 95%).
- Uses KDE smoothing to remove staircase effect in CDF.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm

def get_model_input_vys(tracks_df, dataset_type="exid", frame_rate=25.0):
    lc_vys = []
    lk_vys = []
    
    t_start_offset = 4.5
    t_end_offset = 2.5
    
    offset_start_frm = int(t_start_offset * frame_rate)
    offset_end_frm = int(t_end_offset * frame_rate)
    window_len = offset_start_frm - offset_end_frm 
    
    if "trackId" not in tracks_df.columns: return [], []

    for tid, group in tracks_df.groupby("trackId"):
        group = group.sort_values("frame")
        
        is_lc = False
        lc_idx = -1
        
        if dataset_type == "exid":
            if "laneChange" in group.columns:
                lc_frames = group[group["laneChange"] > 0]
                if not lc_frames.empty:
                    is_lc = True
                    lc_idx = group.index.get_loc(lc_frames.index[0])
        else:
            if "laneId" in group.columns:
                lids = group["laneId"].values
                if len(lids) > 1:
                    diffs = np.where(lids[:-1] != lids[1:])[0]
                    if len(diffs) > 0:
                        is_lc = True
                        lc_idx = diffs[0] + 1 

        vys = None
        if "latVelocity" in group.columns: vys = group["latVelocity"].abs().values
        elif "yVelocity" in group.columns: vys = group["yVelocity"].abs().values
        else: continue

        if is_lc:
            start_idx = lc_idx - offset_start_frm
            end_idx = lc_idx - offset_end_frm
            if start_idx >= 0 and end_idx < len(vys):
                win = vys[start_idx : end_idx]
                if len(win) > 0:
                    lc_vys.append(np.max(win))
        else:
            if len(vys) > window_len:
                mid = len(vys) // 2
                start = max(0, mid - window_len // 2)
                end = start + window_len
                if end <= len(vys):
                    win = vys[start:end]
                    if len(win) > 0:
                        lk_vys.append(np.max(win))
    return lc_vys, lk_vys

def load_data(raw_dir, dataset_type):
    files = list(Path(raw_dir).glob("*_tracks.csv"))
    all_lc, all_lk = [], []
    print(f"[INFO] Loading {len(files)} {dataset_type} files...")
    for f in tqdm(files):
        try:
            meta = f.with_name(f.name.replace("_tracks.csv", "_recordingMeta.csv"))
            fps = 25.0
            if meta.exists():
                try: fps = float(pd.read_csv(meta, nrows=1)["frameRate"].iloc[0])
                except: pass
            
            iter_csv = pd.read_csv(f, nrows=1)
            cols = ["trackId", "id", "frame", "laneChange", "latVelocity", "yVelocity", "laneId"]
            use = [c for c in cols if c in iter_csv.columns]
            df = pd.read_csv(f, usecols=use)
            if "id" in df.columns: df = df.rename(columns={"id": "trackId"})
            
            c, k = get_model_input_vys(df, dataset_type, fps)
            all_lc.extend(c)
            all_lk.extend(k)
        except Exception: pass
    return all_lc, all_lk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exid_raw", type=str, default="data/exiD/raw")
    ap.add_argument("--highd_raw", type=str, default="data/highD/raw")
    ap.add_argument("--lc_prior", type=float, default=0.5, help="Assumed prior P(LC)")
    ap.add_argument("--bandwidth", type=float, default=0.2, help="Smoothing factor")
    ap.add_argument("--out_file", type=str, default="src/analyze/out/threshold_precision.png")
    args = ap.parse_args()

    # 1. Load Data
    lc_data, lk_data = [], []
    if args.exid_raw:
        c, k = load_data(args.exid_raw, "exid")
        lc_data.extend(c); lk_data.extend(k)
    if args.highd_raw:
        c, k = load_data(args.highd_raw, "highd")
        lc_data.extend(c); lk_data.extend(k)

    if not lc_data: print("No data."); return

    # 2. Smooth PDF using KDE
    # Create grid
    max_val = 1.5
    x_grid = np.linspace(0, max_val, 1000)
    dx = x_grid[1] - x_grid[0]
    
    kde_lc = gaussian_kde(lc_data, bw_method=args.bandwidth)
    kde_lk = gaussian_kde(lk_data, bw_method=args.bandwidth)
    
    pdf_lc = kde_lc(x_grid)
    pdf_lk = kde_lk(x_grid)
    
    # 3. Calculate "Tail Probability" (Area under curve > x)
    # 1 - CDF와 같은 개념입니다. (Survival Function)
    # 적분(cumsum)을 뒤에서부터 하면 P(v > x)를 구할 수 있습니다.
    tail_lc = np.cumsum(pdf_lc[::-1])[::-1] * dx
    tail_lk = np.cumsum(pdf_lk[::-1])[::-1] * dx
    
    # Normalize (Ensure max is 1.0)
    tail_lc /= tail_lc[0]
    tail_lk /= tail_lk[0]

    # 4. Calculate Precision Curve
    # Precision(x) = P(LC | v > x)
    #              = P(v > x | LC) * P(LC) / P(v > x)
    #              = (tail_lc * prior) / (tail_lc * prior + tail_lk * (1-prior))
    prior = args.lc_prior
    
    # Avoid division by zero
    denominator = (tail_lc * prior) + (tail_lk * (1 - prior)) + 1e-10
    precision_curve = (tail_lc * prior) / denominator
    
    # Monotonic Constraint (Optional but good for stability)
    # "속도 기준을 높일수록 LC일 확률은 올라가야 한다"
    precision_curve = np.maximum.accumulate(precision_curve)

    # 5. Find Thresholds
    targets = [0.90, 0.95, 0.99]
    found = {}
    
    print("\n" + "="*60)
    print(f" PRECISION-BASED THRESHOLDS (Prior LC={prior})")
    print("="*60)
    
    # 0.1 m/s 미만은 무시
    start_idx = np.searchsorted(x_grid, 0.1)
    
    for t in targets:
        # Find first index where precision > t
        sub_prec = precision_curve[start_idx:]
        sub_grid = x_grid[start_idx:]
        
        idx = np.where(sub_prec > t)[0]
        if len(idx) > 0:
            val = sub_grid[idx[0]]
            found[t] = val
            print(f" Confidence (Precision) > {int(t*100)}% : |vy| > {val:.4f} m/s")
        else:
            print(f" Confidence (Precision) > {int(t*100)}% : Not found")

    # 6. Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Background Density (Optional)
    ax1.fill_between(x_grid, pdf_lk, color='blue', alpha=0.1, label='P(v|LK) Density')
    ax1.fill_between(x_grid, pdf_lc, color='red', alpha=0.1, label='P(v|LC) Density')
    ax1.set_ylabel("Probability Density", color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')
    ax1.set_ylim(0, max(pdf_lk.max(), pdf_lc.max()) * 1.1)
    ax1.set_xlim(0, max_val)

    # Precision Curve (Main)
    ax2 = ax1.twinx()
    ax2.plot(x_grid, precision_curve, color='green', linewidth=3, label='Precision: P(LC | v > x)')
    
    colors = {0.90: 'orange', 0.95: 'blue', 0.99: 'red'}
    for t, val in found.items():
        c = colors[t]
        ax2.axvline(val, color=c, linestyle='--', linewidth=1.5)
        ax2.scatter([val], [t], color=c, s=50, zorder=10)
        ax2.text(val+0.02, t-0.05, f"{int(t*100)}%\n{val:.2f}m/s", color=c, fontweight='bold')

    ax2.set_ylabel("Precision (Confidence)", color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(0, 1.05)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labels+labels2, loc='center right', fontsize=8)

    plt.title("Lane Change Detection Threshold via Precision Analysis")
    plt.tight_layout()
    plt.savefig(args.out_file)
    print(f"[INFO] Plot saved to {args.out_file}")

if __name__ == "__main__":
    main()