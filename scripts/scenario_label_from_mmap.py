#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scenario_label_from_mmap.py

Generate window_labels.csv that matches 100% of Mmap (.npy) samples by labeling ONLY the
(recordingId, trackId, t0_frame) keys that exist in the preprocessed Mmap metadata.

Usage example:
  python3 scripts/scenario_label_from_mmap.py \
    --mmap_dir data/exiD/data_mmap/exid_T2_Tf5_hz3 \
    --tag exid \
    --raw_dir data/exiD/raw \
    --out_csv data/exiD/data_mmap/exid_T2_Tf5_hz3/window_labels.csv \
    --history_sec 2 --future_sec 5 --target_hz 3 \
    --highd_offset 100 \
    --adj_pkl maps/lanelet_adj_allmaps.pkl
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal

import numpy as np
import pandas as pd

# ---- Import existing labeling logic from your project ----
# Ensure scripts/scenario_label.py exists and is importable
from scripts.scenario_label import (
    smart_read_csv,
    detect_schema,
    normalize_highd_tracks,
    normalize_exid_tracks,
    normalize_recmeta,
    build_lane_lookup,
    label_window,
    load_lanelet_adjacency,
)

Dataset = Literal["auto", "exid", "highd"]

def _infer_xx_from_recordingId(rid: int, highd_offset: int) -> Tuple[str, str]:
    """
    Return (dataset_guess, xx_str):
      - exiD rid: 0..92 -> xx = rid
      - highD rid: 101..160 -> xx = rid - offset
    """
    if rid >= highd_offset + 1:
        xx = rid - highd_offset
        return ("highd", f"{int(xx):02d}")
    return ("exid", f"{int(rid):02d}")

def main():
    import argparse

    ap = argparse.ArgumentParser()
    # [변경] pt_dir 대신 mmap_dir과 tag를 입력받음
    ap.add_argument("--mmap_dir", type=str, required=True, help="Directory containing *_meta_recordingId.npy etc.")
    ap.add_argument("--tag", type=str, required=True, help="Dataset tag prefix (e.g. 'exid', 'highd')")
    
    ap.add_argument("--raw_dir", type=str, required=True, help="Directory containing raw XX_tracks.csv")
    ap.add_argument("--out_csv", type=str, required=True, help="Output CSV path")

    ap.add_argument("--dataset", type=str, choices=["auto", "exid", "highd"], default="auto",
                    help="Force schema, or auto-detect per recording.")
    ap.add_argument("--highd_offset", type=int, default=100)

    ap.add_argument("--history_sec", type=float, required=True)
    ap.add_argument("--future_sec", type=float, required=True)
    ap.add_argument("--target_hz", type=float, required=True,
                    help="target_hz used when generating NPZ (needed to reconstruct frame span).")

    ap.add_argument("--adj_pkl", type=str, default="", help="Optional lanelet adjacency pickle (exiD).")
    args = ap.parse_args()

    mmap_dir = Path(args.mmap_dir)
    raw_dir = Path(args.raw_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load Adjacency (Optional) ----
    adj_by_map = None
    if args.adj_pkl:
        p = Path(args.adj_pkl)
        if p.exists():
            try:
                adj_by_map = load_lanelet_adjacency(p)
                print(f"[INFO] Loaded adjacency from {p}")
            except Exception as e:
                print(f"[WARN] Failed to load adjacency: {e}")
                adj_by_map = None

    # ---- 1) Load Keys from Mmap Metadata (.npy) ----
    # preprocess_mmap.py가 생성한 메타 파일들
    rec_path = mmap_dir / f"{args.tag}_meta_recordingId.npy"
    trk_path = mmap_dir / f"{args.tag}_meta_trackId.npy"
    t0_path  = mmap_dir / f"{args.tag}_meta_frame.npy"

    if not (rec_path.exists() and trk_path.exists() and t0_path.exists()):
        raise FileNotFoundError(f"Meta .npy files not found in {mmap_dir} with tag '{args.tag}'. "
                                f"Expected: {rec_path.name}, etc.")

    print(f"[INFO] Loading meta keys from {mmap_dir}...")
    
    # Load entire metadata into RAM (usually small enough)
    rids = np.load(rec_path)
    tids = np.load(trk_path)
    t0s  = np.load(t0_path)

    assert len(rids) == len(tids) == len(t0s), "Metadata length mismatch!"
    n_total = len(rids)

    # Group by recordingId for efficient processing
    # keys_by_rid: { rid: [(tid, t0), ...] }
    keys_by_rid: Dict[int, List[Tuple[int, int]]] = {}
    
    for r, t, f0 in zip(rids, tids, t0s):
        keys_by_rid.setdefault(int(r), []).append((int(t), int(f0)))

    print(f"[INFO] Loaded {n_total:,} samples from Mmap metadata. Unique recordings: {len(keys_by_rid)}")

    # ---- 2) Label each key by slicing raw tracks ----
    rows: List[Dict] = []

    # Window length on target_hz grid
    T  = int(round(args.history_sec * args.target_hz))
    Tf = int(round(args.future_sec * args.target_hz))
    win_len = T + Tf

    # Iterate over recordings
    for rid, klist in sorted(keys_by_rid.items()):
        dataset_guess, xx = _infer_xx_from_recordingId(rid, args.highd_offset)

        tracks_path = raw_dir / f"{xx}_tracks.csv"
        recmeta_path = raw_dir / f"{xx}_recordingMeta.csv"
        
        if not tracks_path.exists() or not recmeta_path.exists():
            print(f"[ERROR] Missing raw files for rid={rid} -> xx={xx}")
            print(f"  tracks: {tracks_path}")
            continue

        # Load Raw Data
        tracks = smart_read_csv(tracks_path)
        recmeta = smart_read_csv(recmeta_path)
        tracks.columns = [c.strip() for c in tracks.columns]
        recmeta.columns = [c.strip() for c in recmeta.columns]

        # Detect Schema
        schema = detect_schema(list(tracks.columns), forced=args.dataset)

        # Normalize tracks/recmeta
        if schema == "highd":
            tracks_n = normalize_highd_tracks(tracks, xx=xx, highd_offset=args.highd_offset)
        else:
            tracks_n = normalize_exid_tracks(tracks, highd_offset=args.highd_offset)

        # Map ID for adjacency (exiD only)
        map_id: Optional[int] = None
        if schema == "exid" and "locationId" in recmeta.columns:
            try:
                map_id = int(pd.to_numeric(recmeta["locationId"], errors="coerce").dropna().iloc[0])
            except Exception:
                map_id = None

        recmeta_n = normalize_recmeta(recmeta, dataset=schema, xx=xx, highd_offset=args.highd_offset)

        # Lane Lookup
        lane_col, lane_lookup = build_lane_lookup(tracks_n, schema=schema)

        # Join FrameRate
        tracks_n = tracks_n.merge(recmeta_n, on="recordingId", how="left")
        tracks_n["frame"] = pd.to_numeric(tracks_n["frame"], errors="coerce")
        tracks_n = tracks_n.dropna(subset=["frame"])
        tracks_n["frame"] = tracks_n["frame"].astype(int)
        tracks_n = tracks_n.sort_values(["recordingId", "trackId", "frame"])

        # FrameRate Check
        fr = float(recmeta_n["frameRate"].iloc[0])
        if not np.isfinite(fr) or fr <= 0:
            print(f"[SKIP] Invalid frameRate for rid={rid} (xx={xx})")
            continue

        # Calculate downsample step (ds_step) used in preprocessing
        ds_step = max(1, int(round(fr / float(args.target_hz))))

        # De-duplicate keys (multiple samples might map to same key if augmented, though unlikely here)
        kset = sorted(set(klist))
        print(f"[PROCESS] rid={rid} ({schema}) keys={len(kset):,} ds_step={ds_step} fr={fr}")

        # Group by trackId for fast access
        by_tid: Dict[int, pd.DataFrame] = {int(t): g for t, g in tracks_n.groupby("trackId", sort=False)}

        # Process each sample key in this recording
        for tid, t0_frame in kset:
            g = by_tid.get(int(tid))
            
            # Common failure fallback
            def add_unknown():
                rows.append({
                    "recordingId": int(rid),
                    "trackId": int(tid),
                    "t0_frame": int(t0_frame),
                    "t1_frame": int(t0_frame),
                    "event_label": "unknown",
                    "state_label": "unknown",
                    "schema": schema,
                    "frameRate": fr,
                    "history_sec": float(args.history_sec),
                    "future_sec": float(args.future_sec),
                    "target_hz": float(args.target_hz),
                    "ds_step": int(ds_step),
                })

            if g is None:
                add_unknown()
                continue

            # Calculate window end frame based on downsampling
            t1_frame = int(t0_frame + ds_step * (win_len - 1))
            
            # Slice the raw dataframe for this window [t0, t1]
            w = g[(g["frame"] >= int(t0_frame)) & (g["frame"] <= int(t1_frame))]

            if len(w) == 0:
                add_unknown()
                continue

            # Perform Labeling
            out = label_window(
                w=w,
                frameRate=fr,
                schema=schema,
                lane_col=lane_col,
                lane_lookup=lane_lookup,
                map_id=map_id,
                adj_by_map=adj_by_map,
            )

            # Append Metadata
            out["recordingId"] = int(rid)
            out["trackId"] = int(tid)
            out["t0_frame"] = int(t0_frame)
            out["t1_frame"] = int(t1_frame)
            out["frameRate"] = float(fr)
            out["history_sec"] = float(args.history_sec)
            out["future_sec"] = float(args.future_sec)
            out["target_hz"] = float(args.target_hz)
            out["ds_step"] = int(ds_step)
            out["schema"] = schema

            rows.append(out)

    # ---- Save Result ----
    df = pd.DataFrame(rows)
    
    # Sort for consistency
    if not df.empty:
        df = df.sort_values(["recordingId", "trackId", "t0_frame"])
        
    df.to_csv(out_csv, index=False)
    print(f"\n[DONE] Wrote {len(df):,} labels to -> {out_csv}")

    # Sanity Check
    if not df.empty:
        if len(df) != n_total:
            print(f"[WARN] Input samples ({n_total}) != Output labels ({len(df)}). Some keys might be missing raw data.")
        
        if "event_label" in df.columns:
            print("\nEvent counts:")
            print(df["event_label"].value_counts())
        if "state_label" in df.columns:
            print("\nState counts:")
            print(df["state_label"].value_counts())

if __name__ == "__main__":
    main()