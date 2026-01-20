#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scenario_label_from_pt.py

Generate window_labels.csv that matches 100% of PT samples by labeling ONLY the
(recordingId, trackId, t0_frame) keys that exist in PT files.

Why:
- 기존 scenario_label.py는 raw frame grid(stride_sec*frameRate)로 window를 생성하지만,
  PT/NPZ는 target_hz grid(stride_sec*target_hz) + ds_step(round(frameRate/target_hz))로 t0_frame이 결정됨.
  → t0_frame mismatch로 LUT 매칭이 깨짐.

Usage example:
  python3 scripts/scenario_label_from_pt.py \
    --pt_dir data/exiD/data_pt/exid_T2_Tf5_hz3 \
    --raw_dir data/exiD/raw \
    --out_csv data/exiD/out/scenarios/window_labels.csv \
    --history_sec 2 --future_sec 5 --target_hz 3 \
    --highd_offset 100 \
    --adj_pkl maps/lanelet_adj_allmaps.pkl
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal

import numpy as np
import pandas as pd
import torch

# ---- import your existing labeling logic ----
# Put scenario_label.py in your repo import path (or adjust import accordingly)
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


def _load_pt_keys(pt_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    # Expect these keys (you confirmed they exist)
    rid = obj["recordingId"].cpu().numpy().astype(np.int32)
    tid = obj["trackId"].cpu().numpy().astype(np.int32)
    t0  = obj["t0_frame"].cpu().numpy().astype(np.int32)
    return rid, tid, t0


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
    ap.add_argument("--pt_dir", type=str, required=True, help="Directory containing *.pt windows")
    ap.add_argument("--pt_glob", type=str, default="*.pt")
    ap.add_argument("--raw_dir", type=str, required=True, help="Directory containing XX_tracks.csv / XX_recordingMeta.csv")
    ap.add_argument("--out_csv", type=str, required=True)

    ap.add_argument("--dataset", type=str, choices=["auto", "exid", "highd"], default="auto",
                    help="Force schema, or auto-detect per recording.")
    ap.add_argument("--highd_offset", type=int, default=100)

    ap.add_argument("--history_sec", type=float, required=True)
    ap.add_argument("--future_sec", type=float, required=True)
    ap.add_argument("--target_hz", type=float, required=True,
                    help="target_hz used when generating PT/NPZ (needed to reconstruct frame span).")

    ap.add_argument("--adj_pkl", type=str, default="", help="Optional lanelet adjacency pickle (exiD).")
    args = ap.parse_args()

    pt_dir = Path(args.pt_dir)
    raw_dir = Path(args.raw_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # ---- load adjacency (optional) ----
    adj_by_map = None
    if args.adj_pkl:
        p = Path(args.adj_pkl)
        if p.exists():
            try:
                adj_by_map = load_lanelet_adjacency(p)
            except Exception:
                adj_by_map = None

    # ---- 1) collect ALL keys from PT files ----
    keys_by_rid: Dict[int, List[Tuple[int, int]]] = {}  # rid -> list of (tid, t0)
    n_total = 0

    pt_paths = sorted(pt_dir.glob(args.pt_glob))
    if not pt_paths:
        raise SystemExit(f"No PT files found: {pt_dir}/{args.pt_glob}")

    for p in pt_paths:
        rid, tid, t0 = _load_pt_keys(p)
        assert len(rid) == len(tid) == len(t0)
        n_total += len(rid)
        for r, t, f0 in zip(rid.tolist(), tid.tolist(), t0.tolist()):
            keys_by_rid.setdefault(int(r), []).append((int(t), int(f0)))

    print(f"[INFO] Loaded PT keys: {n_total:,} samples, unique recordingId={len(keys_by_rid)}")

    # ---- 2) label each key by slicing raw tracks ----
    rows: List[Dict] = []

    # window length on target_hz grid
    T  = int(round(args.history_sec * args.target_hz))
    Tf = int(round(args.future_sec * args.target_hz))
    win_len = T + Tf

    for rid, klist in sorted(keys_by_rid.items()):
        dataset_guess, xx = _infer_xx_from_recordingId(rid, args.highd_offset)

        tracks_path = raw_dir / f"{xx}_tracks.csv"
        recmeta_path = raw_dir / f"{xx}_recordingMeta.csv"
        if not tracks_path.exists() or not recmeta_path.exists():
            raise SystemExit(f"[ERROR] Missing raw files for rid={rid} -> xx={xx}\n"
                             f"  tracks: {tracks_path}\n  recmeta: {recmeta_path}")

        tracks = smart_read_csv(tracks_path)
        recmeta = smart_read_csv(recmeta_path)
        tracks.columns = [c.strip() for c in tracks.columns]
        recmeta.columns = [c.strip() for c in recmeta.columns]

        schema = detect_schema(list(tracks.columns), forced=args.dataset)

        # Normalize tracks/recmeta (same as scenario_label.py)
        if schema == "highd":
            tracks_n = normalize_highd_tracks(tracks, xx=xx, highd_offset=args.highd_offset)
        else:
            tracks_n = normalize_exid_tracks(tracks, highd_offset=args.highd_offset)

        # IMPORTANT: keep raw recmeta for locationId (scenario_label.py normalize_recmeta drops it)
        map_id: Optional[int] = None
        if schema == "exid" and "locationId" in recmeta.columns:
            try:
                map_id = int(pd.to_numeric(recmeta["locationId"], errors="coerce").dropna().iloc[0])
            except Exception:
                map_id = None

        recmeta_n = normalize_recmeta(recmeta, dataset=schema, xx=xx, highd_offset=args.highd_offset)

        # lane lookup
        lane_col, lane_lookup = build_lane_lookup(tracks_n, schema=schema)

        # join frameRate
        tracks_n = tracks_n.merge(recmeta_n, on="recordingId", how="left")
        tracks_n["frame"] = pd.to_numeric(tracks_n["frame"], errors="coerce")
        tracks_n = tracks_n.dropna(subset=["frame"])
        tracks_n["frame"] = tracks_n["frame"].astype(int)
        tracks_n = tracks_n.sort_values(["recordingId", "trackId", "frame"])

        # frameRate for this recording
        fr = float(recmeta_n["frameRate"].iloc[0])
        if not np.isfinite(fr) or fr <= 0:
            raise SystemExit(f"[ERROR] invalid frameRate for rid={rid} (xx={xx})")

        # reconstruct ds_step used in PT sampling:
        # exiD: ds_step = round(frameRate/target_hz)
        # highD: frameRate is also 25, so same formula works
        ds_step = max(1, int(round(fr / float(args.target_hz))))

        # de-dup keys (PT may contain duplicates across multiple pt shards)
        kset = sorted(set(klist))
        print(f"[PROCESS] rid={rid} schema={schema} xx={xx} keys={len(kset):,} ds_step={ds_step} frameRate={fr}")

        # group by trackId for fast slicing
        by_tid: Dict[int, pd.DataFrame] = {int(t): g for t, g in tracks_n.groupby("trackId", sort=False)}

        for tid, t0_frame in kset:
            g = by_tid.get(int(tid))
            if g is None:
                # keep the key but mark unknown so it still matches
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
                continue

            t1_frame = int(t0_frame + ds_step * (win_len - 1))
            w = g[(g["frame"] >= int(t0_frame)) & (g["frame"] <= int(t1_frame))]

            if len(w) == 0:
                rows.append({
                    "recordingId": int(rid),
                    "trackId": int(tid),
                    "t0_frame": int(t0_frame),
                    "t1_frame": int(t1_frame),
                    "event_label": "unknown",
                    "state_label": "unknown",
                    "schema": schema,
                    "frameRate": fr,
                    "history_sec": float(args.history_sec),
                    "future_sec": float(args.future_sec),
                    "target_hz": float(args.target_hz),
                    "ds_step": int(ds_step),
                })
                continue

            out = label_window(
                w=w,
                frameRate=fr,
                schema=schema,
                lane_col=lane_col,
                lane_lookup=lane_lookup,
                map_id=map_id,
                adj_by_map=adj_by_map,
            )

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

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[DONE] wrote {len(df):,} rows -> {out_csv}")

    # quick sanity: duplicates on key?
    dup = df.duplicated(subset=["recordingId", "trackId", "t0_frame"]).sum()
    if dup > 0:
        print(f"[WARN] duplicated keys in output: {dup}")

    if "event_label" in df.columns:
        print("\nEvent counts:\n", df["event_label"].value_counts())
    if "state_label" in df.columns:
        print("\nState counts:\n", df["state_label"].value_counts())


if __name__ == "__main__":
    main()