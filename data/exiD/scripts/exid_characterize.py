#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exid_characterize.py

Run a practical dataset characterization for exiD-like format.

Assumed structure (from project root):
  ./raw/
    00_recordingMeta.csv, 00_tracks.csv, 00_tracksMeta.csv
    01_recordingMeta.csv, ...
  ./scripts/
    exid_characterize.py  (this file)
Outputs:
  ./out/exid_summary/
    tables/*.csv
    figs/*.png
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# IO utils
# -----------------------------
def smart_read_csv(path: Path) -> pd.DataFrame:
    """
    exiD files are usually comma-separated, but sometimes you may face ';'.
    """
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, sep=";", low_memory=False)


def find_recordings(raw_dir: Path) -> List[str]:
    """
    Find available recording prefixes: "00", "01", ...
    We look for *_tracks.csv.
    """
    xs = []
    for p in sorted(raw_dir.glob("*_tracks.csv")):
        name = p.name
        xx = name.split("_")[0]
        if len(xx) == 2 and xx.isdigit():
            xs.append(xx)
    return sorted(set(xs))


# -----------------------------
# Core loading
# -----------------------------
def load_one_recording(raw_dir: Path, xx: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tracks_path = raw_dir / f"{xx}_tracks.csv"
    meta_path   = raw_dir / f"{xx}_recordingMeta.csv"
    tmeta_path  = raw_dir / f"{xx}_tracksMeta.csv"

    if not tracks_path.exists():
        raise FileNotFoundError(tracks_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    if not tmeta_path.exists():
        raise FileNotFoundError(tmeta_path)

    tracks = smart_read_csv(tracks_path)
    recmeta = smart_read_csv(meta_path)
    tracks_meta = smart_read_csv(tmeta_path)
    return tracks, recmeta, tracks_meta


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace, unify column names if needed.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def ensure_track_level_fields(tracks: pd.DataFrame, tracks_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure tracks has class/length/width/numFrames by merging tracksMeta if needed.
    """
    tracks = normalize_columns(tracks)
    tracks_meta = normalize_columns(tracks_meta)

    key_cols = ["recordingId", "trackId"]
    for c in key_cols:
        if c not in tracks.columns or c not in tracks_meta.columns:
            raise KeyError(f"Missing key column {c} in tracks or tracksMeta")

    need = []
    for c in ["class", "length", "width", "numFrames", "initialFrame", "finalFrame"]:
        if c not in tracks.columns and c in tracks_meta.columns:
            need.append(c)

    if need:
        tracks = tracks.merge(tracks_meta[key_cols + need], on=key_cols, how="left")

    return tracks


def add_frame_rate(tracks: pd.DataFrame, recmeta: pd.DataFrame) -> pd.DataFrame:
    """
    Merge frameRate into each row via recordingId.
    """
    tracks = normalize_columns(tracks)
    recmeta = normalize_columns(recmeta)
    if "recordingId" not in recmeta.columns:
        raise KeyError("recordingId not in recordingMeta")
    if "frameRate" not in recmeta.columns:
        raise KeyError("frameRate not in recordingMeta (needed for seconds/jerk)")
    tracks = tracks.merge(recmeta[["recordingId", "frameRate"]].drop_duplicates(),
                          on="recordingId", how="left")
    return tracks


# -----------------------------
# Feature computation
# -----------------------------
def compute_speed_mag(tracks: pd.DataFrame) -> pd.Series:
    if "lonVelocity" in tracks.columns and "latVelocity" in tracks.columns:
        lon = pd.to_numeric(tracks["lonVelocity"], errors="coerce")
        lat = pd.to_numeric(tracks["latVelocity"], errors="coerce")
    else:
        # fallback to x/y velocity if lon/lat not present
        lon = pd.to_numeric(tracks.get("xVelocity"), errors="coerce")
        lat = pd.to_numeric(tracks.get("yVelocity"), errors="coerce")
    return np.sqrt(lon**2 + lat**2)


def compute_jerk(tracks: pd.DataFrame) -> pd.DataFrame:
    """
    jerk_lon = diff(lonAcceleration) * frameRate within each (recordingId, trackId) by frame order.
    jerk_lat analog.
    """
    df = tracks.copy()

    # pick lon/lat accel if exists, else x/y
    if "lonAcceleration" in df.columns and "latAcceleration" in df.columns:
        a_lon = pd.to_numeric(df["lonAcceleration"], errors="coerce")
        a_lat = pd.to_numeric(df["latAcceleration"], errors="coerce")
    else:
        a_lon = pd.to_numeric(df.get("xAcceleration"), errors="coerce")
        a_lat = pd.to_numeric(df.get("yAcceleration"), errors="coerce")

    df["_a_lon"] = a_lon
    df["_a_lat"] = a_lat

    # ensure numeric frame
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")

    # group diff
    df = df.sort_values(["recordingId", "trackId", "frame"])
    fr = pd.to_numeric(df["frameRate"], errors="coerce")
    df["jerk_lon"] = df.groupby(["recordingId", "trackId"])["_a_lon"].diff() * fr
    df["jerk_lat"] = df.groupby(["recordingId", "trackId"])["_a_lat"].diff() * fr
    df.drop(columns=["_a_lon", "_a_lat"], inplace=True)
    return df


def compute_track_duration_sec(tracks_meta: pd.DataFrame, recmeta: pd.DataFrame) -> pd.DataFrame:
    """
    duration_sec = numFrames / frameRate per track.
    """
    tm = normalize_columns(tracks_meta).copy()
    rm = normalize_columns(recmeta).copy()

    if "recordingId" not in tm.columns or "trackId" not in tm.columns:
        raise KeyError("tracksMeta missing recordingId/trackId")
    if "numFrames" not in tm.columns:
        raise KeyError("tracksMeta missing numFrames")
    if "frameRate" not in rm.columns:
        raise KeyError("recordingMeta missing frameRate")

    tm = tm.merge(rm[["recordingId", "frameRate"]].drop_duplicates(), on="recordingId", how="left")
    tm["numFrames"] = pd.to_numeric(tm["numFrames"], errors="coerce")
    tm["frameRate"] = pd.to_numeric(tm["frameRate"], errors="coerce")
    tm["duration_sec"] = tm["numFrames"] / tm["frameRate"]
    return tm


def compute_track_level_from_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    """
    Track-level aggregations from per-frame tracks:
    - duration_sec (from frame count / frameRate)
    - trajectory length (use traveledDistance if exists, else cumulative xy distance)
    - ramp presence + ramp duration
    - lane change counts (if laneChange column exists)
    """
    df = tracks.copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    df["frameRate"] = pd.to_numeric(df["frameRate"], errors="coerce")

    # ramp flags
    has_ramp_col = "ramp_type" in df.columns
    if has_ramp_col:
        df["is_onramp"] = (df["ramp_type"] == "onramp")
        df["is_offramp"] = (df["ramp_type"] == "offramp")
        df["is_mainlane"] = (df["ramp_type"] == "none")
    else:
        df["is_onramp"] = False
        df["is_offramp"] = False
        df["is_mainlane"] = True

    # laneChange
    has_lanechange = "laneChange" in df.columns
    if has_lanechange:
        lc = pd.to_numeric(df["laneChange"], errors="coerce").fillna(0)
        df["_laneChange"] = (lc == 1).astype(int)
    else:
        df["_laneChange"] = 0

    # trajectory length
    if "traveledDistance" in df.columns:
        td = pd.to_numeric(df["traveledDistance"], errors="coerce")
        df["_trajlen_piece"] = td
        use_traveled = True
    else:
        # fallback: accumulate Euclidean distance along xCenter/yCenter
        x = pd.to_numeric(df.get("xCenter"), errors="coerce")
        y = pd.to_numeric(df.get("yCenter"), errors="coerce")
        df["_dx"] = df.groupby(["recordingId", "trackId"])[x.name].diff()
        df["_dy"] = df.groupby(["recordingId", "trackId"])[y.name].diff()
        df["_step"] = np.sqrt(df["_dx"]**2 + df["_dy"]**2)
        use_traveled = False

    group_cols = ["recordingId", "trackId"]

    # duration_sec: (max_frame - min_frame + 1)/frameRate
    g = df.groupby(group_cols, as_index=False)
    base = g.agg(
        frame_min=("frame", "min"),
        frame_max=("frame", "max"),
        frameRate=("frameRate", "first"),
        class_=("class", "first"),
        length=("length", "first"),
        width=("width", "first"),
    )
    base["duration_sec"] = (base["frame_max"] - base["frame_min"] + 1) / base["frameRate"]

    # ramp durations
    ramp_stats = g.agg(
        n_frames=("frame", "count"),
        onramp_frames=("is_onramp", "sum"),
        offramp_frames=("is_offramp", "sum"),
        mainlane_frames=("is_mainlane", "sum"),
        laneChange_count=("_laneChange", "sum"),
    )
    base = base.merge(ramp_stats, on=group_cols, how="left")
    base["onramp_sec"] = base["onramp_frames"] / base["frameRate"]
    base["offramp_sec"] = base["offramp_frames"] / base["frameRate"]
    base["mainlane_sec"] = base["mainlane_frames"] / base["frameRate"]
    base["has_onramp"] = base["onramp_frames"] > 0
    base["has_offramp"] = base["offramp_frames"] > 0
    base["has_any_ramp"] = base["has_onramp"] | base["has_offramp"]

    # trajectory length
    if use_traveled:
        # trajlen = max(td)-min(td)
        tlen = df.groupby(group_cols)["_trajlen_piece"].agg(["min", "max"]).reset_index()
        tlen["traj_len_m"] = tlen["max"] - tlen["min"]
        base = base.merge(tlen[group_cols + ["traj_len_m"]], on=group_cols, how="left")
    else:
        # trajlen = sum(step)
        tlen = df.groupby(group_cols)["_step"].sum(min_count=1).reset_index().rename(columns={"_step": "traj_len_m"})
        base = base.merge(tlen, on=group_cols, how="left")
        # cleanup
        df.drop(columns=["_dx", "_dy", "_step"], inplace=True, errors="ignore")

    return base


# -----------------------------
# Scenario definitions (simple)
# -----------------------------
def classify_scenarios_simple(tracks: pd.DataFrame, track_level: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight scenario labels (track-level) for professor demo.
    Needs ramp_type + laneChange. leadId-based cut-in/out is optional.
    """
    tl = track_level.copy()

    # merging: onramp -> mainlane occurred (track has onramp and also mainlane frames)
    tl["scenario_merging"] = tl["has_onramp"] & (tl["mainlane_frames"] > 0)

    # diverging: mainlane -> offramp occurred
    tl["scenario_diverging"] = tl["has_offramp"] & (tl["mainlane_frames"] > 0)

    # lane change: any laneChange frame
    tl["scenario_lane_change"] = tl["laneChange_count"] > 0

    # cut-in/cut-out (optional): needs leadId and laneChange
    # heuristic: around laneChange frame, leadId appears (cut-in) or disappears (cut-out)
    if "leadId" in tracks.columns and "laneChange" in tracks.columns:
        df = tracks.copy()
        df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
        df["laneChange"] = pd.to_numeric(df["laneChange"], errors="coerce").fillna(0).astype(int)
        df["leadId"] = pd.to_numeric(df["leadId"], errors="coerce")

        df = df.sort_values(["recordingId", "trackId", "frame"])
        # lead presence
        df["has_lead"] = (df["leadId"].fillna(-1) != -1).astype(int)

        # identify laneChange frames
        lc_frames = df[df["laneChange"] == 1][["recordingId", "trackId", "frame"]].copy()
        if len(lc_frames) > 0:
            # take first lane change frame per track (simple)
            lc_first = lc_frames.groupby(["recordingId", "trackId"], as_index=False)["frame"].min().rename(columns={"frame": "lc_frame"})

            # check lead presence before/after lc_frame (±5 frames)
            df = df.merge(lc_first, on=["recordingId", "trackId"], how="left")
            df["win"] = (df["frame"] - df["lc_frame"]).abs() <= 5

            # presence just before and after (use last frame < lc, first frame > lc)
            before = df[df["frame"] < df["lc_frame"]].groupby(["recordingId", "trackId"])["has_lead"].max().rename("lead_before")
            after  = df[df["frame"] > df["lc_frame"]].groupby(["recordingId", "trackId"])["has_lead"].max().rename("lead_after")

            ci = pd.concat([before, after], axis=1).reset_index()
            ci["cut_in"] = (ci["lead_before"] == 0) & (ci["lead_after"] == 1)
            ci["cut_out"] = (ci["lead_before"] == 1) & (ci["lead_after"] == 0)

            tl = tl.merge(ci[["recordingId", "trackId", "cut_in", "cut_out"]], on=["recordingId", "trackId"], how="left")
        else:
            tl["cut_in"] = False
            tl["cut_out"] = False
    else:
        tl["cut_in"] = np.nan
        tl["cut_out"] = np.nan

    # dense vs free flow: requires leadDHW or leadTHW
    if "leadTHW" in tracks.columns or "leadDHW" in tracks.columns:
        df = tracks.copy()
        df["leadTHW"] = pd.to_numeric(df.get("leadTHW"), errors="coerce")
        df["leadDHW"] = pd.to_numeric(df.get("leadDHW"), errors="coerce")

        # simple rule: dense if THW < 1.5s OR DHW < 20m (and valid)
        dense = ((df["leadTHW"].notna() & (df["leadTHW"] > 0) & (df["leadTHW"] < 1.5)) |
                 (df["leadDHW"].notna() & (df["leadDHW"] > 0) & (df["leadDHW"] < 20)))
        df["dense_frame"] = dense.astype(int)

        dens = df.groupby(["recordingId", "trackId"])["dense_frame"].mean().reset_index().rename(columns={"dense_frame": "dense_frame_ratio"})
        tl = tl.merge(dens, on=["recordingId", "trackId"], how="left")
        tl["scenario_dense"] = tl["dense_frame_ratio"].fillna(0) >= 0.3  # 30% 이상 dense면 dense track
        tl["scenario_free_flow"] = tl["dense_frame_ratio"].fillna(0) <= 0.05
    else:
        tl["dense_frame_ratio"] = np.nan
        tl["scenario_dense"] = np.nan
        tl["scenario_free_flow"] = np.nan

    return tl


# -----------------------------
# Plot helpers
# -----------------------------
def save_hist(series: pd.Series, out_png: Path, title: str, xlabel: str, bins: int = 100, logy: bool = False):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return
    plt.figure()
    plt.hist(s.to_numpy(), bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    if logy:
        plt.yscale("log")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_bar_counts(counts: pd.Series, out_png: Path, title: str, xlabel: str):
    if counts is None or len(counts) == 0:
        return
    plt.figure()
    counts = counts.sort_values(ascending=False)
    plt.bar(counts.index.astype(str), counts.values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    raw_dir = project_root / "raw"
    out_dir = project_root / "out" / "exid_summary"
    fig_dir = out_dir / "figs"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    xxs = find_recordings(raw_dir)
    if not xxs:
        raise RuntimeError(f"No *_tracks.csv found under {raw_dir}")

    all_tracks = []
    all_recmeta = []
    all_tracks_meta = []

    for xx in xxs:
        tracks, recmeta, tmeta = load_one_recording(raw_dir, xx)
        tracks = normalize_columns(tracks)
        recmeta = normalize_columns(recmeta)
        tmeta = normalize_columns(tmeta)

        tracks = ensure_track_level_fields(tracks, tmeta)
        tracks = add_frame_rate(tracks, recmeta)

        all_tracks.append(tracks)
        all_recmeta.append(recmeta)
        all_tracks_meta.append(tmeta)

        print(f"[LOAD] {xx}: tracks_rows={len(tracks):,}, tracksMeta={len(tmeta):,}, recMeta={len(recmeta):,}")

    tracks_all = pd.concat(all_tracks, ignore_index=True)
    recmeta_all = pd.concat(all_recmeta, ignore_index=True).drop_duplicates()
    tmeta_all = pd.concat(all_tracks_meta, ignore_index=True)

    # --------------------------------------
    # 1) Basic: class distribution + length/width by class
    # --------------------------------------
    if "class" in tracks_all.columns:
        class_counts = tracks_all.drop_duplicates(["recordingId", "trackId"])["class"].value_counts()
        class_counts.to_csv(tab_dir / "vehicle_class_counts.csv", header=["count"])
        save_bar_counts(class_counts, fig_dir / "vehicle_class_counts.png",
                        title="Vehicle class distribution (tracks)", xlabel="class")

    # track lifetime (seconds) from tracksMeta
    try:
        tmeta_dur = compute_track_duration_sec(tmeta_all, recmeta_all)
        tmeta_dur.to_csv(tab_dir / "track_duration_sec_per_track.csv", index=False)
        save_hist(tmeta_dur["duration_sec"], fig_dir / "track_duration_sec_hist.png",
                  title="Track lifetime distribution", xlabel="duration (sec)", bins=120, logy=True)
    except Exception as e:
        print(f"[WARN] track duration from tracksMeta failed: {e}")

    # length/width by class (track-level unique)
    track_unique = tracks_all.drop_duplicates(["recordingId", "trackId"]).copy()
    if "class" in track_unique.columns and "length" in track_unique.columns:
        # numeric conversion
        track_unique["length"] = pd.to_numeric(track_unique["length"], errors="coerce")
        track_unique["width"] = pd.to_numeric(track_unique.get("width"), errors="coerce")

        lw_stats = track_unique.groupby("class")[["length", "width"]].agg(["count", "mean", "std", "median"]).reset_index()
        lw_stats.to_csv(tab_dir / "length_width_by_class.csv", index=False)

        for cls in track_unique["class"].dropna().unique():
            sub = track_unique[track_unique["class"] == cls]
            save_hist(sub["length"], fig_dir / f"length_hist_{cls}.png", f"Length distribution ({cls})", "length (m)", bins=80)
            save_hist(sub["width"], fig_dir / f"width_hist_{cls}.png", f"Width distribution ({cls})", "width (m)", bins=80)

    # --------------------------------------
    # 2) Kinematics: velocity/accel/speed/jerk
    # --------------------------------------
    # speed magnitude
    tracks_all["speed_mag"] = compute_speed_mag(tracks_all)
    save_hist(tracks_all["speed_mag"], fig_dir / "speed_magnitude_hist.png",
              "Speed magnitude distribution", "speed (m/s)", bins=140, logy=True)

    # lon/lat velocity
    if "lonVelocity" in tracks_all.columns:
        save_hist(tracks_all["lonVelocity"], fig_dir / "lonVelocity_hist.png",
                  "Longitudinal velocity distribution", "lonVelocity (m/s)", bins=140, logy=True)
    if "latVelocity" in tracks_all.columns:
        save_hist(tracks_all["latVelocity"], fig_dir / "latVelocity_hist.png",
                  "Lateral velocity distribution", "latVelocity (m/s)", bins=140, logy=True)

    # lon/lat acceleration
    if "lonAcceleration" in tracks_all.columns:
        save_hist(tracks_all["lonAcceleration"], fig_dir / "lonAcceleration_hist.png",
                  "Longitudinal acceleration distribution", "lonAcceleration (m/s^2)", bins=140, logy=True)
    if "latAcceleration" in tracks_all.columns:
        save_hist(tracks_all["latAcceleration"], fig_dir / "latAcceleration_hist.png",
                  "Lateral acceleration distribution", "latAcceleration (m/s^2)", bins=140, logy=True)

    # jerk
    tracks_jerk = compute_jerk(tracks_all)
    tracks_jerk.to_csv(tab_dir / "tracks_with_jerk_sample.csv", index=False)  # large; you may later remove or sample
    save_hist(tracks_jerk["jerk_lon"], fig_dir / "jerk_lon_hist.png",
              "Longitudinal jerk distribution", "jerk_lon (m/s^3)", bins=160, logy=True)
    save_hist(tracks_jerk["jerk_lat"], fig_dir / "jerk_lat_hist.png",
              "Lateral jerk distribution", "jerk_lat (m/s^3)", bins=160, logy=True)

    # --------------------------------------
    # 3) Track-level aggregation: duration, traj length, ramp times, lane changes
    # --------------------------------------
    track_level = compute_track_level_from_tracks(tracks_all)
    track_level.to_csv(tab_dir / "track_level_summary.csv", index=False)

    # trajectory length histogram
    save_hist(track_level["traj_len_m"], fig_dir / "trajectory_length_hist.png",
              "Trajectory length distribution", "trajectory length (m)", bins=140, logy=True)

    # average duration
    avg_dur = track_level["duration_sec"].dropna().mean()
    pd.DataFrame({"avg_duration_sec": [avg_dur]}).to_csv(tab_dir / "avg_trajectory_duration.csv", index=False)

    # ramp vs mainlane trajectory duration compare (simple: two hists)
    if "has_any_ramp" in track_level.columns:
        save_hist(track_level.loc[track_level["has_any_ramp"], "duration_sec"],
                  fig_dir / "duration_sec_ramp_tracks.png",
                  "Trajectory duration (tracks with any ramp)", "duration (sec)", bins=120, logy=True)
        save_hist(track_level.loc[~track_level["has_any_ramp"], "duration_sec"],
                  fig_dir / "duration_sec_mainlane_only.png",
                  "Trajectory duration (mainlane-only tracks)", "duration (sec)", bins=120, logy=True)

        # compare traj length too
        save_hist(track_level.loc[track_level["has_any_ramp"], "traj_len_m"],
                  fig_dir / "trajlen_ramp_tracks.png",
                  "Trajectory length (tracks with any ramp)", "traj len (m)", bins=140, logy=True)
        save_hist(track_level.loc[~track_level["has_any_ramp"], "traj_len_m"],
                  fig_dir / "trajlen_mainlane_only.png",
                  "Trajectory length (mainlane-only tracks)", "traj len (m)", bins=140, logy=True)

    # lane change stats
    if "laneChange_count" in track_level.columns:
        pd.DataFrame({
            "total_laneChange_events": [int(track_level["laneChange_count"].sum())],
            "tracks_with_laneChange": [int((track_level["laneChange_count"] > 0).sum())],
            "mean_laneChange_per_track": [float(track_level["laneChange_count"].mean())],
            "median_laneChange_per_track": [float(track_level["laneChange_count"].median())],
        }).to_csv(tab_dir / "lanechange_overall_stats.csv", index=False)

        save_hist(track_level["laneChange_count"], fig_dir / "laneChange_count_per_track_hist.png",
                  "Lane change count per track", "count", bins=60, logy=True)

        if "class_" in track_level.columns:
            lc_by_class = track_level.groupby("class_")["laneChange_count"].agg(["count", "mean", "median", "sum"]).reset_index()
            lc_by_class.to_csv(tab_dir / "lanechange_by_class.csv", index=False)

    # onramp/offramp lane change ratio (frame-level intersection)
    if "ramp_type" in tracks_all.columns and "laneChange" in tracks_all.columns:
        df = tracks_all.copy()
        df["laneChange"] = pd.to_numeric(df["laneChange"], errors="coerce").fillna(0).astype(int)
        lc_frames = df[df["laneChange"] == 1]
        total_lc = len(lc_frames)
        if total_lc > 0:
            ratio = lc_frames["ramp_type"].value_counts(dropna=False) / total_lc
            ratio.to_csv(tab_dir / "lanechange_ramp_type_ratio.csv", header=["ratio"])
        lc_counts = lc_frames["ramp_type"].value_counts(dropna=False)
        lc_counts.to_csv(tab_dir / "lanechange_ramp_type_counts.csv", header=["count"])
        save_bar_counts(lc_counts, fig_dir / "lanechange_ramp_type_counts.png",
                        "LaneChange events by ramp_type", "ramp_type")

    # --------------------------------------
    # 4) Simple scenario classification + scenario stats
    # --------------------------------------
    scen = classify_scenarios_simple(tracks_all, track_level)
    scen.to_csv(tab_dir / "track_level_scenarios.csv", index=False)

    # scenario counts
    scenario_cols = ["scenario_lane_change", "scenario_merging", "scenario_diverging", "scenario_dense", "scenario_free_flow"]
    present = [c for c in scenario_cols if c in scen.columns]
    if present:
        counts = {c: int(scen[c].fillna(False).sum()) for c in present}
        pd.DataFrame([counts]).to_csv(tab_dir / "scenario_counts.csv", index=False)

    # scenario-wise mean speed/acc (track-level means computed from per-frame)
    # compute per-track means from tracks_all (speed_mag, lon/lat accel)
    df = tracks_all.copy()
    df["speed_mag"] = compute_speed_mag(df)
    df["lonAcceleration"] = pd.to_numeric(df.get("lonAcceleration"), errors="coerce")
    df["latAcceleration"] = pd.to_numeric(df.get("latAcceleration"), errors="coerce")

    per_track_means = df.groupby(["recordingId", "trackId"]).agg(
        mean_speed=("speed_mag", "mean"),
        mean_lonAcc=("lonAcceleration", "mean"),
        mean_latAcc=("latAcceleration", "mean"),
    ).reset_index()

    scen2 = scen.merge(per_track_means, on=["recordingId", "trackId"], how="left")

    rows = []
    for c in ["scenario_lane_change", "scenario_merging", "scenario_diverging"]:
        if c not in scen2.columns:
            continue
        sub = scen2[scen2[c] == True]
        rows.append({
            "scenario": c,
            "n_tracks": int(len(sub)),
            "mean_speed": float(sub["mean_speed"].mean()) if len(sub) else np.nan,
            "mean_lonAcc": float(sub["mean_lonAcc"].mean()) if len(sub) else np.nan,
            "mean_latAcc": float(sub["mean_latAcc"].mean()) if len(sub) else np.nan,
        })
    if rows:
        pd.DataFrame(rows).to_csv(tab_dir / "scenario_mean_kinematics.csv", index=False)

    print(f"\n[DONE] Outputs written to: {out_dir}\n")


if __name__ == "__main__":
    main()
