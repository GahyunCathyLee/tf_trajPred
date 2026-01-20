#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenario_report_tracklevel.py

Track-level report from window-level labels (window_labels.csv).

Input:
  ./out/scenarios/window_labels.csv

Optional:
  ./raw/*_tracksMeta.csv  (to attach vehicle class; best effort)

Outputs (track-level):
  ./out/scenarios/report_track/track_summary.csv
  ./out/scenarios/report_track/track_event_presence_counts.csv
  ./out/scenarios/report_track/track_state_presence_counts.csv
  ./out/scenarios/report_track/event_presence_by_class.csv      (if class available)
  ./out/scenarios/report_track/summary.txt

Design:
- Each track aggregates ALL its window labels.
- Multi-scenario handling:
  - present_event_labels: pipe-joined set of event labels appearing in track
  - present_state_labels: pipe-joined set of state labels appearing in track
  - dominant_event_label: most frequent event label among windows
  - dominant_state_label: most frequent state label among windows
  - event_ratio_* columns: per-label ratio within track (for common labels)
  - transitions_event: number of label changes along time-ordered windows
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
import numpy as np


def smart_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, sep=";", low_memory=False)


def _maybe_load_class_lookup(raw_dir: Path) -> Optional[pd.DataFrame]:
    """Build (recordingId, trackId) -> class from *_tracksMeta.csv if available."""
    metas = sorted(raw_dir.glob("*_tracksMeta.csv"))
    if not metas:
        return None

    parts = []
    for p in metas:
        try:
            m = smart_read_csv(p)
        except Exception:
            continue

        m.columns = [c.strip() for c in m.columns]
        if "recordingId" not in m.columns or "trackId" not in m.columns:
            continue

        class_col = None
        for cand in ["class", "vehicleClass", "vehicle_class", "agentClass"]:
            if cand in m.columns:
                class_col = cand
                break
        if class_col is None:
            continue

        mm = m[["recordingId", "trackId", class_col]].copy()
        mm = mm.rename(columns={class_col: "class"})
        parts.append(mm)

    if not parts:
        return None

    out = pd.concat(parts, ignore_index=True)
    out["recordingId"] = pd.to_numeric(out["recordingId"], errors="coerce").astype("Int64")
    out["trackId"] = pd.to_numeric(out["trackId"], errors="coerce").astype("Int64")
    out["class"] = out["class"].astype(str)
    out = out.dropna(subset=["recordingId", "trackId"]).drop_duplicates(["recordingId", "trackId"])
    return out


def pipe_join_unique(vals: pd.Series) -> str:
    s = vals.dropna().astype(str)
    uniq = sorted(set(s.tolist()))
    return "|".join(uniq) if uniq else ""


def dominant_label(vals: pd.Series) -> str:
    s = vals.dropna().astype(str)
    if len(s) == 0:
        return ""
    vc = s.value_counts()
    return str(vc.index[0])


def count_transitions(labels_in_time: pd.Series) -> int:
    """
    Count number of times label changes along ordered windows.
    e.g. A A B B A -> transitions = 2 (A->B, B->A)
    """
    s = labels_in_time.dropna().astype(str).tolist()
    if len(s) <= 1:
        return 0
    t = 0
    prev = s[0]
    for x in s[1:]:
        if x != prev:
            t += 1
            prev = x
    return t


def add_ratio_columns(
    track_df: pd.DataFrame,
    group_cols: List[str],
    label_col: str,
    prefix: str,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    For each track, compute ratios of the most common labels (global top_k).
    Adds columns like f"{prefix}_{label}" = ratio in that track.

    Example:
      label_col="event_label", prefix="event_ratio"
      -> event_ratio_merging, event_ratio_cut_in, ...

    Note: label values are sanitized to be column-safe.
    """
    # choose global top labels
    labels = track_df[label_col].fillna("NA").astype(str)
    top_labels = labels.value_counts().head(top_k).index.tolist()

    # compute per-track counts
    ctab = pd.crosstab(
        index=[track_df[c] for c in group_cols],
        columns=labels,
    )

    # ensure top labels exist
    for lab in top_labels:
        if lab not in ctab.columns:
            ctab[lab] = 0

    # total windows per track
    totals = ctab.sum(axis=1).replace(0, np.nan)

    out = ctab[top_labels].div(totals, axis=0).fillna(0.0)

    # rename columns
    def sanitize(x: str) -> str:
        x = x.strip().replace(" ", "_").replace("/", "_").replace("-", "_")
        x = x.replace("__", "_")
        return x

    out.columns = [f"{prefix}_{sanitize(str(c))}" for c in out.columns]
    out = out.reset_index()

    return out


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    in_csv = project_root / "out" / "scenarios" / "highd_window_labels.csv"
    raw_dir = project_root / "raw"
    out_dir = project_root / "out" / "scenarios" / "report_track"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(f"Missing: {in_csv} (run scenario_label_windows.py first)")

    df = smart_read_csv(in_csv)
    df.columns = [c.strip() for c in df.columns]

    # Required keys
    for c in ["recordingId", "trackId"]:
        if c not in df.columns:
            raise KeyError(f"window_labels.csv missing required column: {c}")

    # For ordering windows
    order_col = "t0_frame" if "t0_frame" in df.columns else ("t_mid_frame" if "t_mid_frame" in df.columns else None)
    if order_col is None:
        raise KeyError("Need t0_frame or t_mid_frame in window_labels.csv to order windows per track.")

    # Attach class if possible
    class_lut = _maybe_load_class_lookup(raw_dir)
    has_class = False
    if class_lut is not None:
        df = df.merge(class_lut, on=["recordingId", "trackId"], how="left")
        has_class = True

    # Normalize label cols
    if "event_label" not in df.columns:
        df["event_label"] = "NA"
    if "state_label" not in df.columns:
        df["state_label"] = "NA"

    df["event_label"] = df["event_label"].fillna("NA").astype(str)
    df["state_label"] = df["state_label"].fillna("NA").astype(str)

    # Track-level aggregation
    group_cols = ["recordingId", "trackId"]
    agg_rows = []

    # sort for transitions
    df_sorted = df.sort_values(group_cols + [order_col])

    for (rid, tid), g in df_sorted.groupby(group_cols, sort=False):
        row: Dict = {
            "recordingId": rid,
            "trackId": tid,
            "n_windows": int(len(g)),
            "t0_min": int(g[order_col].min()),
            "t0_max": int(g[order_col].max()),
        }

        if has_class:
            # class might be NaN
            row["class"] = str(g["class"].dropna().iloc[0]) if ("class" in g.columns and g["class"].notna().any()) else ""

        # multi-label presence
        row["present_event_labels"] = pipe_join_unique(g["event_label"])
        row["present_state_labels"] = pipe_join_unique(g["state_label"])

        # dominant labels
        row["dominant_event_label"] = dominant_label(g["event_label"])
        row["dominant_state_label"] = dominant_label(g["state_label"])

        # event flags (useful in report)
        ev_set = set(g["event_label"].tolist())
        row["has_merging"] = int("merging" in ev_set)
        row["has_diverging"] = int("diverging" in ev_set)
        row["has_cut_in"] = int("cut_in" in ev_set)
        row["has_simple_lc"] = int("simple_lane_change" in ev_set)
        row["has_any_lane_change"] = int(any(x in ev_set for x in ["merging", "diverging", "cut_in", "simple_lane_change", "lane_change_other"]))

        # transitions
        row["transitions_event"] = count_transitions(g["event_label"])
        row["transitions_state"] = count_transitions(g["state_label"])

        # mean ratios (if available)
        for col in ["dense_ratio", "free_ratio", "onramp_ratio", "offramp_ratio", "mainlane_ratio"]:
            if col in g.columns:
                row[f"mean_{col}"] = float(pd.to_numeric(g[col], errors="coerce").mean())
            else:
                row[f"mean_{col}"] = np.nan

        agg_rows.append(row)

    track_summary = pd.DataFrame(agg_rows)

    # Add per-track label ratios for top labels (event/state)
    event_ratios = add_ratio_columns(df, group_cols, "event_label", "event_ratio", top_k=10)
    state_ratios = add_ratio_columns(df, group_cols, "state_label", "state_ratio", top_k=10)

    track_summary = track_summary.merge(event_ratios, on=group_cols, how="left")
    track_summary = track_summary.merge(state_ratios, on=group_cols, how="left")

    # Save track summary
    out_track = out_dir / "track_summary.csv"
    track_summary.to_csv(out_track, index=False)

    # Presence counts across tracks
    # event presence: count tracks that contain each event label at least once
    # derive from present_event_labels column
    def explode_presence(col: str) -> pd.DataFrame:
        tmp = track_summary[[*group_cols, col]].copy()
        tmp[col] = tmp[col].fillna("").astype(str)
        tmp = tmp.assign(_lab=tmp[col].str.split("|")).explode("_lab")
        tmp["_lab"] = tmp["_lab"].fillna("").astype(str)
        tmp = tmp[tmp["_lab"] != ""]
        return tmp

    ev_pres = explode_presence("present_event_labels")
    if len(ev_pres) > 0:
        ev_count = ev_pres["_lab"].value_counts().reset_index()
        ev_count.columns = ["event_label", "n_tracks_with_label"]
        ev_count["ratio_of_tracks"] = ev_count["n_tracks_with_label"] / max(1, track_summary.shape[0])
        ev_count.to_csv(out_dir / "track_event_presence_counts.csv", index=False)
    else:
        pd.DataFrame(columns=["event_label", "n_tracks_with_label", "ratio_of_tracks"]).to_csv(
            out_dir / "track_event_presence_counts.csv", index=False
        )

    st_pres = explode_presence("present_state_labels")
    if len(st_pres) > 0:
        st_count = st_pres["_lab"].value_counts().reset_index()
        st_count.columns = ["state_label", "n_tracks_with_label"]
        st_count["ratio_of_tracks"] = st_count["n_tracks_with_label"] / max(1, track_summary.shape[0])
        st_count.to_csv(out_dir / "track_state_presence_counts.csv", index=False)
    else:
        pd.DataFrame(columns=["state_label", "n_tracks_with_label", "ratio_of_tracks"]).to_csv(
            out_dir / "track_state_presence_counts.csv", index=False
        )

    # Presence by class (optional)
    if has_class and "class" in track_summary.columns and len(ev_pres) > 0:
        ev_class = ev_pres.merge(track_summary[[*group_cols, "class"]], on=group_cols, how="left")
        ct = pd.crosstab(ev_class["class"].fillna("NA"), ev_class["_lab"].fillna("NA"))
        ct.to_csv(out_dir / "event_presence_by_class.csv")
    else:
        pd.DataFrame().to_csv(out_dir / "event_presence_by_class.csv", index=False)

    # Text summary
    summary_txt = out_dir / "summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Track-level Scenario Report\n")
        f.write(f"- input: {in_csv}\n")
        f.write(f"- n_tracks: {len(track_summary):,}\n")
        f.write(f"- has_class: {has_class}\n\n")

        # Simple headline stats
        if len(track_summary) > 0:
            f.write("[Track flags]\n")
            for k in ["has_merging", "has_diverging", "has_cut_in", "has_simple_lc", "has_any_lane_change"]:
                if k in track_summary.columns:
                    c = int(track_summary[k].sum())
                    f.write(f"- {k}: {c} tracks ({c/len(track_summary):.3f})\n")
            f.write("\n")

            f.write("[Outputs]\n")
            for p in sorted(out_dir.glob("*.csv")):
                f.write(f"- {p.name}\n")
            f.write(f"- {summary_txt.name}\n")

    print(f"[DONE] wrote track-level report to: {out_dir}")
    print(f" - {out_track}")
    print(f" - {summary_txt}")


if __name__ == "__main__":
    main()