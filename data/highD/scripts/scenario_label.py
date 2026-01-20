#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenario_label.py

Window-level scenario labeling for exiD and highD tracks files.

Project layout (default):
  ./raw/00_tracks.csv, 00_recordingMeta.csv, 00_tracksMeta.csv, ...     (exiD-like)
  ./raw/01_tracks.csv, 01_recordingMeta.csv, 01_tracksMeta.csv, ...     (highD-like)

Output:
  ./out/scenarios/window_labels.csv

Key features
------------
- Supports BOTH exiD-style and highD-style schemas.
- Auto-detects dataset schema per file unless --dataset is forced.
- recordingId mapping to avoid collisions:
    exiD recordingId 00-92   -> keep as-is
    highD recordingId 01-60  -> 101-160 by default (offset=100)
  (configurable via --highd_recording_offset)

Event labeling (priority):
  Merging > Diverging > CutIn > LaneChange(Simple/Other) > None

Notes about highD:
- highD has no ramp_type, so merging/diverging are always False.
- lane-change detection uses laneId transitions (highD definition) instead of a laneChange flag.
- lane-change direction uses yVelocity sign (per your preference):
    yVelocity > 0 => right, yVelocity < 0 => left

Implementation notes:
- For cut-in / simple-LC rules we normalize both datasets into a common set of columns:
    frame, recordingId, trackId,
    laneId, laneChange,
    latVelocity,
    leadId, rearId,
    leftLeadId, leftRearId, leftAlongsideId,
    rightLeadId, rightRearId, rightAlongsideId,
    ramp_type (optional)
- If some columns are missing, labeling degrades gracefully.

"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Literal

import numpy as np
import pandas as pd


Dataset = Literal["auto", "exid", "highd"]


# -------------------------
# IO
# -------------------------
def smart_read_csv(path: Path) -> pd.DataFrame:
    """Read comma or semicolon delimited CSVs."""
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, sep=";", low_memory=False)


def find_recordings(raw_dir: Path) -> List[str]:
    """Find 'XX_tracks.csv' under raw_dir and return unique XX strings."""
    xs = []
    for p in sorted(raw_dir.glob("*_tracks.csv")):
        xx = p.name.split("_")[0]
        if len(xx) == 2 and xx.isdigit():
            xs.append(xx)
    return sorted(set(xs))


# -------------------------
# Helpers: alongside parsing
# -------------------------
def parse_id_list_cell(x) -> List[int]:
    """
    exiD alongside fields can be:
      - "-1"
      - empty
      - "12;45;78"
    highD alongside fields are scalar int with 0 meaning none (but we accept anyway).

    Return list of ints excluding {0, -1}.
    """
    if pd.isna(x):
        return []
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return []
    # numeric scalar
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        v = int(s)
        return [] if v in (-1, 0) else [v]
    parts = [p.strip() for p in s.split(";") if p.strip() != ""]
    out: List[int] = []
    for p in parts:
        if p.lstrip("-").isdigit():
            v = int(p)
            if v not in (-1, 0):
                out.append(v)
    return out


def to_int_series(s: pd.Series, none_values: Tuple[int, ...] = (-1, 0)) -> pd.Series:
    """
    Convert to int series, mapping NaN -> none_values[0] and preserving ints.
    Many highD neighbor id fields use 0 for "none"; exiD uses -1.
    We keep values as-is, but callers can discard both -1 and 0.
    """
    out = pd.to_numeric(s, errors="coerce").fillna(none_values[0]).astype(int)
    return out


# -------------------------
# Schema normalization
# -------------------------
def detect_schema(tracks_cols: List[str], forced: Dataset = "auto") -> Dataset:
    cols = set([c.strip() for c in tracks_cols])
    if forced != "auto":
        return forced
    # exiD-like has recordingId in tracks and laneChange flag and/or latVelocity
    if "recordingId" in cols and ("laneChange" in cols or "latVelocity" in cols):
        return "exid"
    # highD has laneId + xVelocity/yVelocity and id (track id) but no recordingId column
    if "laneId" in cols and ("xVelocity" in cols or "yVelocity" in cols) and "id" in cols:
        return "highd"
    # fallback: if laneChange exists, assume exiD; else if laneId exists, assume highD
    if "laneChange" in cols:
        return "exid"
    if "laneId" in cols:
        return "highd"
    return "exid"


def map_recording_id(rid: int, dataset: Dataset, highd_offset: int) -> int:
    """
    Avoid recordingId collisions between datasets.

    - exiD: keep original (00-92)
    - highD: add offset (default +100): 01-60 -> 101-160
    """
    if dataset == "highd":
        return int(rid) + int(highd_offset)
    return int(rid)


def normalize_highd_tracks(
    tracks: pd.DataFrame,
    xx: str,
    highd_offset: int,
) -> pd.DataFrame:
    """
    Convert highD tracks schema to the common schema used by labeling logic.

    highD tracks columns (from format):
      frame, id, x, y, width, height, xVelocity, yVelocity, xAcceleration, yAcceleration,
      dhw, thw, ttc,
      precedingId, followingId,
      leftPrecedingId, leftAlongsideId, leftFollowingId,
      rightPrecedingId, rightAlongsideId/rightAlsongsideId, rightFollowingId,
      laneId, ...

    Common schema fields produced:
      recordingId, trackId, frame,
      laneId,
      latVelocity,
      leadId, rearId,
      leftLeadId, leftRearId, leftAlongsideId,
      rightLeadId, rightRearId, rightAlongsideId
    """
    df = tracks.copy()
    df.columns = [c.strip() for c in df.columns]

    # recordingId: not present in highD tracks.csv -> derive from filename XX
    rid_raw = int(xx)
    rid = map_recording_id(rid_raw, "highd", highd_offset)

    df["recordingId"] = rid

    # track id
    if "trackId" not in df.columns:
        if "id" in df.columns:
            df["trackId"] = df["id"]
        else:
            raise KeyError("highD tracks missing 'id' (track id)")

    # latVelocity: use yVelocity (per user preference)
    if "latVelocity" not in df.columns:
        if "yVelocity" in df.columns:
            df["latVelocity"] = df["yVelocity"]
        else:
            df["latVelocity"] = np.nan

    # laneChange flag: compute per-track via laneId transitions (later in label_window)
    if "laneChange" not in df.columns:
        df["laneChange"] = 0  # placeholder

    # lead/rear in same lane
    if "leadId" not in df.columns:
        if "precedingId" in df.columns:
            df["leadId"] = df["precedingId"]
        else:
            df["leadId"] = 0
    if "rearId" not in df.columns:
        if "followingId" in df.columns:
            df["rearId"] = df["followingId"]
        else:
            df["rearId"] = 0

    # side neighbors
    rename_map = {}
    for a, b in [
        ("leftPrecedingId", "leftLeadId"),
        ("leftFollowingId", "leftRearId"),
        ("leftAlongsideId", "leftAlongsideId"),
        ("rightPrecedingId", "rightLeadId"),
        ("rightFollowingId", "rightRearId"),
        ("rightAlongsideId", "rightAlongsideId"),
        ("rightAlsongsideId", "rightAlongsideId"),  # some files have this typo
    ]:
        if a in df.columns and b not in df.columns:
            rename_map[a] = b
    if rename_map:
        df = df.rename(columns=rename_map)

    # ensure all expected columns exist
    for c in ["leftLeadId", "leftRearId", "leftAlongsideId", "rightLeadId", "rightRearId", "rightAlongsideId"]:
        if c not in df.columns:
            df[c] = 0

    # ensure required columns exist
    for c in ["frame", "laneId", "trackId", "recordingId"]:
        if c not in df.columns:
            raise KeyError(f"highD normalization missing required column: {c}")

    return df


def normalize_exid_tracks(
    tracks: pd.DataFrame,
    highd_offset: int,
) -> pd.DataFrame:
    """
    Ensure exiD tracks schema matches the common schema used by labeling logic.
    (Mostly passthrough; just make sure required column names exist.)
    """
    df = tracks.copy()
    df.columns = [c.strip() for c in df.columns]

    # track id column: exiD uses trackId already
    if "trackId" not in df.columns and "id" in df.columns:
        df["trackId"] = df["id"]

    # ensure recordingId exists
    if "recordingId" not in df.columns:
        raise KeyError("exiD tracks missing recordingId")

    # mapping: exiD rid kept as-is
    df["recordingId"] = df["recordingId"].apply(lambda x: map_recording_id(int(x), "exid", highd_offset))

    # ensure latVelocity exists if possible
    if "latVelocity" not in df.columns:
        df["latVelocity"] = np.nan

    # ensure laneId exists if possible (not required for exiD)
    if "laneId" not in df.columns:
        df["laneId"] = np.nan

    # ensure leadId/rearId naming used in label_window logic
    if "leadId" not in df.columns and "leadId" in df.columns:
        pass

    return df


def normalize_recmeta(recmeta: pd.DataFrame, dataset: Dataset, xx: str, highd_offset: int) -> pd.DataFrame:
    """
    Normalize recordingMeta into columns: recordingId, frameRate
    - exiD: already recordingId, frameRate
    - highD: uses 'id' for recording id
    """
    rm = recmeta.copy()
    rm.columns = [c.strip() for c in rm.columns]

    if dataset == "highd":
        if "recordingId" not in rm.columns:
            if "id" in rm.columns:
                rm = rm.rename(columns={"id": "recordingId"})
        if "recordingId" not in rm.columns:
            # derive from filename as fallback
            rm["recordingId"] = int(xx)
        rm["recordingId"] = rm["recordingId"].apply(lambda x: map_recording_id(int(x), "highd", highd_offset))
    else:
        if "recordingId" not in rm.columns:
            raise KeyError("exiD recordingMeta missing recordingId")
        rm["recordingId"] = rm["recordingId"].apply(lambda x: map_recording_id(int(x), "exid", highd_offset))

    if "frameRate" not in rm.columns:
        raise KeyError("recordingMeta missing frameRate")

    return rm[["recordingId", "frameRate"]].drop_duplicates()


# -------------------------
# Cut-in detection (window-level)
# -------------------------
def build_lane_lookup(df_all: pd.DataFrame, schema: Dataset) -> Tuple[str, Dict[int, Dict[int, int]]]:
    """
    Build (lane_col, lookup) where lookup[trackId][frame] = lane_id (laneletId for exiD, laneId for highD).
    """
    lane_col = "laneletId" if (schema == "exid" and "laneletId" in df_all.columns) else "laneId"
    if lane_col not in df_all.columns:
        raise ValueError(f"Missing {lane_col} in tracks file for schema={schema}")

    tmp = df_all[["trackId", "frame", lane_col]].copy()
    tmp["trackId"] = pd.to_numeric(tmp["trackId"], errors="coerce").astype(int)
    tmp["frame"] = pd.to_numeric(tmp["frame"], errors="coerce").astype(int)
    tmp[lane_col] = pd.to_numeric(tmp[lane_col], errors="coerce")

    lookup: Dict[int, Dict[int, int]] = {}
    for tid, g in tmp.groupby("trackId", sort=False):
        fr = g["frame"].to_numpy()
        ln = g[lane_col].to_numpy()
        d: Dict[int, int] = {}
        for f, l in zip(fr, ln):
            if np.isnan(l):
                continue
            d[int(f)] = int(l)
        lookup[int(tid)] = d
    return lane_col, lookup


def load_lanelet_adjacency(pkl_path: Path) -> Dict[int, Dict[int, set]]:
    """
    Load lanelet adjacency DB (exiD only). Expected structure:
      db = {"adj_by_map": {map_id: {lanelet_id: set(adj_lanelet_ids)}}}
    """
    with open(pkl_path, "rb") as f:
        db = pickle.load(f)
    return db.get("adj_by_map", {})


def is_adjacent_lane(schema: Dataset, map_id: Optional[int], ego_lane: int, nb_lane: int, adj_by_map: Optional[Dict]) -> bool:
    if schema == "highd":
        return abs(int(ego_lane) - int(nb_lane)) == 1
    if adj_by_map is None or map_id is None:
        return True
    return int(nb_lane) in adj_by_map.get(int(map_id), {}).get(int(ego_lane), set())


def _majority_ramp_type(s: pd.Series) -> Optional[str]:
    """Return majority ramp_type in series among {'none','onramp','offramp'}, else None."""
    if s is None or len(s) == 0:
        return None
    v = s.astype(str).value_counts(dropna=True)
    if v.empty:
        return None
    # Normalize unexpected labels to string
    top = v.index[0]
    return str(top)


def detect_merging_diverging_exid(w: pd.DataFrame, lc_frame: int, W: int = 10) -> Optional[str]:
    """
    exiD-only: detect merging/diverging around a lane change using ramp_type transition.
      - merging: pre majority 'onramp' and aft majority 'none'
      - diverging: pre majority 'none' and aft majority 'offramp'
    Returns 'merging'/'diverging' or None if not applicable/insufficient data.
    """
    if "ramp_type" not in w.columns or lc_frame is None:
        return None
    # Use frames strictly before/after the lane-change frame
    pre = w[w["frame"] < lc_frame].tail(W)
    aft = w[w["frame"] > lc_frame].head(W)
    if len(pre) == 0 or len(aft) == 0:
        return None
    pre_rt = _majority_ramp_type(pre["ramp_type"])
    aft_rt = _majority_ramp_type(aft["ramp_type"])
    if pre_rt is None or aft_rt is None:
        return None
    if pre_rt == "onramp" and aft_rt == "none":
        return "merging"
    if pre_rt == "none" and aft_rt == "offramp":
        return "diverging"
    return None


def get_lane_at(lane_lookup: Dict[int, Dict[int, int]], track_id: int, frame: int) -> Optional[int]:
    d = lane_lookup.get(int(track_id))
    if not d:
        return None
    return d.get(int(frame))


def target_side_cols(direction: str) -> Tuple[str, str, str]:
    if direction == "left":
        return ("leftLeadId", "leftAlongsideId", "leftRearId")
    return ("rightLeadId", "rightAlongsideId", "rightRearId")


def adjacent_presence_in_pre_window(
    w_ego: pd.DataFrame,
    lc_frame: int,
    direction: str,
    schema: Dataset,
    lane_col: str,
    lane_lookup: Dict[int, Dict[int, int]],
    map_id: Optional[int],
    adj_by_map: Optional[Dict],
    W: int = 10,
) -> Tuple[bool, bool, bool]:
    """
    Look at frames [lc_frame-W, lc_frame-1]. Consider only neighbors that are in the IMMEDIATELY adjacent lane.

    Returns:
      any_adjacent_vehicle (lead/alongside/rear)
      any_adjacent_rear_or_alongside
      valid_eval: whether we had enough information (at least one non-NaN ego lane)
                  to judge adjacency. If False, callers should treat the case as
                  ambiguous (e.g., lane_change_other) rather than "simple".
    """
    if "frame" not in w_ego.columns:
        return (False, False, False)

    df = w_ego.copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype(int)
    pre = df[(df["frame"] >= int(lc_frame) - W) & (df["frame"] <= int(lc_frame) - 1)]
    if len(pre) == 0:
        return (False, False, False)

    lead_col, alongside_col, rear_col = target_side_cols(direction)

    any_adj = False
    any_adj_rear_or_along = False
    valid_seen = False

    for _, row in pre.iterrows():
        f = int(row["frame"])
        ego_lane = row.get(lane_col, np.nan)
        if pd.isna(ego_lane):
            continue
        ego_lane = int(ego_lane)
        valid_seen = True

        for col in (lead_col, alongside_col, rear_col):
            if col not in pre.columns:
                continue
            nb_id = row.get(col, -1)
            try:
                nb_id = int(nb_id)
            except Exception:
                continue
            if nb_id == -1:
                continue
            nb_lane = get_lane_at(lane_lookup, nb_id, f)
            if nb_lane is None:
                continue
            if not is_adjacent_lane(schema, map_id, ego_lane, nb_lane, adj_by_map):
                continue
            any_adj = True
            if col in (alongside_col, rear_col):
                any_adj_rear_or_along = True

    if not valid_seen:
        return (False, False, False)
    return (any_adj, any_adj_rear_or_along, True)
def infer_lc_direction(w: pd.DataFrame, lc_frame: int, schema: Dataset, K: int = 5) -> Optional[str]:
    """
    Infer lane-change direction around lc_frame.

    exiD: use latVelocity (positive=right, negative=left)
    highD: use yVelocity (user rule: positive=right, negative=left)

    Returns: "left" / "right" / None
    """
    if "frame" not in w.columns:
        return None
    df = w.copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype(int)

    if schema == "exid":
        vcol = "latVelocity"
    else:
        vcol = "yVelocity"

    if vcol not in df.columns:
        return None

    win = df[(df["frame"] >= int(lc_frame) - K) & (df["frame"] <= int(lc_frame) + K)]
    v = pd.to_numeric(win[vcol], errors="coerce")
    if v.notna().sum() == 0:
        return None
    mv = float(v.mean())
    if mv > 0:
        return "right"
    if mv < 0:
        return "left"
    return None
def has_alongside_ids(series: pd.Series) -> bool:
    for x in series.tolist():
        if len(parse_id_list_cell(x)) > 0:
            return True
    return False


def is_simple_lane_change_directional(track_df: pd.DataFrame, lc_frame: int, direction: str, W: int = 10) -> bool:
    """
    Simple LC (directional):
      In the BEFORE window [lc_frame-W, lc_frame-1],
      - no alongside on the target side
      - no rear/following on the target side (targetRearId == -1/0)

    direction: "left" or "right"
    """
    if "frame" not in track_df.columns:
        return False

    df = track_df.copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype(int)
    pre = df[(df["frame"] >= lc_frame - W) & (df["frame"] <= lc_frame - 1)]
    if len(pre) == 0:
        return False

    if direction == "left":
        rear_col = "leftRearId"
        alongside_col = "leftAlongsideId"
    elif direction == "right":
        rear_col = "rightRearId"
        alongside_col = "rightAlongsideId"
    else:
        return False

    if alongside_col in pre.columns:
        if has_alongside_ids(pre[alongside_col]):
            return False
    else:
        return False

    if rear_col in pre.columns:
        rear_ids = set(to_int_series(pre[rear_col]).tolist())
        rear_ids.discard(-1)
        rear_ids.discard(0)
        if len(rear_ids) > 0:
            return False
    else:
        return False

    return True


# -------------------------
# Window labeling rules
# -------------------------
def _detect_lane_changes_in_window_highd(w: pd.DataFrame) -> Tuple[bool, int, Optional[int]]:
    """
    For highD: lane-change detection uses laneId transitions.
    Returns: (has_lanechange, laneChange_count, first_lc_frame)
    """
    if "laneId" not in w.columns or "frame" not in w.columns:
        return (False, 0, None)

    df = w.copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype(int)
    df = df.sort_values("frame")
    lane = pd.to_numeric(df["laneId"], errors="coerce")

    if lane.isna().all():
        return (False, 0, None)

    lane_filled = lane.ffill().bfill()
    changes = (lane_filled != lane_filled.shift(1)).fillna(False)
    # first row shift introduces change; remove it
    changes.iloc[0] = False
    count = int(changes.sum())
    if count == 0:
        return (False, 0, None)

    first_frame = int(df.loc[changes, "frame"].iloc[0])
    return (True, count, first_frame)


def label_window(
    w: pd.DataFrame,
    frameRate: float,
    schema: Dataset,
    lane_col: str,
    lane_lookup: Dict[int, Dict[int, int]],
    map_id: Optional[int] = None,
    adj_by_map: Optional[Dict] = None,
    W_adj: int = 10,
    dense_thw: float = 1.5,
    free_thw: float = 3.0,
    dense_dhw: float = 20.0,
    free_no_lead_ratio: float = 0.7,
    dense_ratio_th: float = 0.3,
    free_ratio_th: float = 0.7,
) -> Dict:
    """
    Event labeling (merging/diverging excluded):

      If lane change occurs in the window:
        - direction = inferred from velocity (exiD: latVelocity, highD: yVelocity)
        - Look at BEFORE window [lc_frame-W_adj, lc_frame-1]
        - Consider ONLY vehicles that are in the IMMEDIATELY adjacent lane:
            * exiD: lanelet adjacency via (map_id, laneletId)
            * highD: abs(laneId diff) == 1

        Rules (target side = left/right):
          1) If NO adjacent {targetRear, targetAlongside} -> simple_lane_change
             (NOTE: targetLead presence in the adjacent lane does NOT prevent "simple".)
          2) Else if any adjacent {targetRear, targetAlongside} exists -> cut_in
          3) Else (cannot evaluate adjacency / direction unknown) -> lane_change_other

      If no lane change: event_label = none

    State labeling: dense / free_flow / car_following / ramp_driving / other
    """
    out: Dict = {}

    n = len(w)
    out["n_frames_in_window"] = int(n)
    if n == 0:
        out["event_label"] = "none"
        out["state_label"] = "other"
        return out

    has_lanechange = False
    laneChange_count = 0
    lc_frame: Optional[int] = None

    if schema == "highd":
        has_lanechange, laneChange_count, lc_frame = _detect_lane_changes_in_window_highd(w)
    else:
        if "laneChange" in w.columns:
            lc = pd.to_numeric(w["laneChange"], errors="coerce").fillna(0).astype(int)
            laneChange_count = int((lc == 1).sum())
            has_lanechange = laneChange_count > 0
            if has_lanechange and "frame" in w.columns:
                f = pd.to_numeric(w.loc[lc == 1, "frame"], errors="coerce").dropna().astype(int)
                if len(f) > 0:
                    lc_frame = int(f.iloc[0])

    out["has_laneChange"] = bool(has_lanechange)
    out["laneChange_count_in_window"] = int(laneChange_count)
    out["lc_frame"] = int(lc_frame) if lc_frame is not None else -1

    dense_ratio = np.nan
    free_ratio = np.nan
    if "thw" in w.columns:
        thw = pd.to_numeric(w["thw"], errors="coerce")
        dense_ratio = float((thw <= dense_thw).mean()) if thw.notna().sum() > 0 else np.nan
        free_ratio = float((thw >= free_thw).mean()) if thw.notna().sum() > 0 else np.nan
    elif "dhw" in w.columns:
        dhw = pd.to_numeric(w["dhw"], errors="coerce")
        dense_ratio = float((dhw <= dense_dhw).mean()) if dhw.notna().sum() > 0 else np.nan
        if "leadId" in w.columns:
            lead = to_int_series(w["leadId"])
            free_ratio = float((lead == -1).mean())
    else:
        if "leadId" in w.columns:
            lead = to_int_series(w["leadId"])
            free_ratio = float((lead == -1).mean())

    out["dense_ratio"] = dense_ratio
    out["free_ratio"] = free_ratio

    if "ramp_type" in w.columns:
        rt = w["ramp_type"].astype(str)
        out["onramp_ratio"] = float((rt == "onramp").mean())
        out["offramp_ratio"] = float((rt == "offramp").mean())
        out["mainlane_ratio"] = float((rt == "none").mean())
    else:
        out["onramp_ratio"] = np.nan
        out["offramp_ratio"] = np.nan
        out["mainlane_ratio"] = np.nan

    event_label = "none"
    lc_direction = None

    if has_lanechange and lc_frame is not None:
        lc_direction = infer_lc_direction(w, lc_frame=lc_frame, schema=schema, K=5)
        out["lc_direction"] = lc_direction if lc_direction is not None else "unknown"
        if lc_direction is None:
            event_label = "lane_change_other"
        else:
            # exiD: (re-)enable merging/diverging based on ramp_type transition around the lane change
            if schema == "exid":
                md = detect_merging_diverging_exid(w, lc_frame=lc_frame, W=W_adj)
            else:
                md = None
            if md is not None:
                event_label = md
                out["target_adj_any"] = False
                out["target_adj_rear_or_alongside"] = False
                out["target_adj_valid"] = True
            else:
                any_adj, any_adj_rear_along, adj_valid = adjacent_presence_in_pre_window(
                w_ego=w,
                lc_frame=lc_frame,
                direction=lc_direction,
                schema=schema,
                lane_col=lane_col,
                lane_lookup=lane_lookup,
                map_id=map_id,
                adj_by_map=adj_by_map,
                W=W_adj,
            )
                out["target_adj_any"] = bool(any_adj)
                out["target_adj_rear_or_alongside"] = bool(any_adj_rear_along)
                out["target_adj_valid"] = bool(adj_valid)

                # User rule:
                # - simple_lane_change: no adjacent REAR/ALONGSIDE on target side (lead presence is ignored)
                # - cut_in: adjacent rear OR alongside exists on target side
                # - lc_other: ambiguous (e.g., cannot evaluate adjacency)
                if not adj_valid:
                    event_label = "lane_change_other"
                elif any_adj_rear_along:
                    event_label = "cut_in"
                else:
                    event_label = "simple_lane_change"

    out["event_label"] = event_label
    out["has_cutin"] = bool(event_label == "cut_in")

    if ("ramp_type" in w.columns) and (event_label == "none"):
        rt = w["ramp_type"].astype(str)
        if ((rt == "onramp").mean() >= 0.5) or ((rt == "offramp").mean() >= 0.5):
            out["state_label"] = "ramp_driving"
            return out

    if isinstance(dense_ratio, float) and not np.isnan(dense_ratio) and dense_ratio >= dense_ratio_th:
        state = "dense"
    elif isinstance(free_ratio, float) and not np.isnan(free_ratio) and free_ratio >= free_ratio_th:
        state = "free_flow"
    else:
        if "leadId" in w.columns:
            leadId = to_int_series(w["leadId"])
            state = "car_following" if (leadId != -1).mean() >= (1.0 - free_no_lead_ratio) else "other"
        else:
            state = "other"

    out["state_label"] = state
    return out
def iter_windows_for_track(
    track_df: pd.DataFrame,
    schema: Dataset,
    lane_col: str,
    lane_lookup: Dict[int, Dict[int, int]],
    map_id: Optional[int],
    adj_by_map: Optional[Dict],
    history_sec: float,
    future_sec: float,
    stride_sec: float,
    frameRate: float,
) -> List[Dict]:
    if "frame" not in track_df.columns:
        return []

    df = track_df.copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype(int)
    df = df.sort_values("frame")

    hist_frames = int(round(history_sec * frameRate))
    fut_frames = int(round(future_sec * frameRate))
    stride_frames = int(round(stride_sec * frameRate))
    win_len = hist_frames + fut_frames

    frames = df["frame"].to_numpy()
    if len(frames) == 0:
        return []

    f0 = int(frames.min())
    f1 = int(frames.max())

    rows: List[Dict] = []
    for t0 in range(f0, f1 - win_len + 1, stride_frames):
        t1 = t0 + win_len - 1
        w = df[(df["frame"] >= t0) & (df["frame"] <= t1)]
        if len(w) == 0:
            continue

        out = label_window(
            w=w,
            frameRate=frameRate,
            schema=schema,
            lane_col=lane_col,
            lane_lookup=lane_lookup,
            map_id=map_id,
            adj_by_map=adj_by_map,
        )

        out["recordingId"] = int(pd.to_numeric(w["recordingId"], errors="coerce").dropna().iloc[0]) if "recordingId" in w.columns else -1
        out["trackId"] = int(pd.to_numeric(w["trackId"], errors="coerce").dropna().iloc[0]) if "trackId" in w.columns else -1
        out["t0_frame"] = int(t0)
        out["t1_frame"] = int(t1)
        out["t_mid_frame"] = int((t0 + t1) // 2)
        out["frameRate"] = float(frameRate)
        rows.append(out)

    return rows
def main():
    import argparse

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default=str(project_root / "raw"))
    ap.add_argument("--out_dir", type=str, default=str(project_root / "out" / "scenarios"))
    ap.add_argument("--out_csv", type=str, default="window_labels.csv",
                    help="Output CSV filename inside out_dir (e.g., exid_window_labels.csv).")
    ap.add_argument("--adj_pkl", type=str, default=str(project_root / "maps" / "lanelet_adj_allmaps.pkl"),
                    help="Path to lanelet adjacency pickle (exiD only). If missing/unreadable, adjacency filtering falls back to treating all neighbors as adjacent.")
    ap.add_argument("--dataset", type=str, choices=["auto", "exid", "highd"], default="auto",
                    help="Force schema for all recordings, or auto-detect per file.")
    ap.add_argument("--highd_recording_offset", type=int, default=100,
                    help="Offset added to highD recording ids (01-60 -> 101-160 by default).")
    ap.add_argument("--history_sec", type=float, default=2.0)
    ap.add_argument("--future_sec", type=float, default=5.0)
    ap.add_argument("--stride_sec", type=float, default=1.0)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load exiD lanelet adjacency DB (optional; used only to ignore non-adjacent left/right neighbors)
    adj_by_map = None
    try:
        adj_pkl = Path(args.adj_pkl)
        if adj_pkl.exists():
            adj_by_map = load_lanelet_adjacency(adj_pkl)
        else:
            adj_by_map = None
    except Exception:
        adj_by_map = None


    history_sec = float(args.history_sec)
    future_sec = float(args.future_sec)
    stride_sec = float(args.stride_sec)

    xxs = find_recordings(raw_dir)
    if not xxs:
        raise RuntimeError(f"No *_tracks.csv found under {raw_dir}")

    all_rows: List[Dict] = []

    for xx in xxs:
        tracks_path = raw_dir / f"{xx}_tracks.csv"
        recmeta_path = raw_dir / f"{xx}_recordingMeta.csv"

        tracks = smart_read_csv(tracks_path)
        recmeta = smart_read_csv(recmeta_path)

        tracks.columns = [c.strip() for c in tracks.columns]
        recmeta.columns = [c.strip() for c in recmeta.columns]

        schema = detect_schema(list(tracks.columns), forced=args.dataset)

        # Normalize
        if schema == "highd":
            tracks_n = normalize_highd_tracks(tracks, xx=xx, highd_offset=args.highd_recording_offset)
        else:
            tracks_n = normalize_exid_tracks(tracks, highd_offset=args.highd_recording_offset)

        recmeta_n = normalize_recmeta(recmeta, dataset=schema, xx=xx, highd_offset=args.highd_recording_offset)

        # map_id for lanelet adjacency (exiD): use locationId from recordingMeta
        map_id = None
        if 'locationId' in recmeta_n.columns:
            try:
                map_id = int(pd.to_numeric(recmeta_n['locationId'], errors='coerce').dropna().iloc[0])
            except Exception:
                map_id = None

        # Build lane lookup for adjacency checks
        lane_col, lane_lookup = build_lane_lookup(tracks_n, schema=schema)


        # frameRate join
        tracks_n = tracks_n.merge(recmeta_n, on="recordingId", how="left")

        # required columns
        for c in ["recordingId", "trackId", "frame"]:
            if c not in tracks_n.columns:
                raise KeyError(f"{tracks_path.name} missing required column after normalization: {c}")

        # numeric conversion
        tracks_n["frame"] = pd.to_numeric(tracks_n["frame"], errors="coerce")
        tracks_n = tracks_n.dropna(subset=["frame"])
        tracks_n["frame"] = tracks_n["frame"].astype(int)

        tracks_n = tracks_n.sort_values(["recordingId", "trackId", "frame"])

        print(f"[PROCESS] {xx} schema={schema} rows={len(tracks_n):,}")

        # group by track
        for (rid, tid), g in tracks_n.groupby(["recordingId", "trackId"], sort=False):
            fr = g["frameRate"].iloc[0]
            if pd.isna(fr) or fr <= 0:
                continue
            fr = float(fr)

            labeled_rows = iter_windows_for_track(
                track_df=g,
                schema=schema,
                lane_col=lane_col,
                lane_lookup=lane_lookup,
                map_id=map_id,
                adj_by_map=adj_by_map,
                history_sec=history_sec,
                future_sec=future_sec,
                stride_sec=stride_sec,
                frameRate=fr,
            )
            if not labeled_rows:
                continue

            for row in labeled_rows:
                row["history_sec"] = history_sec
                row["future_sec"] = future_sec
                row["stride_sec"] = stride_sec
                row["schema"] = schema
                all_rows.append(row)

    out_df = pd.DataFrame(all_rows)
    out_csv = out_dir / args.out_csv
    out_df.to_csv(out_csv, index=False)
    print(f"\n[DONE] wrote {len(out_df):,} rows to: {out_csv}\n")

    if "event_label" in out_df.columns:
        print("Event label counts:")
        print(out_df["event_label"].value_counts())
    if "state_label" in out_df.columns:
        print("\nState label counts:")
        print(out_df["state_label"].value_counts())


if __name__ == "__main__":
    main()