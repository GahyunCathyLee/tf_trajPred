from pathlib import Path
import pandas as pd

def load_window_labels_csv(path: Path):
    """
    window_labels.csv:
      recordingId, trackId, t0_frame, ... , event_label, state_label
    -> dict[(recordingId,trackId,t0_frame)] = {"event_label":..., "state_label":...}
    """
    if not path.exists():
        return None

    df = pd.read_csv(path)
    required = {"recordingId", "trackId", "t0_frame"}
    if not required.issubset(set(df.columns)):
        print(f"[WARN] window_labels.csv missing keys: {required - set(df.columns)} -> stratified eval disabled")
        return None

    if "event_label" not in df.columns and "state_label" not in df.columns:
        print("[WARN] window_labels.csv has no event_label/state_label -> stratified eval disabled")
        return None

    cols = ["recordingId", "trackId", "t0_frame"]
    if "event_label" in df.columns:
        cols.append("event_label")
    if "state_label" in df.columns:
        cols.append("state_label")
    df = df[cols]

    lut = {}
    for r in df.itertuples(index=False):
        rid = int(getattr(r, "recordingId"))
        tid = int(getattr(r, "trackId"))
        t0  = int(getattr(r, "t0_frame"))
        lut[(rid, tid, t0)] = {
            "event_label": getattr(r, "event_label", None),
            "state_label": getattr(r, "state_label", None),
        }
    print(f"[INFO] Loaded window labels: {len(lut):,} from {path}")
    return lut