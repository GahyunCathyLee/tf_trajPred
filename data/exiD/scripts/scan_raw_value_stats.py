#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

TRACKS_DIR = Path("raw")   # 수정 가능
GLOB = "*_tracks.csv"

def scan_one(csv_path: Path):
    df = pd.read_csv(csv_path, low_memory=False)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    rows = []
    for col in numeric_cols:
        s = df[col]
        rows.append({
            "file": csv_path.name,
            "column": col,
            "count": len(s),
            "nan": int(s.isna().sum()),
            "posinf": int(np.isposinf(s).sum()),
            "neginf": int(np.isneginf(s).sum()),
            "minus_one": int((s == -1).sum()),
            "zero": int((s == 0).sum()),
            "min": s.min(skipna=True),
            "max": s.max(skipna=True),
        })
    return rows

def main():
    all_rows = []
    for csv in sorted(TRACKS_DIR.glob(GLOB)):
        print(f"[SCAN] {csv.name}")
        all_rows.extend(scan_one(csv))

    out = pd.DataFrame(all_rows)
    out.to_csv("raw_value_stats.csv", index=False)
    print("[DONE] saved raw_value_stats.csv")

if __name__ == "__main__":
    main()