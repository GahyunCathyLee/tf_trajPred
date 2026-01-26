import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# =========================================================
# 설정
# =========================================================
LABEL_FILE_MAP = {
    "exid": "./data/exiD/data_pt/exid_T2_Tf5_hz3/window_labels.csv" ,
    "highd": "./data/highD/data_pt/highd_T2_Tf5_hz3/window_labels.csv"
}

OUTPUT_DIR_MAP = {
    "exid": "./data/exiD/splits",
    "highd": "./data/highD/splits",
    "combined": "./data/combined/splits"
}

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42

def map_event_label(raw_label):
    if pd.isna(raw_label): return "Lane Following"
    raw = str(raw_label).lower()
    if "cut_in" in raw: return "Cut-in"
    elif "merging" in raw: return "Merging"
    elif "diverging" in raw: return "Diverging"
    elif "lane_change" in raw: return "Lane Change"
    else: return "Lane Following"

def get_track_representative_label(group):
    unique_events = set(group["mapped_event"].unique())
    if "Cut-in" in unique_events: return "Cut-in"
    elif "Merging" in unique_events: return "Merging"
    elif "Diverging" in unique_events: return "Diverging"
    elif "Lane Change" in unique_events: return "Lane Change"
    else: return "Lane Following"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["exid", "highd", "combined"],
                        help="Which dataset mode to generate splits for.")
    args = parser.parse_args()
    
    # 1. Target Datasets 결정
    targets = []
    if args.mode == "combined":
        targets = ["exid", "highd"] # 순서 중요 (train.py ConcatDataset 순서와 일치)
    else:
        targets = [args.mode]

    print(f"[INFO] Generating splits for MODE: {args.mode}")
    print(f"       Target Datasets: {targets}")

    # 2. 데이터 로드 및 병합 (Index Offset 처리)
    dfs = []
    current_offset = 0
    
    for name in targets:
        path = Path(LABEL_FILE_MAP[name])
        if not path.exists():
            raise FileNotFoundError(f"Label file not found: {path}")
            
        df = pd.read_csv(path)
        print(f"  - Loaded {name}: {len(df):,} samples")
        
        # ConcatDataset 인덱스 정렬을 위한 Offset 적용
        df["global_index"] = df.index + current_offset
        df["dataset_source"] = name
        
        dfs.append(df)
        current_offset += len(df)
        
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Total merged samples: {len(full_df):,}")
    
    # 3. Label Mapping & Grouping
    full_df["mapped_event"] = full_df["event_label"].apply(map_event_label)
    
    print("\n[INFO] Grouping by Track...")
    track_groups = full_df.groupby(["dataset_source", "recordingId", "trackId"])
    
    track_infos = []
    for (src, rid, tid), group in track_groups:
        rep_label = get_track_representative_label(group)
        indices = group["global_index"].tolist()
        track_infos.append({"label": rep_label, "indices": indices})
        
    track_df = pd.DataFrame(track_infos)
    print(f"[INFO] Unique Tracks: {len(track_df):,}")
    print(track_df["label"].value_counts())

    # 4. Stratified Split
    print(f"\n[INFO] Splitting...")
    train_tracks, temp_tracks = train_test_split(
        track_df, test_size=(1.0 - TRAIN_RATIO), stratify=track_df["label"], random_state=SEED
    )
    val_ratio_adj = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_tracks, test_tracks = train_test_split(
        temp_tracks, test_size=(1.0 - val_ratio_adj), stratify=temp_tracks["label"], random_state=SEED
    )
    
    train_indices = np.array(sorted([idx for t in train_tracks["indices"] for idx in t]))
    val_indices = np.array(sorted([idx for t in val_tracks["indices"] for idx in t]))
    test_indices = np.array(sorted([idx for t in test_tracks["indices"] for idx in t]))
    
    # 5. 저장 
    out_dir = Path(OUTPUT_DIR_MAP[args.mode])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / "train_indices.npy", train_indices)
    np.save(out_dir / "val_indices.npy", val_indices)
    np.save(out_dir / "test_indices.npy", test_indices)
    
    print(f"\n✅ Saved splits to: {out_dir.resolve()}")
    print(f"   Train: {len(train_indices):,}")
    print(f"   Val  : {len(val_indices):,}")
    print(f"   Test : {len(test_indices):,}")

if __name__ == "__main__":
    main()