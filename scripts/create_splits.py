import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# =========================================================
# 설정 (User Configuration)
# =========================================================

LABEL_FILES = [
    {
        "name": "exid", 
        "path": "./data/exiD/data_pt/exid_T2_Tf5_hz3/window_labels.csv"
    },
    {
        "name": "highd", 
        "path": "./data/highD/data_pt/highd_T2_Tf5_hz3/window_labels.csv"
    }
]

OUTPUT_DIR = "./data/combined/splits"

# Split 비율
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
    dfs = []
    current_offset = 0
    
    print(f"[INFO] Loading and merging label files...")
    
    for item in LABEL_FILES:
        path = Path(item["path"])
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        df = pd.read_csv(path)
        print(f"  - Loaded {item['name']}: {len(df):,} samples")
        
        df["global_index"] = df.index + current_offset
        df["dataset_source"] = item["name"]
        
        dfs.append(df)
        current_offset += len(df) # 다음 데이터셋을 위해 오프셋 증가
        
    # 전체 데이터 병합
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Total merged samples: {len(full_df):,}")
    
    
    # 1. Event Label 매핑
    full_df["mapped_event"] = full_df["event_label"].apply(map_event_label)
    
    print("\n--- Combined Label Distribution ---")
    print(full_df["mapped_event"].value_counts())
    
    # 2. Track 단위 그룹화
    print("\n[INFO] Grouping by Track...")
    track_groups = full_df.groupby(["dataset_source", "recordingId", "trackId"])
    
    track_infos = []
    for (src, rid, tid), group in track_groups:
        rep_label = get_track_representative_label(group)
        indices = group["global_index"].tolist() 
        
        track_infos.append({
            "label": rep_label,
            "indices": indices
        })
        
    track_df = pd.DataFrame(track_infos)
    print(f"[INFO] Total Unique Tracks: {len(track_df):,}")

    # 3. Stratified Split
    print(f"\n[INFO] Splitting...")
    train_tracks, temp_tracks = train_test_split(
        track_df, test_size=(1.0 - TRAIN_RATIO), stratify=track_df["label"], random_state=SEED
    )
    
    val_ratio_adj = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_tracks, test_tracks = train_test_split(
        temp_tracks, test_size=(1.0 - val_ratio_adj), stratify=temp_tracks["label"], random_state=SEED
    )
    
    # 4. 인덱스 추출 (Flatten)
    train_indices = np.array(sorted([idx for t in train_tracks["indices"] for idx in t]))
    val_indices = np.array(sorted([idx for t in val_tracks["indices"] for idx in t]))
    test_indices = np.array(sorted([idx for t in test_tracks["indices"] for idx in t]))
    
    # 5. 저장
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    np.save(f"{OUTPUT_DIR}/train_indices.npy", train_indices)
    np.save(f"{OUTPUT_DIR}/val_indices.npy", val_indices)
    np.save(f"{OUTPUT_DIR}/test_indices.npy", test_indices)
    
    print(f"\n✅ Done! Saved splits to {OUTPUT_DIR}")
    print(f"   Train: {len(train_indices):,}")
    print(f"   Val  : {len(val_indices):,}")
    print(f"   Test : {len(test_indices):,}")

if __name__ == "__main__":
    main()