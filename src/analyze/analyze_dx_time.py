import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# --- 설정 ---
HIGHD_NEIGHBORS = [
    "precedingId", "followingId", 
    "leftPrecedingId", "leftAlongsideId", "leftFollowingId",
    "rightPrecedingId", "rightAlongsideId", "rightFollowingId"
]
EXID_NEIGHBORS = [
    "leadId", "rearId",
    "leftLeadId", "leftAlongsideId", "leftRearId",
    "rightLeadId", "rightAlongsideId", "rightRearId"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze dx_time for Gate Decision")
    parser.add_argument("--highd_dir", type=str, default="data/highD/raw")
    parser.add_argument("--exid_dir", type=str, default="data/exiD/raw")
    parser.add_argument("--sample_rate", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    return parser.parse_args()

def process_highd_file(file_path):
    try:
        df = pd.read_csv(file_path)
        meta_path = str(file_path).replace("_tracks.csv", "_tracksMeta.csv")
        if Path(meta_path).exists():
            meta_df = pd.read_csv(meta_path)
            if "drivingDirection" in meta_df.columns:
                dir_map = dict(zip(meta_df["id"], meta_df["drivingDirection"]))
                df["drivingDirection"] = df["id"].map(dir_map).fillna(1)
            else:
                df["drivingDirection"] = 1
        else:
            df["drivingDirection"] = 1

        mask = df["drivingDirection"] == 1
        x_max = df["x"].max()
        x = df["x"].values.copy()
        vx = df["xVelocity"].values.copy()
        x[mask] = x_max - x[mask]
        vx[mask] = -vx[mask]
        df["x_norm"] = x
        df["vx_norm"] = vx
        return df, HIGHD_NEIGHBORS
    except Exception as e:
        return None, None

def process_exid_file(file_path):
    try:
        df = pd.read_csv(file_path, low_memory=False)
        if "trackId" in df.columns:
            df.rename(columns={"trackId": "id"}, inplace=True)
        for col in EXID_NEIGHBORS:
            if col in df.columns:
                df[col] = df[col].astype(str).str.split(';').str[0]
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(int)
        df["x_norm"] = df["xCenter"]
        df["vx_norm"] = df["lonVelocity"]
        return df, EXID_NEIGHBORS
    except Exception as e:
        return None, None

def analyze_dataset(files, dataset_name, process_func, args):
    dx_times = []
    
    if args.sample_rate < 1.0:
        import random
        random.seed(42)
        k = int(len(files) * args.sample_rate)
        if k > 0: files = random.sample(files, k)
    
    print(f"Processing {len(files)} files for {dataset_name}...")

    for f in tqdm(files):
        df, nb_cols = process_func(f)
        if df is None or "id" not in df.columns: continue

        # Indexing for speed
        df_indexed = df.set_index(["frame", "id"])[["x_norm", "vx_norm"]]
        
        for col in nb_cols:
            if col not in df.columns: continue
            valid_mask = df[col] > 0
            ego_df = df[valid_mask].copy()
            if ego_df.empty: continue

            ego_x = ego_df["x_norm"].values
            ego_vx = ego_df["vx_norm"].values
            ego_frames = ego_df["frame"].values
            nb_ids = ego_df[col].values
            
            nb_keys = list(zip(ego_frames, nb_ids))
            
            try:
                nb_data = df_indexed.reindex(nb_keys)
                valid_match = nb_data["x_norm"].notna().values
                if not np.any(valid_match): continue

                x_nb = nb_data["x_norm"].values[valid_match]
                vx_nb = nb_data["vx_norm"].values[valid_match]
                x_ego = ego_x[valid_match]
                vx_ego = ego_vx[valid_match]
                
                # Calculation
                dx = x_nb - x_ego
                dv = vx_nb - vx_ego
                denom = dv + np.sign(dv + 1e-9) * args.epsilon 
                val = dx / denom
                
                dx_times.extend(val.tolist())
            except: continue

    return np.array(dx_times)

def main():
    args = parse_args()
    all_dx_times = []
    
    if args.highd_dir:
        files = sorted(glob.glob(f"{args.highd_dir}/*_tracks.csv"))
        if files: all_dx_times.append(analyze_dataset(files, "HighD", process_highd_file, args))
            
    if args.exid_dir:
        files = sorted(glob.glob(f"{args.exid_dir}/*_tracks.csv"))
        if files: all_dx_times.append(analyze_dataset(files, "ExiD", process_exid_file, args))

    if not all_dx_times:
        print("No data found.")
        return

    data = np.concatenate(all_dx_times)
    
    # 1. Raw Statistics (No Clipping)
    print("\n" + "="*40)
    print(" [1] Raw Statistics (Entire Range)")
    print("="*40)
    print(f"Total Samples : {len(data)}")
    print(f"Min dx_time   : {np.min(data):.4f} s")
    print(f"Max dx_time   : {np.max(data):.4f} s")
    print(f"Mean          : {np.mean(data):.4f} s")
    print(f"Median        : {np.median(data):.4f} s")
    
    # 2. Closing In Analysis (Negative Values)
    closing_in_data = data[data < 0]
    print("\n" + "="*40)
    print(" [2] Closing In (Approaching) Analysis (dx_time < 0)")
    print("="*40)
    print(f"Count (Closing): {len(closing_in_data)} ({len(closing_in_data)/len(data)*100:.1f}%)")
    print(f"Min (Most Urgent): {np.max(closing_in_data):.4f} s (closest to 0)")
    print(f"Max (Furthest)   : {np.min(closing_in_data):.4f} s (large negative)")
    
    # Threshold Histogram for Decision
    print("\n--- Distribution in Critical Closing Range [-10s, 0s] ---")
    thresholds = [-1, -2, -3, -4, -5, -6, -8, -10]
    for t in thresholds:
        # t보다 큰 값 (예: -2보다 크고 0보다 작은 값 = -2초 이내 충돌/도달)
        count = np.sum((data > t) & (data < 0))
        ratio = count / len(data) * 100
        print(f"Samples within {t}s ~ 0s : {count:6d} ({ratio:.2f}%)")

    print("\n" + "="*40)
    print(" [3] Moving Away Analysis (dx_time > 0)")
    print("="*40)
    # 0 ~ 5s 사이 분포 확인
    thresholds_pos = [1, 2, 3, 4, 5, 8, 10]
    for t in thresholds_pos:
        count = np.sum((data > 0) & (data < t))
        ratio = count / len(data) * 100
        print(f"Samples within 0s ~ {t}s : {count:6d} ({ratio:.2f}%)")

    # 3. Visualization
    plt.figure(figsize=(12, 5))

    plt.plot()
    # 히스토그램을 -15 ~ +15 범위로 제한해서 자세히 봄 (Raw Min/Max는 출력했으므로)
    bins = np.linspace(-15, 15, 151) # 0.2s 단위
    plt.hist(data, bins=bins, color='gray', alpha=0.5, label='Others')
    
    # Highlight Closing In
    plt.hist(data[(data > -10) & (data < 0)], bins=bins, color='red', alpha=0.7, label='Closing In (-10s~0s)')
    plt.hist(data[(data > 0) & (data < 10)], bins=bins, color='green', alpha=0.7, label='Moving Away (0s~+10s)')
    
    plt.axvline(0, color='black', linestyle='--')
    plt.title("Focused Distribution (-15s to +15s)\nRed area is critical for T_back")
    plt.xlabel("dx_time (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    

    plt.tight_layout()
    plt.savefig("src/analyze/out/dx_time_analysis.png")
    print("\nSaved visualization to src/analyze/out/dx_time_analysis.png")

if __name__ == "__main__":
    main()