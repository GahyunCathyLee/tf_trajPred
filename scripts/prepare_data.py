#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd: list):
    print(f"\n[EXEC] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with return code {result.returncode}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Automated Preprocessing Pipeline for highD and exiD")
    parser.add_argument("--t_front", type=int, required=True, help="Future horizon (TF)")
    parser.add_argument("--t_back", type=int, required=True, help="History horizon (TB)")
    parser.add_argument("--vy_eps", type=float, default=0.27, help="Velocity epsilon (e.g., 0.05)")
    args = parser.parse_args()

    # 1. 태그 생성 (예: 0.05 -> 05)
    vy_int = int(round(args.vy_eps * 100))
    tag_suffix = f"TB{args.t_back}_TF{args.t_front}_vy{vy_int:02d}"
    
    exid_tag = f"exid_{tag_suffix}"
    highd_tag = f"highd_{tag_suffix}"

    # 2. Raw -> NPZ 변환
    # highD
    run_command([
        "python3", "data/highD/scripts/highd_raw_to_npz.py",
        "--t_front", str(args.t_front),
        "--t_back", str(args.t_back),
        "--vy_eps", str(args.vy_eps)
    ])

    # exiD
    run_command([
        "python3", "data/exiD/scripts/exid_raw_to_npz.py",
        "--t_front", str(args.t_front),
        "--t_back", str(args.t_back),
        "--vy_eps", str(args.vy_eps),
    ])

    # 3. NPZ -> Mmap 변환 (preprocess_mmap)
    # exiD Mmap
    run_command([
        "python3", "-m", "scripts.preprocess_mmap",
        "--npz_dir", f"data/exiD/data_npz/{exid_tag}",
        "--out_dir", f"data/exiD/data_mmap/{exid_tag}",
        "--calc_stats"
    ])

    # highD Mmap
    run_command([
        "python3", "-m", "scripts.preprocess_mmap",
        "--npz_dir", f"data/highD/data_npz/{highd_tag}",
        "--out_dir", f"data/highD/data_mmap/{highd_tag}",
        "--calc_stats"
    ])

    print("\n" + "="*40)
    print(f"✅ All preprocessing steps completed for {tag_suffix}!")
    print("="*40)

if __name__ == "__main__":
    main()