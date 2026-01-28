import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader, Subset, ConcatDataset, WeightedRandomSampler

from src.utils import resolve_data_paths
from src.datasets.pt_dataset import PtWindowDataset
from src.datasets.collate import collate_batch
from src.scenarios import load_window_labels_csv, build_sample_weights
from src.stats import load_stats_npz_strict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # 1. Config 로드
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    
    # 2. 데이터셋 준비
    mode = cfg["data"].get("mode", "combined").lower()
    paths = resolve_data_paths(cfg)
    tag = str(paths.get("tag", "unknown"))
    
    # 경로 설정 (train.py 로직과 동일)
    exid_pt_dir = paths.get("exid_pt_dir", Path(f"./data/exiD/data_pt/exid_{tag}"))
    highd_pt_dir = paths.get("highd_pt_dir", Path(f"./data/highD/data_pt/highd_{tag}"))
    
    if mode == "exid":
        splits_index_dir = Path("./data/exiD/splits")
    elif mode == "highd":
        splits_index_dir = Path("./data/highD/splits")
    else:
        splits_index_dir = Path("./data/combined/splits")

    # 3. 데이터셋 로드 & Split
    print(f"[INFO] Mode: {mode}, Loading Datasets...")
    full_ds = None
    if mode in ("exid", "combined"):
        ds_exid = PtWindowDataset(exid_pt_dir, split_txt=None, return_meta=True, dataset_name="exid")
        full_ds = ds_exid
    if mode in ("highd", "combined"):
        ds_highd = PtWindowDataset(highd_pt_dir, split_txt=None, return_meta=True, dataset_name="highd")
        if full_ds: full_ds = ConcatDataset([full_ds, ds_highd])
        else: full_ds = ds_highd
    
    train_idx = np.load(splits_index_dir / "train_indices.npy")
    train_ds = Subset(full_ds, train_idx)
    
    # 4. 라벨 로드 & Weights 계산
    labels_cfg = cfg["data"].get("scenario_labels", {})
    labels_lut = {}
    if isinstance(labels_cfg, str):
         labels_lut.update(load_window_labels_csv(Path(labels_cfg)))
    elif isinstance(labels_cfg, dict):
        if "exid" in labels_cfg: labels_lut.update(load_window_labels_csv(Path(labels_cfg["exid"])))
        if "highd" in labels_cfg: labels_lut.update(load_window_labels_csv(Path(labels_cfg["highd"])))

    # [수정] Config에서 Alpha 값 읽어오기
    sam_cfg = cfg["data"].get("scenario_sampling", {})
    target_alpha = float(sam_cfg.get("alpha", 0.5))
    
    print(f"\n[INFO] Building Weights with ALPHA = {target_alpha} ...")
    
    # [수정] target_alpha 변수 전달
    weights = build_sample_weights(
        train_ds, labels_lut, mode="event", alpha=target_alpha, log=True
    )
    
    # 5. Sampler & Loader 생성
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
    loader = DataLoader(
        train_ds, 
        batch_size=128, 
        sampler=sampler, 
        collate_fn=collate_batch,
        num_workers=8
    )

    # 6. 실제 분포 확인
    print("\n[CHECK] Iterating through DataLoader to count ACTUAL samples...")
    actual_counts = Counter()
    
    total_samples = 0
    # 전체를 다 돌면 오래 걸리니, 테스트용으로 100 배치(약 12,800개 샘플)만 확인해도 충분합니다.
    # 전체를 보려면 enumerate(loader) 전체를 순회하세요.
    for i, batch in enumerate(loader):
        if i >= 200: break # 속도를 위해 200 배치만 확인 (필요시 삭제)
        
        metas = batch["meta"]
        for m in metas:
            rid = int(m["recordingId"])
            tid = int(m["trackId"])
            t0 = int(m["t0_frame"])
            
            rec = labels_lut.get((rid, tid, t0), {})
            lab = str(rec.get("event_label", "unknown"))
            
            # 통합 로직 적용 (scenarios.py 수정본과 동일하게)
            if lab in ["simple_lane_change", "lane_change_other"]:
                lab = "lane_change"
                
            actual_counts[lab] += 1
            total_samples += 1

    # 7. 결과 출력
    print(f"\n=== ACTUAL Sampling Distribution (Checked {total_samples:,} samples with alpha={target_alpha}) ===")
    for lab, count in actual_counts.most_common():
        ratio = count / total_samples * 100
        print(f"  - {lab:>16s}: {count:8d} ({ratio:6.2f}%)")

if __name__ == "__main__":
    main()