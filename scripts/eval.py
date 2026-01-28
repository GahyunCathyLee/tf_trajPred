#!/usr/bin/env python3
# scripts/eval.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
import torch
from torch.utils.data import DataLoader, ConcatDataset

import platform

# [변경] MmapDataset 및 관련 모듈 import
from src.datasets.mmap_dataset import MmapDataset
from src.datasets.collate import collate_batch
from src.models.build import build_model
from src.utils import set_seed, resolve_path, resolve_data_paths
# [변경] 통계 로드 함수 교체
from src.stats import load_stats_for_ablation, assert_stats_match_batch_dims, make_stats_filename
from src.log import log_eval_to_csv
from src.engine import evaluate
from src.scenarios import load_window_labels_csv

def print_env_info(device: torch.device):
    print("[ENV] Python:", platform.python_version())
    print("[ENV] OS:", platform.platform())
    print("[ENV] Torch:", torch.__version__)
    print("[ENV] CUDA available:", torch.cuda.is_available())
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(idx)
        print(f"[ENV] GPU[{idx}]: {torch.cuda.get_device_name(idx)}")
        print(f"[ENV] GPU Mem: {prop.total_memory / (1024**3):.2f} GB")
    print()

def _load_ckpt_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError(f"Unknown ckpt format: {type(ckpt)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="YAML config path")
    ap.add_argument("--ckpt", type=str, required=True, help="checkpoint .pt path")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv_out", type=str, default="results/results.csv", help="append one-line summary to this CSV")

    # stratified eval
    ap.add_argument("--window_labels", type=str, default="", help="path to window_labels.csv (overrides config)")
    ap.add_argument("--save_event_csv", type=str, default="results/result_per_event.csv")
    ap.add_argument("--save_state_csv", type=str, default="results/result_per_state.csv")
    ap.add_argument("--epoch", type=int, default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    # Config Overrides
    if args.batch_size: cfg["data"]["batch_size"] = args.batch_size
    if args.num_workers: cfg["data"]["num_workers"] = args.num_workers

    batch_size = int(cfg.get("data", {}).get("batch_size", 128))
    num_workers = int(cfg.get("data", {}).get("num_workers", 8))

    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print_env_info(device)

    # -------------------------
    # Path Resolution
    # -------------------------
    paths = resolve_data_paths(cfg)
    tag = str(paths.get("tag", "unknown"))

    # [수정] Mmap 경로 설정 (train.py와 동일하게)
    exid_dir = paths.get("exid_pt_dir", Path(f"./data/exiD/data_mmap/exid_{tag}"))
    highd_dir = paths.get("highd_pt_dir", Path(f"./data/highD/data_mmap/highd_{tag}"))
    
    exid_splits_dir = paths.get("exid_splits_dir", Path("./data/exiD/splits"))
    highd_splits_dir = paths.get("highd_splits_dir", Path("./data/highD/splits"))
    splits_dir = Path("./data/combined/splits") # combined split dir default

    mode = str(cfg.get("data", {}).get("mode", "exid")).lower()
    
    use_ego_static = bool(cfg.get("features", {}).get("use_ego_static", True))
    use_nb_static = bool(cfg.get("features", {}).get("use_nb_static", True))
    use_neighbors = bool(cfg.get("model", {}).get("use_neighbors", True))

    # -------------------------
    # Stats (Ablation Support)
    # -------------------------
    print(f"[INFO] Loading Stats (EgoStatic={use_ego_static}, NbStatic={use_nb_static}, Neighbors={use_neighbors})")
    stats = None
    if mode == "exid":
        stats = load_stats_for_ablation(exid_dir, use_ego_static, use_nb_static, use_neighbors)
    elif mode == "highd":
        stats = load_stats_for_ablation(highd_dir, use_ego_static, use_nb_static, use_neighbors)
    else: # combined
        s1 = load_stats_for_ablation(exid_dir, use_ego_static, use_nb_static, use_neighbors)
        s2 = load_stats_for_ablation(highd_dir, use_ego_static, use_nb_static, use_neighbors)
        if s1 and s2:
            stats = {}
            for k in s1:
                if s1[k] is not None and s2[k] is not None:
                    stats[k] = (s1[k] + s2[k]) / 2.0
        elif s1: stats = s1
        elif s2: stats = s2

    # -------------------------
    # Datasets Setup (MmapDataset)
    # -------------------------
    import numpy as np
    
    # 1. Split Indices 로드
    target_split = args.split  # 'test', 'val', 'train'
    idx_filename = f"{target_split}_indices.npy"
    
    # Combined 모드일 때와 개별 모드일 때 경로 처리
    if mode == "combined":
        idx_path = splits_dir / idx_filename
    elif mode == "exid":
        idx_path = exid_splits_dir / idx_filename
    else:
        idx_path = highd_splits_dir / idx_filename
        
    if not idx_path.exists():
        # Fallback: try txt based loading logic or raise error
        # 여기서는 Mmap workflow에 맞춰 npy 파일이 있다고 가정
        raise FileNotFoundError(f"Split index file not found: {idx_path}")
        
    split_indices = np.load(idx_path)
    print(f"[INFO] Loaded {len(split_indices)} indices from {idx_path}")

    # 2. Dataset 인스턴스 생성
    # Eval에서는 항상 meta가 필요함 (Stratified Eval)
    ds_kwargs = {
        "use_ego_static": use_ego_static,
        "use_nb_static": use_nb_static,
        "use_neighbors": use_neighbors,
        "stats": stats,
        "return_meta": True 
    }

    eval_targets = []

    if mode == "exid":
        ds = MmapDataset(exid_dir, "exid", **ds_kwargs)
        # 전체 데이터셋 중 split indices에 해당하는 부분만 선택
        test_ds = torch.utils.data.Subset(ds, split_indices)
        eval_targets.append(("exid", test_ds))

    elif mode == "highd":
        ds = MmapDataset(highd_dir, "highd", **ds_kwargs)
        test_ds = torch.utils.data.Subset(ds, split_indices)
        eval_targets.append(("highd", test_ds))

    else: # combined
        # Combined의 경우 split_indices는 ConcatDataset 전체에 대한 인덱스임.
        ds_exid = MmapDataset(exid_dir, "exid", **ds_kwargs)
        ds_highd = MmapDataset(highd_dir, "highd", **ds_kwargs)
        full_ds = ConcatDataset([ds_exid, ds_highd])
        
        test_ds = torch.utils.data.Subset(full_ds, split_indices)
        eval_targets.append(("combined", test_ds))
        
        # (옵션) Combined 학습 모델을 exid/highd 각각에 대해 평가하려면
        # 별도의 indices가 필요하거나 전체를 평가해야 함. 
        # 여기서는 편의상 Combined Subset 하나만 추가합니다.

    # -------------------------
    # Loop over targets
    # -------------------------
    # Label LUT
    labels_lut = None
    labels_cfg = cfg.get("data", {}).get("scenario_labels", None)
    # CLI override
    if args.window_labels:
        labels_cfg = args.window_labels

    if labels_cfg:
        if isinstance(labels_cfg, str):
            labels_lut = load_window_labels_csv(Path(labels_cfg))
        elif isinstance(labels_cfg, dict):
            merged = {}
            if "exid" in labels_cfg:
                merged.update(load_window_labels_csv(Path(labels_cfg["exid"])))
            if "highd" in labels_cfg:
                merged.update(load_window_labels_csv(Path(labels_cfg["highd"])))
            labels_lut = merged

    # Model Load
    model = build_model(cfg).to(device)
    state_dict = _load_ckpt_state_dict(Path(args.ckpt))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    loss_cfg = cfg.get("loss", {})
    predict_delta = bool(cfg.get("model", {}).get("predict_delta", False))
    data_hz = float(cfg.get("data", {}).get("hz", 0.0))

    for target_name, target_ds in eval_targets:
        print(f"\n{'='*10} Evaluating: {target_name} (Size: {len(target_ds)}) {'='*10}")
        
        curr_loader = DataLoader(
            target_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            collate_fn=collate_batch,
        )

        curr_save_event = None
        if args.save_event_csv:
             p = Path(args.save_event_csv)
             curr_save_event = p.parent / f"{p.stem}_{target_name}{p.suffix}"
        
        curr_save_state = None
        if args.save_state_csv:
             p = Path(args.save_state_csv)
             curr_save_state = p.parent / f"{p.stem}_{target_name}{p.suffix}"

        metrics = evaluate(
            model=model,
            loader=curr_loader,
            device=device,
            use_amp=bool(args.use_amp),
            predict_delta=predict_delta,
            w_ade=float(loss_cfg.get("w_ade", 1.0)),
            w_fde=float(loss_cfg.get("w_fde", 0.0)),
            w_cls=float(loss_cfg.get("w_cls", 0.5)),
            w_rmse=float(loss_cfg.get("w_rmse", 0.0)),
            data_hz=data_hz,
            labels_lut=labels_lut,
            save_event_path=curr_save_event,
            save_state_path=curr_save_state,
            epoch=args.epoch,
            measure_latency=True,
        )

        # CSV Logging
        if args.csv_out:
            stats_path_log = exid_dir / "stats.npz" # 임시 경로 (로깅용)
            log_eval_to_csv(
                csv_out=Path(args.csv_out),
                cfg=cfg,
                cfg_path=cfg_path,
                ckpt_path=Path(args.ckpt),
                split=args.split,
                mode=target_name,
                tag=tag,
                device=str(device),
                batch_size=int(batch_size),
                num_workers=int(num_workers),
                seed=int(args.seed),
                use_amp=bool(args.use_amp),
                stats_path=stats_path_log,
                use_ego_static=use_ego_static,
                use_nb_static=use_nb_static,
                metrics=metrics,
            )
            print(f"[OK] appended to: {args.csv_out}")

        print(f"[RESULT] ({target_name}) loss={metrics['loss']:.4f} ADE={metrics['ade']:.4f} FDE={metrics['fde']:.4f}")

if __name__ == "__main__":
    main()
