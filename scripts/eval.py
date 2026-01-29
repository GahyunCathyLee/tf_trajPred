#!/usr/bin/env python3
# scripts/eval.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset

import platform

from src.datasets.mmap_dataset import MmapDataset
from src.datasets.collate import collate_batch
from src.models.build import build_model
from src.utils import set_seed, resolve_data_paths
from src.stats import load_stats_for_ablation
from src.log import log_eval_to_csv
from src.engine import evaluate
from src.scenarios import load_window_labels_csv

def print_env_info(device: torch.device):
    print("[ENV] Python:", platform.python_version())
    print("[ENV] OS:", platform.platform())
    print("[ENV] Torch:", torch.__version__)
    print("[ENV] CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[ENV] Torch CUDA:", torch.version.cuda)
        print("[ENV] cuDNN:", torch.backends.cudnn.version())
    
    try:
        print("[ENV] torch.get_num_threads:", torch.get_num_threads())
    except Exception:
        pass

    if device.type == "cuda":
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        prop = torch.cuda.get_device_properties(idx)
        total_gb = prop.total_memory / (1024**3)
        print(f"[ENV] GPU[{idx}]: {name}")
        print(f"[ENV] GPU Mem: {total_gb:.2f} GB, SMs={prop.multi_processor_count}")
    print()

def _load_ckpt_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # 1. state_dict 추출
    state_dict = None
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise RuntimeError(f"Unknown ckpt format: {type(ckpt)}")
    
    # 2. 접두사 제거 (Compiled Model 호환)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
            
    return new_state_dict


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
    print(f"[INFO] device={device}")
    print_env_info(device)

    # -------------------------
    # Path Resolution
    # -------------------------
    paths = resolve_data_paths(cfg)
    tag = str(paths.get("tag", "unknown"))

    exid_mmap_dir = Path(f"./data/exiD/data_mmap/exid_{tag}")
    highd_mmap_dir = Path(f"./data/highD/data_mmap/highd_{tag}")
    
    # Splits 경로
    splits_dir = Path("./data/combined/splits") 
    exid_splits_dir = Path("./data/exiD/splits")
    highd_splits_dir = Path("./data/highD/splits")

    mode = str(cfg.get("data", {}).get("mode", "exid")).lower()
    if mode not in ("exid", "highd", "combined"):
        raise ValueError(f"data.mode must be one of exid/highd/combined, got: {mode}")

    use_ego_static = bool(cfg.get("features", {}).get("use_ego_static", True))
    use_nb_static = bool(cfg.get("features", {}).get("use_nb_static", True))
    use_neighbors = bool(cfg.get("model", {}).get("use_neighbors", True))

    # -------------------------
    # Stats Loading
    # -------------------------
    print(f"[INFO] Loading Stats (EgoStatic={use_ego_static}, NbStatic={use_nb_static}, Neighbors={use_neighbors})")
    stats = None
    # MmapStats 로드 (경로는 Mmap dir 내부 stats.npz)
    try:
        if mode == "exid":
            stats = load_stats_for_ablation(exid_mmap_dir, use_ego_static, use_nb_static, use_neighbors)
        elif mode == "highd":
            stats = load_stats_for_ablation(highd_mmap_dir, use_ego_static, use_nb_static, use_neighbors)
        else: # combined
            s1 = load_stats_for_ablation(exid_mmap_dir, use_ego_static, use_nb_static, use_neighbors)
            s2 = load_stats_for_ablation(highd_mmap_dir, use_ego_static, use_nb_static, use_neighbors)
            if s1 and s2:
                stats = {}
                for k in s1:
                    if s1[k] is not None and s2[k] is not None:
                        stats[k] = (s1[k] + s2[k]) / 2.0
            elif s1: stats = s1
            elif s2: stats = s2
    except Exception as e:
        print(f"[WARN] Failed to load stats automatically: {e}")
        print("       Ensure stats.npz exists in data_mmap directories.")

    if stats is None:
        print("[WARN] Stats not loaded. Evaluation might be incorrect if model expects normalized data.")
    else:
        # Save stats path for logging later
        stats_path_log = exid_mmap_dir / "stats.npz" if mode != "highd" else highd_mmap_dir / "stats.npz"

    # -------------------------
    # Datasets Setup (MmapDataset)
    # -------------------------
    target_split = args.split  # 'test', 'val', 'train'
    idx_filename = f"{target_split}_indices.npy"
    
    # 1. 인덱스 로드 경로 결정
    if mode == "combined":
        idx_path = splits_dir / idx_filename
    elif mode == "exid":
        idx_path = exid_splits_dir / idx_filename
    else:
        idx_path = highd_splits_dir / idx_filename
        
    if not idx_path.exists():
        raise FileNotFoundError(f"Split index file not found: {idx_path}\nDid you run create_splits.py?")
        
    split_indices = np.load(idx_path)
    print(f"[INFO] Loaded {len(split_indices)} indices from {idx_path}")

    # 2. Dataset kwargs (MmapDataset 공통 인자)
    ds_kwargs = {
        "use_ego_static": use_ego_static,
        "use_nb_static": use_nb_static,
        "use_neighbors": use_neighbors,
        "stats": stats,
        "return_meta": True  # Stratified Eval을 위해 필수
    }

    eval_targets = []

    # 3. 데이터셋 구성 및 타겟 추가
    if mode == "exid":
        full_ds = MmapDataset(exid_mmap_dir, "exid", dataset_name="exid", **ds_kwargs)
        test_ds = Subset(full_ds, split_indices)
        eval_targets.append(("exid", test_ds))

    elif mode == "highd":
        full_ds = MmapDataset(highd_mmap_dir, "highd", dataset_name="highd", **ds_kwargs)
        test_ds = Subset(full_ds, split_indices)
        eval_targets.append(("highd", test_ds))

    else: # combined
        # (1) Full Datasets
        exid_full = MmapDataset(exid_mmap_dir, "exid", dataset_name="exid", **ds_kwargs)
        highd_full = MmapDataset(highd_mmap_dir, "highd", dataset_name="highd", **ds_kwargs)
        
        # (2) Combined Target
        combined_full = ConcatDataset([exid_full, highd_full])
        combined_subset = Subset(combined_full, split_indices)
        eval_targets.append(("combined", combined_subset))
        
        # (3) Sub-targets (ExiD only / HighD only)
        # ConcatDataset에서 앞부분은 ExiD, 뒷부분은 HighD임.
        cutoff = len(exid_full)
        
        # exiD에 해당하는 인덱스 (cutoff보다 작은 인덱스)
        exid_indices = split_indices[split_indices < cutoff]
        if len(exid_indices) > 0:
            exid_subset = Subset(exid_full, exid_indices)
            eval_targets.append(("exid_only", exid_subset))
        
        # highD에 해당하는 인덱스 (cutoff 이상인 인덱스 -> Offset 조정 필요)
        highd_indices = split_indices[split_indices >= cutoff]
        if len(highd_indices) > 0:
            highd_subset = Subset(highd_full, highd_indices - cutoff)
            eval_targets.append(("highd_only", highd_subset))

    # -------------------------
    # Label LUT Loading
    # -------------------------
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

    # -------------------------
    # Model Setup
    # -------------------------
    model = build_model(cfg).to(device)
    state_dict = _load_ckpt_state_dict(Path(args.ckpt))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    loss_cfg = cfg.get("loss", {})
    predict_delta = bool(cfg.get("model", {}).get("predict_delta", False))
    data_hz = float(cfg.get("data", {}).get("hz", 0.0))
    epoch_for_csv = args.epoch if args.epoch is not None else -1

    # -------------------------
    # Evaluation Loop
    # -------------------------
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
            persistent_workers=(num_workers > 0),
        )

        curr_save_event = None
        if args.save_event_csv:
             p = Path(args.save_event_csv)
             curr_save_event = p.parent / f"{p.stem}_{target_name}{p.suffix}"
        
        curr_save_state = None
        if args.save_state_csv:
             p = Path(args.save_state_csv)
             curr_save_state = p.parent / f"{p.stem}_{target_name}{p.suffix}"

        # Engine evaluate call
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
            epoch=epoch_for_csv,
            measure_latency=True,
            latency_iters=200,
            latency_warmup=30,
            latency_per_sample=True,
        )

        # CSV Logging
        if args.csv_out:
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
                stats_path=stats_path_log if stats else None,
                use_ego_static=use_ego_static,
                use_nb_static=use_nb_static,
                metrics=metrics,
            )
            print(f"[OK] appended to: {args.csv_out} (mode={target_name})")

        # Detailed Print (Local Script style)
        print(f"[RESULT] ({target_name}) loss={metrics['loss']:.6f} ADE={metrics['ade']:.6f} FDE={metrics['fde']:.6f} RMSE={metrics.get('rmse', float('nan')):.6f}")
        print(  
            f"[RESULT] ({target_name}) VEL={metrics.get('vel', float('nan')):.6f} "
            f"ACC={metrics.get('acc', float('nan')):.6f} "
            f"JERK={metrics.get('jerk', float('nan')):.6f}"
        )
        
        rmse_str = " ".join([f"RMSE({t}s)={metrics.get(f'rmse_{t}s', float('nan')):.4f}" for t in [1, 2, 3, 4, 5]])
        print(f"[RESULT] ({target_name}) {rmse_str}")

if __name__ == "__main__":
    main()
