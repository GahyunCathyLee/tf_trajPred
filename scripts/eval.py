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

from src.datasets.pt_dataset import PtWindowDataset
from src.datasets.collate import collate_batch
from src.models.build import build_model
from src.utils import set_seed, resolve_data_paths
from src.stats import load_stats_npz_strict, assert_stats_match_batch_dims, make_stats_filename
from src.log import log_eval_to_csv
from src.engine import evaluate
from src.scenarios import load_window_labels_csv

def print_env_info(device: torch.device):
    print("[ENV] Python:", platform.python_version())
    print("[ENV] OS:", platform.platform())
    print("[ENV] Torch:", torch.__version__)
    print("[ENV] CUDA available:", torch.cuda.is_available())
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
    ap.add_argument("--window_labels", type=str, default="", help="path to window_labels.csv")
    ap.add_argument("--save_event_csv", type=str, default="results/result_per_event.csv", help="append per-epoch stratified EVENT summary to csv")
    ap.add_argument("--save_state_csv", type=str, default="results/result_per_state.csv", help="append per-epoch stratified STATE summary to csv")
    ap.add_argument("--epoch", type=int, default=None, help="epoch number to write into stratified csv (optional)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    batch_size = int(cfg.get("data", {}).get("batch_size", 128))
    num_workers = int(cfg.get("data", {}).get("num_workers", 2))

    set_seed(int(args.seed))

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"[INFO] device={device}")
    print_env_info(device)

    # -------------------------
    # resolve paths
    # -------------------------
    paths = resolve_data_paths(cfg)
    tag = str(paths.get("tag", "unknown"))

    exid_pt_dir = paths.get("exid_pt_dir", Path(f"./data/exiD/data_pt/exid_{tag}"))
    highd_pt_dir = paths.get("highd_pt_dir", Path(f"./data/highD/data_pt/highd_{tag}"))
    exid_splits_dir = paths.get("exid_splits_dir", Path("./data/exiD/splits"))
    highd_splits_dir = paths.get("highd_splits_dir", Path("./data/highD/splits"))
    
    exid_stats_dir = paths.get("exid_stats_dir", Path("./data/exiD/stats"))
    highd_stats_dir = paths.get("highd_stats_dir", Path("./data/highD/stats"))
    combined_stats_dir = paths.get("combined_stats_dir", Path("./data/combined/stats"))

    use_ego_static = bool(cfg.get("features", {}).get("use_ego_static", True))
    use_nb_static = bool(cfg.get("features", {}).get("use_nb_static", True))


    mode = str(cfg.get("data", {}).get("mode", "exid")).lower()
    if mode not in ("exid", "highd", "combined"):
        raise ValueError(f"data.mode must be one of exid/highd/combined, got: {mode}")

    if mode == "exid":
        splits_index_dir = Path("./data/exiD/splits")
    elif mode == "highd":
        splits_index_dir = Path("./data/highD/splits")
    else:
        splits_index_dir = Path("./data/combined/splits")

    # -------------------------
    # stats (subprocess only)
    # -------------------------
    stats_fname = make_stats_filename(tag, use_ego_static, use_nb_static)

    if mode == "exid":
        stats_path = exid_stats_dir / stats_fname
    elif mode == "highd":
        stats_path = highd_stats_dir / stats_fname
    else:
        stats_path = combined_stats_dir / stats_fname

    stats: Optional[Dict[str, torch.Tensor]] = None
    if stats_path.exists():
        print(f"[INFO] Loading stats: {stats_path}")
        stats = load_stats_npz_strict(stats_path)
    else:
        raise FileNotFoundError(f"Missing stats: {stats_path}")

    if stats is not None:
        ego_dim_cfg = int(cfg["model"]["ego_dim"])
        nb_dim_cfg = int(cfg["model"]["nb_dim"])
        assert_stats_match_batch_dims(stats, ego_dim_cfg, nb_dim_cfg, stats_path)

    # -------------------------
    # Scenario Labels & Config Check
    # -------------------------
    labels_lut = None
    labels_cfg = cfg.get("data", {}).get("scenario_labels", None)
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

    sam_cfg = cfg.get("data", {}).get("scenario_sampling", None)
    use_scenario_sampling = bool(sam_cfg and labels_lut is not None)

    # -------------------------
    # datasets / loaders setup
    # -------------------------
    eval_targets = []
    
    target_split = args.split  # "test" or "val"

    if use_scenario_sampling:
        print(f"\n[DATA-MODE] Scenario Sampling ON -> Loading indices from {splits_index_dir}/{target_split}_indices.npy")
        
        # 1. 인덱스 로드
        try:
            indices = np.load(splits_index_dir / f"{target_split}_indices.npy")
        except FileNotFoundError:
            raise FileNotFoundError(f"Index file not found. Did you run create_splits.py for mode={mode}?")

        # 2. 전체 데이터셋 로드 (Full Load)
        if mode == "exid":
            full_ds = PtWindowDataset(exid_pt_dir, split_txt=None, stats=stats, return_meta=True, 
                                      use_ego_static=use_ego_static, use_nb_static=use_nb_static, dataset_name="exid")
            test_ds = Subset(full_ds, indices)
            eval_targets.append(("exid", test_ds))

        elif mode == "highd":
            full_ds = PtWindowDataset(highd_pt_dir, split_txt=None, stats=stats, return_meta=True,
                                      use_ego_static=use_ego_static, use_nb_static=use_nb_static, dataset_name="highd")
            test_ds = Subset(full_ds, indices)
            eval_targets.append(("highd", test_ds))

        else: # combined
            # (1) Full Datasets 로드
            exid_full = PtWindowDataset(exid_pt_dir, split_txt=None, stats=stats, return_meta=True, 
                                        use_ego_static=use_ego_static, use_nb_static=use_nb_static, dataset_name="exid")
            highd_full = PtWindowDataset(highd_pt_dir, split_txt=None, stats=stats, return_meta=True,
                                         use_ego_static=use_ego_static, use_nb_static=use_nb_static, dataset_name="highd")
            
            # (2) Combined Dataset 생성
            combined_full = ConcatDataset([exid_full, highd_full])
            
            # (3) Combined Eval Target
            combined_subset = Subset(combined_full, indices)
            eval_targets.append(("combined", combined_subset))
            
            # (4) ExiD Only / HighD Only 분리 (인덱스 기준)
            cutoff = len(exid_full)
            
            # exiD에 해당하는 인덱스만 골라내기
            exid_indices = indices[indices < cutoff]
            if len(exid_indices) > 0:
                exid_subset = Subset(exid_full, exid_indices)
                eval_targets.append(("exid_only", exid_subset))
            
            # highD에 해당하는 인덱스만 골라내기 (Offset 보정 필요)
            highd_indices = indices[indices >= cutoff]
            if len(highd_indices) > 0:
                # highD 개별 데이터셋 기준으로는 인덱스가 0부터 시작해야 하므로 cutoff를 뺌
                highd_subset = Subset(highd_full, highd_indices - cutoff)
                eval_targets.append(("highd_only", highd_subset))

    else:
        # [OLD WAY] 기존 텍스트 파일 기반 로딩
        print(f"\n[DATA-MODE] Scenario Sampling OFF -> Loading files via {target_split}.txt")
        
        if mode == "exid":
            test_ds = PtWindowDataset(
                data_dir=exid_pt_dir,
                split_txt=exid_splits_dir / f"{target_split}.txt",
                stats=stats,
                return_meta=True,
                use_ego_static=use_ego_static,
                use_nb_static=use_nb_static,
                dataset_name="exid",
            )
            eval_targets.append(("exid", test_ds))

        elif mode == "highd":
            test_ds = PtWindowDataset(
                data_dir=highd_pt_dir,
                split_txt=highd_splits_dir / f"{target_split}.txt",
                stats=stats,
                return_meta=True,
                use_ego_static=use_ego_static,
                use_nb_static=use_nb_static,
                dataset_name="highd",
            )
            eval_targets.append(("highd", test_ds))

        else: # combined
            exid_test_ds = PtWindowDataset(
                data_dir=exid_pt_dir,
                split_txt=exid_splits_dir / f"{target_split}.txt",
                stats=stats,
                return_meta=True,
                use_ego_static=use_ego_static,
                use_nb_static=use_nb_static,
                dataset_name="exid",
            )
            highd_test_ds = PtWindowDataset(
                data_dir=highd_pt_dir,
                split_txt=highd_splits_dir / f"{target_split}.txt",
                stats=stats,
                return_meta=True,
                use_ego_static=use_ego_static,
                use_nb_static=use_nb_static,
                dataset_name="highd",
            )
            combined_ds = ConcatDataset([exid_test_ds, highd_test_ds])

            eval_targets.append(("combined", combined_ds))
            eval_targets.append(("exid_only", exid_test_ds))
            eval_targets.append(("highd_only", highd_test_ds))

    # -------------------------
    # Loop over targets
    # -------------------------
    labels_lut = None
    labels_cfg = cfg.get("data", {}).get("scenario_labels", None)
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


    model = build_model(cfg).to(device)
    state_dict = _load_ckpt_state_dict(Path(args.ckpt))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval() 

    loss_cfg = cfg.get("loss", {})
    predict_delta = bool(cfg.get("model", {}).get("predict_delta", False))
    epoch_for_csv = args.epoch if args.epoch is not None else -1
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
            # [NEW] Pass paths
            cfg_path=cfg_path,
            ckpt_path=Path(args.ckpt),
        )

        # CSV Logging
        csv_out = Path(args.csv_out) if args.csv_out else None
        if csv_out is not None:
            log_eval_to_csv(
                csv_out=csv_out,
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
                stats_path=stats_path,
                use_ego_static=use_ego_static,
                use_nb_static=use_nb_static,
                metrics=metrics,
            )
            print(f"[OK] appended to: {csv_out} (mode={target_name})")

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