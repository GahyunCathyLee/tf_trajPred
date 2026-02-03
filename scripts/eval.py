#!/usr/bin/env python3
# scripts/eval.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset

import platform

from src.datasets.mmap_dataset import MmapDataset
from src.datasets.collate import collate_batch
from src.models.build import build_model
from src.utils import set_seed, resolve_data_paths
from src.stats import make_stats_filename, compute_stats_if_needed, load_stats_npz_strict
from src.engine import evaluate
from src.scenarios import load_window_labels_csv
from src.log import log_eval_to_csv


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

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise RuntimeError(f"Unknown ckpt format: {type(ckpt)}")

    # compiled model prefix 대응
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

    ap.add_argument("--window_labels", type=str, default="", help="path to window_labels.csv (overrides config)")
    ap.add_argument("--save_event_csv", type=str, default="results/result_per_event.csv")
    ap.add_argument("--save_state_csv", type=str, default="results/result_per_state.csv")
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--csv_out", type=str, default="results/results.csv", help="append one-line summary to this CSV")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg: Dict[str, Any] = yaml.safe_load(cfg_path.read_text())

    # config override
    cfg.setdefault("data", {})
    if args.batch_size:
        cfg["data"]["batch_size"] = args.batch_size
    if args.num_workers:
        cfg["data"]["num_workers"] = args.num_workers

    batch_size = int(cfg.get("data", {}).get("batch_size", 128))
    num_workers = int(cfg.get("data", {}).get("num_workers", 8))

    mode = str(cfg.get("data", {}).get("mode", "combined")).lower()
    if mode not in ("exid", "highd", "combined"):
        raise ValueError(f"data.mode must be one of exid/highd/combined, got: {mode}")

    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"[INFO] device={device}")
    print_env_info(device)

    # -------------------------
    # Feature Toggles
    # -------------------------
    feat_cfg = cfg.get("features", {})
    use_ego_static = bool(feat_cfg.get("use_ego_static", True))
    use_nb_static = bool(feat_cfg.get("use_nb_static", True))
    use_neighbors = bool(cfg.get("model", {}).get("use_neighbors", True))

    use_lead = bool(feat_cfg.get("use_lead", False))
    use_lc_state = bool(feat_cfg.get("use_lc_state", True))
    use_dxtime = bool(feat_cfg.get("use_dxtime", True))
    use_gate = bool(feat_cfg.get("use_gate", True))

    print("==== Feature Toggles ====")
    print(f"use_neighbors  = {use_neighbors}")
    print(f"use_lead       = {use_lead}")
    print(f"use_ego_static = {use_ego_static}")
    print(f"use_nb_static  = {use_nb_static}")
    print(f"use_lc_state   = {use_lc_state}")
    print(f"use_dxtime     = {use_dxtime}")
    print(f"use_gate       = {use_gate}")

    # -------------------------
    # Paths
    # -------------------------
    paths = resolve_data_paths(cfg)
    tag = str(paths.get("tag", "T2_Tf5_hz3"))

    exid_dir = paths.get("exid_pt_dir", Path(f"./data/exiD/data_mmap/exid_{tag}"))
    highd_dir = paths.get("highd_pt_dir", Path(f"./data/highD/data_mmap/highd_{tag}"))

    splits_dir = Path("./data/combined/splits") if mode == "combined" else \
                 Path(f"./data/{'exiD' if mode=='exid' else 'highD'}/splits")

    print(f"[INFO] Data Dir (ExID): {exid_dir}")
    print(f"[INFO] Data Dir (HighD): {highd_dir}")
    print(f"[INFO] Splits Dir: {splits_dir}")

    # -------------------------
    # Stats
    # -------------------------
    print("[INFO] Loading/Ensuring stats...")

    stats_fname = make_stats_filename(
        tag=tag,
        use_ego_static=use_ego_static,
        use_nb_static=use_nb_static,
        use_neighbors=use_neighbors,
        use_lead=use_lead,
        use_lc_state=use_lc_state,
        use_dxtime=use_dxtime,
        use_gate=use_gate,
    )

    if mode == "exid":
        stats_dir = Path("./data/exiD/stats")
        data_dirs = [exid_dir]
        splits_dirs = [Path("./data/exiD/splits")]
    elif mode == "highd":
        stats_dir = Path("./data/highD/stats")
        data_dirs = [highd_dir]
        splits_dirs = [Path("./data/highD/splits")]
    else:
        stats_dir = Path("./data/combined/stats")
        data_dirs = [exid_dir, highd_dir]
        splits_dirs = [Path("./data/exiD/splits"), Path("./data/highD/splits")]

    stats_path = stats_dir / stats_fname

    compute_stats_if_needed(
        stats_path=stats_path,
        data_dir=data_dirs,
        splits_dir=splits_dirs,
        stats_split="train",
        batch_size=int(cfg.get("data", {}).get("batch_size", 512)),
        num_workers=int(cfg.get("data", {}).get("num_workers", 16)),
        data_tag=tag,
        use_neighbors=use_neighbors,
        use_ego_static=use_ego_static,
        use_nb_static=use_nb_static,
        use_lead=use_lead,
        use_lc_state=use_lc_state,
        use_dxtime=use_dxtime,
        use_gate=use_gate,
    )

    stats = load_stats_npz_strict(stats_path)
    print(f"[INFO] Stats loaded: {stats_path}")

    # -------------------------
    # Split indices
    # -------------------------
    idx_path = splits_dir / f"{args.split}_indices.npy"
    if not idx_path.exists():
        raise FileNotFoundError(f"Split index file not found: {idx_path}\nDid you run create_splits.py?")

    split_indices = np.load(idx_path)
    print(f"[INFO] Loaded {len(split_indices)} indices from {idx_path}")

    # -------------------------
    # Dataset build
    # -------------------------
    ds_kwargs = {
        "use_ego_static": use_ego_static,
        "use_nb_static": use_nb_static,
        "use_neighbors": use_neighbors,
        "use_lc_state": use_lc_state,
        "use_dxtime": use_dxtime,
        "use_gate": use_gate,
        "stats": stats,
        "return_meta": True,
        "is_pre_normalized": False,  # ✅ train과 동일
    }

    eval_targets = []

    if mode == "exid":
        full = MmapDataset(exid_dir, tag, dataset_name="exid", **ds_kwargs)
        subset = Subset(full, split_indices)
        eval_targets.append(("exid", subset))

    elif mode == "highd":
        full = MmapDataset(highd_dir, tag, dataset_name="highd", **ds_kwargs)
        subset = Subset(full, split_indices)
        eval_targets.append(("highd", subset))

    else:
        exid_full = MmapDataset(exid_dir, tag, dataset_name="exid", **ds_kwargs)
        highd_full = MmapDataset(highd_dir, tag, dataset_name="highd", **ds_kwargs)

        combined_full = ConcatDataset([exid_full, highd_full])
        combined_subset = Subset(combined_full, split_indices)
        eval_targets.append(("combined", combined_subset))

        cutoff = len(exid_full)
        exid_idx = split_indices[split_indices < cutoff]
        highd_idx = split_indices[split_indices >= cutoff]

        if len(exid_idx) > 0:
            eval_targets.append(("exid_only", Subset(exid_full, exid_idx)))
        if len(highd_idx) > 0:
            eval_targets.append(("highd_only", Subset(highd_full, highd_idx - cutoff)))

    # -------------------------
    # Labels LUT
    # -------------------------
    labels_lut = None
    labels_cfg = cfg.get("data", {}).get("scenario_labels", None)
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
    # Model
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
    # Eval loop
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

        log_eval_to_csv(
            csv_out=Path(args.csv_out),
            cfg=cfg,
            cfg_path=cfg_path,
            ckpt_path=Path(args.ckpt),
            split=args.split if target_name == "combined" else f"{args.split}:{target_name}",
            mode=mode,
            tag=tag,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=int(args.seed),
            use_amp=bool(args.use_amp),
            stats_path=stats_path,
            use_neighbors=use_neighbors,
            use_lead=use_lead,
            use_ego_static=use_ego_static,
            use_nb_static=use_nb_static,
            use_lc_state=use_lc_state,
            use_dxtime=use_dxtime,
            use_gate=use_gate,
            metrics=metrics,
        )

        print(f"[RESULT] ({target_name}) loss={metrics['loss']:.6f} ADE={metrics['ade']:.6f} "
              f"FDE={metrics['fde']:.6f} RMSE={metrics.get('rmse', float('nan')):.6f}")
        print(
            f"[RESULT] ({target_name}) VEL={metrics.get('vel', float('nan')):.6f} "
            f"ACC={metrics.get('acc', float('nan')):.6f} "
            f"JERK={metrics.get('jerk', float('nan')):.6f}"
        )

        rmse_str = " ".join([f"RMSE({t}s)={metrics.get(f'rmse_{t}s', float('nan')):.4f}" for t in [1, 2, 3, 4, 5]])
        print(f"[RESULT] ({target_name}) {rmse_str}")


if __name__ == "__main__":
    main()
