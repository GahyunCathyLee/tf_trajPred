#!/usr/bin/env python3
# scripts/train.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, Subset
from torch.amp import GradScaler

import yaml

from src.utils import set_seed, resolve_path, resolve_data_paths
from src.datasets.collate import collate_batch
from src.models.build import build_model, build_scheduler
from src.engine import train_one_epoch, evaluate
from src.scenarios import load_window_labels_csv, build_sample_weights

from src.datasets.mmap_dataset import MmapDataset
from src.stats import load_stats_for_ablation


def compute_weights_fast(dataset, labels_df, mode="event", alpha=1.0, unknown_w=0.0, clip_max=None):
    print(f"   -> Fast Weight Computation for {dataset.dataset_name}...")
    
    # 1. 데이터셋 메타데이터를 DataFrame으로 변환 (메모리 내 배열 활용)
    # MmapDataset은 meta_rec, meta_track, meta_frame을 이미 numpy 배열로 가지고 있음
    df_data = pd.DataFrame({
        "recordingId": dataset.meta_rec,
        "trackId": dataset.meta_track,
        "t0_frame": dataset.meta_frame
    })
    
    # 2. 라벨 데이터와 병합 (Left Join) - 여기가 핵심 속도 향상 지점
    # (recordingId, trackId, t0_frame) 기준으로 매칭
    merged = df_data.merge(labels_df, on=["recordingId", "trackId", "t0_frame"], how="left")
    
    # 3. 타겟 컬럼 선택 (event_label or state_label)
    col = "event_label" if mode == "event" else "state_label"
    
    # 라벨이 없는 경우(매칭 실패) 처리
    merged[col] = merged[col].fillna("unknown")
    
    # 4. 클래스별 빈도 계산 및 가중치 산출 (Inverse Frequency)
    # 전체 데이터셋 내에서의 빈도를 기준으로 함
    counts = merged[col].value_counts()
    total_count = len(merged)
    
    # 가중치: Total / Count (희귀할수록 값이 커짐)
    weights_map = (total_count / counts).to_dict()
    
    # Unknown(라벨 없음)에 대한 가중치 처리
    if "unknown" in weights_map:
        weights_map["unknown"] = unknown_w
    
    # 5. 각 샘플에 가중치 매핑
    weights = merged[col].map(weights_map).fillna(unknown_w).values
    
    # 6. Alpha 적용 (Smoothing) 및 Clipping
    weights = weights ** alpha
    if clip_max is not None:
        weights = np.clip(weights, 0, clip_max)
        
    return torch.from_numpy(weights).double()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg: Dict[str, Any] = yaml.safe_load(cfg_path.read_text())

    # -------------------------
    # 1. Environment Setup
    # -------------------------
    mode = str(cfg.get("data", {}).get("mode", "combined")).lower()
    if mode not in ("exid", "highd", "combined"):
        raise ValueError(f"data.mode must be one of exid/highd/combined, got: {mode}")

    seed = int(cfg.get("train", {}).get("seed", 42))
    set_seed(seed)

    want_cuda = (str(cfg.get("train", {}).get("device", "cuda")).lower() == "cuda")
    device = torch.device("cuda" if (want_cuda and torch.cuda.is_available()) else "cpu")

    print("==== Environment ====")
    print("torch:", torch.__version__)
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    # -------------------------
    # 2. Config & Features
    # -------------------------
    feat_cfg = cfg.get("features", {})
    use_ego_static = bool(feat_cfg.get("use_ego_static", True))
    use_nb_static = bool(feat_cfg.get("use_nb_static", True))
    use_neighbors = bool(cfg.get("model", {}).get("use_neighbors", True))

    # Scenario Sampling Check (Used for return_meta decision)
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
    # 3. Path Resolution
    # -------------------------
    paths = resolve_data_paths(cfg)
    tag = str(paths.get("tag", "unknown"))

    exid_dir = paths.get("exid_pt_dir", Path(f"./data/exiD/data_mmap/exid_{tag}"))
    highd_dir = paths.get("highd_pt_dir", Path(f"./data/highD/data_mmap/highd_{tag}"))
    
    splits_dir = Path("./data/combined/splits") if mode == "combined" else \
                 Path(f"./data/{'exiD' if mode=='exid' else 'highD'}/splits")

    print(f"[INFO] Data Dir (ExID): {exid_dir}")
    print(f"[INFO] Data Dir (HighD): {highd_dir}")
    print(f"[INFO] Splits Dir: {splits_dir}")

    ckpt_dir = resolve_path(cfg.get("train", {}).get("ckpt_dir", "ckpts"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    event_csv = ckpt_dir / "val_stratified_event.csv"
    state_csv = ckpt_dir / "val_stratified_state.csv"

    # -------------------------
    # 4. Stats Loading (Ablation Support)
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

    if stats is None:
        print("[WARN] Stats not found. Training without normalization.")
    else:
        print("[INFO] Stats loaded successfully.")

    # -------------------------
    # 5. Dataset Construction (Mmap)
    # -------------------------
    print(f"[INFO] Loading MmapDataset (Mode: {mode})...")
    
    # Load Indices (Global indices created by create_splits.py)
    try:
        train_idx = np.load(splits_dir / "train_indices.npy")
        val_idx = np.load(splits_dir / "val_indices.npy")
    except FileNotFoundError:
        raise FileNotFoundError(f"Split files not found in {splits_dir}. Run create_splits.py.")

    # Dataset kwargs
    train_kwargs = {
        "use_ego_static": use_ego_static,
        "use_nb_static": use_nb_static,
        "use_neighbors": use_neighbors,
        "stats": stats,
        "return_meta": use_scenario_sampling
    }
    
    val_kwargs = {
        "use_ego_static": use_ego_static,
        "use_nb_static": use_nb_static,
        "use_neighbors": use_neighbors,
        "stats": stats,
        "return_meta": True
    }

    if mode == "combined":
        # Train Sets
        tr_exid = MmapDataset(exid_dir, "exid", dataset_name="exid", **train_kwargs)
        tr_highd = MmapDataset(highd_dir, "highd", dataset_name="highd", **train_kwargs)
        full_train_ds = ConcatDataset([tr_exid, tr_highd])
        
        # Val Sets 
        va_exid = MmapDataset(exid_dir, "exid", dataset_name="exid", **val_kwargs)
        va_highd = MmapDataset(highd_dir, "highd", dataset_name="highd", **val_kwargs)
        full_val_ds = ConcatDataset([va_exid, va_highd])
        
    elif mode == "exid":
        full_train_ds = MmapDataset(exid_dir, "exid", dataset_name="exid", **train_kwargs)
        full_val_ds = MmapDataset(exid_dir, "exid", dataset_name="exid", **val_kwargs)
    else:
        full_train_ds = MmapDataset(highd_dir, "highd", dataset_name="highd", **train_kwargs)
        full_val_ds = MmapDataset(highd_dir, "highd", dataset_name="highd", **val_kwargs)

    # Subset Creation
    train_ds = Subset(full_train_ds, train_idx)
    val_ds = Subset(full_val_ds, val_idx)
    
    print(f"[INFO] Dataset Ready -> Train: {len(train_ds):,}, Val: {len(val_ds):,}")

    # -------------------------
    # 6. DataLoaders & Sampling
    # -------------------------
    batch_size = int(cfg.get("data", {}).get("batch_size", 128))
    num_workers = int(cfg.get("data", {}).get("num_workers", 16)) 

    # Sampling Config
    mode_key = str(sam_cfg.get("mode", "event")).lower() if sam_cfg else "event"
    alpha = float(sam_cfg.get("alpha", 0.5)) if sam_cfg else 0.5
    unknown_w = float(sam_cfg.get("unknown_weight", 0.0)) if sam_cfg else 0.0
    clip_max = sam_cfg.get("clip_max", None) if sam_cfg else None
    if clip_max is not None: clip_max = float(clip_max)

    train_sampler = None
    
    # [TRAIN SAMPLER] - Fast Implementation
    if use_scenario_sampling:
        print(f"[INFO] Scenario Sampling ON (Mode={mode_key}, Alpha={alpha})")
        print("       Calculating weights using Vectorized Pandas Merge (Fast)...")
        
        # 1. 라벨 CSV 로드 (DataFrame 형태)
        if isinstance(labels_cfg, str):
            labels_df = pd.read_csv(labels_cfg)
        elif isinstance(labels_cfg, dict):
            dfs = []
            if "exid" in labels_cfg: dfs.append(pd.read_csv(labels_cfg["exid"]))
            if "highd" in labels_cfg: dfs.append(pd.read_csv(labels_cfg["highd"]))
            labels_df = pd.concat(dfs, ignore_index=True)
        else:
            raise ValueError("Invalid scenario_labels config")

        # 필요한 컬럼만 선택하여 메모리 절약
        target_col = "event_label" if mode_key == "event" else "state_label"
        labels_df = labels_df[["recordingId", "trackId", "t0_frame", target_col]]

        # 2. 전체 데이터셋에 대한 가중치 계산 (exid, highd 각각)
        full_weights = None
        
        if mode == "combined":
            w_exid = compute_weights_fast(tr_exid, labels_df, mode_key, alpha, unknown_w, clip_max)
            w_highd = compute_weights_fast(tr_highd, labels_df, mode_key, alpha, unknown_w, clip_max)
            full_weights = torch.cat([w_exid, w_highd])
            
        elif mode == "exid":
            full_weights = compute_weights_fast(full_train_ds, labels_df, mode_key, alpha, unknown_w, clip_max)
            
        elif mode == "highd":
            full_weights = compute_weights_fast(full_train_ds, labels_df, mode_key, alpha, unknown_w, clip_max)

        # 3. Train Subset에 해당하는 가중치만 추출
        train_weights = full_weights[train_idx]
        
        print(f"[INFO] Weights Ready. Valid weights: {(train_weights > 0).sum()}/{len(train_weights)}")
        
        train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_ds), replacement=True)
        
    # DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, 
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_batch, prefetch_factor=4, persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate_batch, prefetch_factor=4, persistent_workers=(num_workers > 0),
    )

    # -------------------------
    # 7. Model / Optim / Sched
    # -------------------------
    model = build_model(cfg).to(device)

    lr = float(cfg.get("train", {}).get("lr", 3e-4))
    weight_decay = float(cfg.get("train", {}).get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    use_amp = bool(cfg.get("train", {}).get("use_amp", True)) and (device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    predict_delta = bool(cfg.get("model", {}).get("predict_delta", False))
    
    w_ade = float(cfg.get("train", {}).get("w_ade", 1.0)) 
    w_fde = float(cfg.get("train", {}).get("w_fde", 0.0))
    w_rmse = float(cfg.get("train", {}).get("w_rmse", 0.0)) 
    w_cls = float(cfg.get("train", {}).get("w_cls", 1.0))

    epochs = int(cfg.get("train", {}).get("epochs", 50))
    grad_clip_norm = float(cfg.get("train", {}).get("grad_clip_norm", 1.0))
    log_every = int(cfg.get("train", {}).get("log_every", 50))

    warmup_steps = int(cfg.get("train", {}).get("warmup_steps", 0))
    lr_schedule = str(cfg.get("train", {}).get("lr_schedule", "none"))
    total_steps = epochs * max(1, len(train_loader))
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps, lr_schedule)

    monitor = str(cfg.get("train", {}).get("monitor", "val_ade")).lower()
    best = float("inf")
    global_step = 0

    print("\n==== Train ====")
    print(f"mode        : {mode}")
    print(f"tag         : {tag}")
    print(f"ckpt_dir    : {ckpt_dir}")
    print(f"epochs      : {epochs}  bs={batch_size}")
    print(f"monitor     : {monitor}")

    # -------------------------
    # 8. Training Loop
    # -------------------------
    for ep in range(1, epochs + 1):
        print(f"\n===== Epoch {ep}/{epochs} =====")

        tr = train_one_epoch(
            model=model, loader=train_loader, device=device, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler, use_amp=use_amp, predict_delta=predict_delta,
            grad_clip_norm=grad_clip_norm, w_ade=w_ade, w_fde=w_fde, w_rmse=w_rmse, w_cls=w_cls,
            global_step=global_step, log_every=log_every, epoch=ep,
        )
        global_step = int(tr["global_step_end"])

        # Stratified Eval Check
        do_strat = bool(cfg.get("train", {}).get("stratified_eval", False))
        
        va = evaluate(
            model=model, loader=val_loader, device=device, use_amp=use_amp,
            predict_delta=predict_delta, w_ade=w_ade, w_fde=w_fde, w_rmse=w_rmse, w_cls=w_cls,
            labels_lut=(labels_lut if do_strat else None),
            save_event_path=(event_csv if (do_strat and labels_lut is not None) else None),
            save_state_path=(state_csv if (do_strat and labels_lut is not None) else None),
            epoch=(ep if (do_strat and labels_lut is not None) else None),
            data_hz = float(cfg.get("data", {}).get("hz", 0.0))
        )

        print(
            f"[epoch {ep}] "
            f"train: loss={tr['loss']:.4f} ADE={tr['ade']:.3f} | "
            f"val: loss={va['loss']:.4f} ADE={va['ade']:.3f} RMSE={va['rmse']:.3f} FDE={va['fde']:.3f}"
        )
        
        if monitor == "val_loss": score = va["loss"]
        elif monitor == "val_ade": score = va["ade"]
        elif monitor == "val_rmse": score = va["rmse"]
        elif monitor == "val_fde": score = va["fde"]
        else: score = va.get(monitor.replace("val_", ""), va["loss"])

        is_best = score < best
        if is_best: 
            best = score

        # 1. 공통 저장 데이터 구성
        save_dict = {
            "epoch": ep, 
            "global_step": global_step, 
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), 
            "cfg": cfg, 
            "best_monitor": best,
        }

        # 2. last.pt 저장
        torch.save(save_dict, ckpt_dir / "last.pt")

        # 3. best.pt 저장 
        if is_best:
            best_path = ckpt_dir / "best.pt"
            torch.save(save_dict, best_path)
            print(f"✅[CKPT] best -> {best_path} ({monitor}={best:.4f})")

        # 4. 100 epoch 단위 best 저장 (추가된 부분)
        if ep % 100 == 0:
            interval_best_path = ckpt_dir / f"best_{ep}.pt"
            torch.save(torch.load(ckpt_dir / "best.pt"), interval_best_path)
            print(f"💾[INTERVAL] Saved best model up to epoch {ep} as {interval_best_path.name}")

    print("\n[DONE] Training finished.")
    print(f"Best {monitor}: {best:.4f}")
    print(f"{best_path}")


if __name__ == "__main__":
    main()
