#!/usr/bin/env python3
# scripts/train.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.amp import GradScaler

import yaml

from src.utils import set_seed, resolve_path, resolve_data_paths
from src.stats import compute_stats_if_needed, load_stats_npz_strict, assert_stats_match_batch_dims, make_stats_filename
from src.datasets.pt_dataset import PtWindowDataset
from src.datasets.collate import collate_batch
from src.models.build import build_model, build_scheduler
from src.engine import train_one_epoch, evaluate
from src.scenarios import load_window_labels_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg: Dict[str, Any] = yaml.safe_load(cfg_path.read_text())

    # -------------------------
    # mode / seed / device
    # -------------------------
    mode = str(cfg.get("data", {}).get("mode", "exid")).lower()  # exid | highd | combined
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
    # features
    # -------------------------
    feat_cfg = cfg.get("features", {})
    use_ego_static = bool(feat_cfg.get("use_ego_static", True))
    use_nb_static = bool(feat_cfg.get("use_nb_static", True))

    # -------------------------
    # paths (mode-aware)
    # -------------------------

    paths = resolve_data_paths(cfg)
    tag = str(paths.get("tag", "unknown"))

    exid_pt_dir = paths.get("exid_pt_dir", Path(f"./data/exiD/data_pt/exid_{tag}"))
    highd_pt_dir = paths.get("highd_pt_dir", Path(f"./data/highD/data_pt/highd_{tag}"))
    exid_splits_dir = paths.get("exid_splits_dir", Path("./data/exiD/splits"))
    highd_splits_dir = paths.get("highd_splits_dir", Path("./data/highD/splits"))
    exid_stats_dir = paths.get("exid_stats_dir", Path("./data/exiD/stats"))
    highd_stats_dir = paths.get("highd_stats_dir", Path("./data/highD/stats"))

    ckpt_dir = resolve_path(cfg.get("train", {}).get("ckpt_dir", "ckpts"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    event_csv = ckpt_dir / "val_stratified_event.csv"
    state_csv = ckpt_dir / "val_stratified_state.csv"

    batch_size = int(cfg.get("data", {}).get("batch_size", 128))
    num_workers = int(cfg.get("data", {}).get("num_workers", 8))

    # -------------------------
    # stats (subprocess only)
    # -------------------------
    stats_split = str(cfg.get("train", {}).get("stats_split", "train"))
    stats_fname = make_stats_filename(tag, use_ego_static, use_nb_static)

    if mode == "exid":
        stats_path=Path("./data/exiD/stats") / stats_fname
        compute_stats_if_needed(
            stats_path=stats_path,
            data_dir=exid_pt_dir,
            splits_dir=exid_splits_dir,
            stats_split="train",
            batch_size=batch_size,
            num_workers=num_workers,
            use_ego_static=use_ego_static,
            use_nb_static=use_nb_static,
        )
    elif mode == "highd":
        stats_path=Path("./data/highD/stats") / stats_fname
        compute_stats_if_needed(
            stats_path=stats_path,
            data_dir=highd_pt_dir,
            splits_dir=highd_splits_dir,
            stats_split="train",
            batch_size=batch_size,
            num_workers=num_workers,
            use_ego_static=use_ego_static,
            use_nb_static=use_nb_static,
        )
    else:
        stats_path=Path("./data/combined/stats") / stats_fname
        compute_stats_if_needed(
            stats_path=stats_path,
            data_dir=[exid_pt_dir, highd_pt_dir],
            splits_dir=[exid_splits_dir, highd_splits_dir],
            stats_split="train",
            batch_size=batch_size,
            num_workers=num_workers,
            use_ego_static=use_ego_static,
            use_nb_static=use_nb_static,
        )

    stats: Optional[Dict[str, torch.Tensor]] = None
    print(f"[INFO] Loading stats: {stats_path}")
    stats = load_stats_npz_strict(stats_path)

    if stats is not None:
        ego_dim_cfg = int(cfg["model"]["ego_dim"])
        nb_dim_cfg = int(cfg["model"]["nb_dim"])
        assert_stats_match_batch_dims(stats, ego_dim_cfg, nb_dim_cfg, stats_path)

    # -------------------------
    # datasets / loaders
    # -------------------------
    if mode == "exid":
        train_split = exid_splits_dir / "train.txt"
        val_split = exid_splits_dir / "val.txt"

        train_ds = PtWindowDataset(
            data_dir=exid_pt_dir,
            split_txt=train_split,
            stats=stats,
            return_meta=False,
            use_ego_static=use_ego_static,
            use_nb_static=use_nb_static,
            dataset_name="exid",
        )
        val_ds = PtWindowDataset(
            data_dir=exid_pt_dir,
            split_txt=val_split,
            stats=stats,
            return_meta=True,
            use_ego_static=use_ego_static,
            use_nb_static=use_nb_static,
            dataset_name="exid",
        )

    elif mode == "highd":
        train_split = highd_splits_dir / "train.txt"
        val_split = highd_splits_dir / "val.txt"

        train_ds = PtWindowDataset(
            data_dir=highd_pt_dir,
            split_txt=train_split,
            stats=stats,
            return_meta=False,
            use_ego_static=use_ego_static,
            use_nb_static=use_nb_static,
            dataset_name="highd",
        )
        val_ds = PtWindowDataset(
            data_dir=highd_pt_dir,
            split_txt=val_split,
            stats=stats,
            return_meta=True,
            use_ego_static=use_ego_static,
            use_nb_static=use_nb_static,
            dataset_name="highd",
        )

    else:
        exid_train = PtWindowDataset(
            data_dir=exid_pt_dir,
            split_txt=exid_splits_dir / "train.txt",
            stats=stats,
            return_meta=False,
            use_ego_static=use_ego_static,
            use_nb_static=use_nb_static,
            dataset_name="exid",
        )
        highd_train = PtWindowDataset(
            data_dir=highd_pt_dir,
            split_txt=highd_splits_dir / "train.txt",
            stats=stats,
            return_meta=False,
            use_ego_static=use_ego_static,
            use_nb_static=use_nb_static,
            dataset_name="highd",
        )
        train_ds = ConcatDataset([exid_train, highd_train])

        exid_val = PtWindowDataset(
            data_dir=exid_pt_dir,
            split_txt=exid_splits_dir / "val.txt",
            stats=stats,
            return_meta=True,
            use_ego_static=use_ego_static,
            use_nb_static=use_nb_static,
            dataset_name="exid",
        )
        highd_val = PtWindowDataset(
            data_dir=highd_pt_dir,
            split_txt=highd_splits_dir / "val.txt",
            stats=stats,
            return_meta=True,
            use_ego_static=use_ego_static,
            use_nb_static=use_nb_static,
            dataset_name="highd",
        )
        val_ds = ConcatDataset([exid_val, highd_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=collate_batch,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_batch,
        persistent_workers=(num_workers > 0),
    )

    # -------------------------
    # scenario labels
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

    # -------------------------
    # model / optim / sched
    # -------------------------
    model = build_model(cfg).to(device)

    lr = float(cfg.get("train", {}).get("lr", 3e-4))
    weight_decay = float(cfg.get("train", {}).get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    use_amp = bool(cfg.get("train", {}).get("use_amp", True)) and (device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    predict_delta = bool(cfg.get("model", {}).get("predict_delta", False))
    w_traj = float(cfg.get("train", {}).get("w_traj", 1.0))
    w_fde = float(cfg.get("train", {}).get("w_fde", 0.0))
    w_cls = float(cfg.get("train", {}).get("w_cls", 1.0))

    epochs = int(cfg.get("train", {}).get("epochs", 50))
    grad_clip_norm = float(cfg.get("train", {}).get("grad_clip_norm", 1.0))
    log_every = int(cfg.get("train", {}).get("log_every", 50))

    warmup_steps = int(cfg.get("train", {}).get("warmup_steps", 0))
    lr_schedule = str(cfg.get("train", {}).get("lr_schedule", "none"))
    total_steps = epochs * max(1, len(train_loader))
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps, lr_schedule)

    monitor = str(cfg.get("train", {}).get("monitor", "val_loss")).lower()  # val_loss | val_ade
    best = float("inf")
    global_step = 0

    print("==== Train ====")
    print(f"mode        : {mode}")
    print(f"tag         : {tag}")
    print(f"exid_pt_dir : {exid_pt_dir}")
    print(f"highd_pt_dir: {highd_pt_dir}")
    print(f"exid_splits : {exid_splits_dir}")
    print(f"highd_splits: {highd_splits_dir}")
    print(f"stats_path  : {stats_path}")
    print(f"ckpt_dir    : {ckpt_dir}")
    print(f"epochs      : {epochs}  bs={batch_size}  workers={num_workers}")
    print(f"use_amp     : {use_amp}  predict_delta={predict_delta}")
    print(f"monitor     : {monitor}  lr_schedule={lr_schedule} warmup_steps={warmup_steps}")

    # -------------------------
    # training loop
    # -------------------------
    for ep in range(1, epochs + 1):
        print(f"\n===== Epoch {ep}/{epochs} =====")

        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            use_amp=use_amp,
            predict_delta=predict_delta,
            grad_clip_norm=grad_clip_norm,
            w_traj=w_traj,
            w_fde=w_fde,
            w_cls=w_cls,
            global_step=global_step,
            log_every=log_every,
            epoch=ep,
        )
        global_step = int(tr["global_step_end"])

        va = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            use_amp=use_amp,
            predict_delta=predict_delta,
            w_traj=w_traj,
            w_fde=w_fde,
            w_cls=w_cls,
            labels_lut=labels_lut,
            save_event_path=event_csv if labels_lut is not None else None,
            save_state_path=state_csv if labels_lut is not None else None,
            epoch=ep if labels_lut is not None else None,
        )

        print(
            f"[epoch {ep}] "
            f"train: loss={tr['loss']:.4f} ADE={tr['ade']:.3f} FDE={tr['fde']:.3f} | "
            f"val: loss={va['loss']:.4f} ADE={va['ade']:.3f} FDE={va['fde']:.3f}"
        )

        score = va["loss"] if monitor == "val_loss" else va["ade"]
        is_best = score < best
        if is_best:
            best = score

        last_path = ckpt_dir / "last.pt"
        torch.save(
            {
                "epoch": ep,
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg": cfg,
                "best_monitor": best,
                "monitor_name": monitor,
            },
            last_path,
        )

        if is_best:
            best_path = ckpt_dir / "best.pt"
            torch.save(
                {
                    "epoch": ep,
                    "global_step": global_step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg": cfg,
                    "best_monitor": best,
                    "monitor_name": monitor,
                },
                best_path,
            )
            print(f"✅[CKPT] best -> {best_path} ({monitor}={best:.4f})")

    print("\n[DONE] Training finished.")
    print(f"Best {monitor}: {best:.4f}")
    print(f"Best ckpt: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()