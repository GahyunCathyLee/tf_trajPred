#!/usr/bin/env python3
# scripts/train.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torch.amp import GradScaler

import yaml

from src.utils import set_seed, resolve_path, resolve_data_paths
from src.stats import compute_stats_if_needed, load_stats_npz_strict, assert_stats_match_batch_dims, make_stats_filename
from src.datasets.pt_dataset import PtWindowDataset
from src.datasets.collate import collate_batch
from src.models.build import build_model, build_scheduler
from src.engine import train_one_epoch, evaluate
from src.scenarios import load_window_labels_csv, build_sample_weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg: Dict[str, Any] = yaml.safe_load(cfg_path.read_text())

    # -------------------------
    # mode / seed / device
    # -------------------------
    mode = str(cfg.get("data", {}).get("mode", "exid")).lower()
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
            return_meta=True,
            use_ego_static=use_ego_static,
            use_nb_static=use_nb_static,
            dataset_name="exid",
        )
        val_ds = PtWindowDataset(
            data_dir=exid_pt_dir,
            split_txt=val_split,
            stats=stats,
            return_meta=False,
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
            return_meta=True,
            use_ego_static=use_ego_static,
            use_nb_static=use_nb_static,
            dataset_name="highd",
        )
        val_ds = PtWindowDataset(
            data_dir=highd_pt_dir,
            split_txt=val_split,
            stats=stats,
            return_meta=False,
            use_ego_static=use_ego_static,
            use_nb_static=use_nb_static,
            dataset_name="highd",
        )

    else:
        exid_train = PtWindowDataset(
            data_dir=exid_pt_dir,
            split_txt=exid_splits_dir / "train.txt",
            stats=stats,
            return_meta=True,
            use_ego_static=use_ego_static,
            use_nb_static=use_nb_static,
            dataset_name="exid",
        )
        highd_train = PtWindowDataset(
            data_dir=highd_pt_dir,
            split_txt=highd_splits_dir / "train.txt",
            stats=stats,
            return_meta=True,
            use_ego_static=use_ego_static,
            use_nb_static=use_nb_static,
            dataset_name="exid",
        )
        train_ds = ConcatDataset([exid_train, highd_train])

        exid_val = PtWindowDataset(
            data_dir=exid_pt_dir,
            split_txt=exid_splits_dir / "val.txt",
            stats=stats,
            return_meta=False,
            use_ego_static=use_ego_static,
            use_nb_static=use_nb_static,
            dataset_name="exid",
        )
        highd_val = PtWindowDataset(
            data_dir=highd_pt_dir,
            split_txt=highd_splits_dir / "val.txt",
            stats=stats,
            return_meta=False,
            use_ego_static=use_ego_static,
            use_nb_static=use_nb_static,
            dataset_name="highd",
        )
        val_ds = ConcatDataset([exid_val, highd_val])

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

    # ---- scenario-aware sampling config ----
    sam_cfg = cfg.get("data", {}).get("scenario_sampling", None)
    use_scenario_sampling = bool(sam_cfg and labels_lut is not None)

    if use_scenario_sampling:
        mode_key = str(sam_cfg.get("mode", "event")).lower()
        alpha = float(sam_cfg.get("alpha", 0.5))
        unknown_w = float(sam_cfg.get("unknown_weight", 0.0))
        clip_max = sam_cfg.get("clip_max", None)
        clip_max = float(clip_max) if clip_max is not None else None

        weights = build_sample_weights(
            train_ds, labels_lut,
            mode=("event" if mode_key == "event" else "state"),
            alpha=alpha,
            unknown_weight=unknown_w,
            clip_max=clip_max,
            log = True
        )

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_ds),
            replacement=True
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=8,
            persistent_workers=True,
            collate_fn=collate_batch,
        )
        print(f"[INFO] Scenario-aware sampling enabled: mode={mode_key} alpha={alpha}")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_batch,
            prefetch_factor=8,
            persistent_workers=(num_workers > 0),
        )
        if sam_cfg and labels_lut is None:
            print("[WARN] scenario_sampling set but labels_lut not loaded -> disabled")

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        prefetch_factor=8,
        collate_fn=collate_batch,
        persistent_workers=(num_workers > 0),
    )


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

    monitor = str(cfg.get("train", {}).get("monitor", "val_loss")).lower()
    best = float("inf")
    global_step = 0

    print("==== Train ====")
    print(f"mode        : {mode}")
    print(f"tag         : {tag}")
    print(f"ckpt_dir    : {ckpt_dir}")
    print(f"epochs      : {epochs}  bs={batch_size}")
    print(f"monitor     : {monitor}")

    # -------------------------
    # training loop
    # -------------------------
    for ep in range(1, epochs + 1):
        print(f"\\n===== Epoch {ep}/{epochs} =====")

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

        do_strat = bool(cfg.get("train", {}).get("stratified_eval", False))

        va = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            use_amp=use_amp,
            predict_delta=predict_delta,
            w_traj=w_traj,
            w_fde=w_fde,
            w_cls=w_cls,
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
        
        # [수정] Monitor Selection Logic
        if monitor == "val_loss":
            score = va["loss"]
        elif monitor == "val_ade":
            score = va["ade"]
        elif monitor == "val_rmse":
            score = va["rmse"]
        elif monitor == "val_fde":
            score = va["fde"]
        else:
            # support keys like "val_rmse_3s" -> "rmse_3s"
            key = monitor.replace("val_", "")
            if key in va:
                score = va[key]
            else:
                # fallback
                score = va["loss"]
                print(f"[WARN] Monitor '{monitor}' metric not found in validation results. Fallback to val_loss.")

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

    print("\\n[DONE] Training finished.")
    print(f"Best {monitor}: {best:.4f}")


if __name__ == "__main__":
    main()
