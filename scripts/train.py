#!/usr/bin/env python3
# scripts/train.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, Subset
from torch.amp import GradScaler

import yaml

from src.utils import set_seed, resolve_path, resolve_data_paths
from src.stats import (
    compute_stats_if_needed,
    load_stats_npz_strict,
    assert_stats_match_batch_dims,
    make_stats_filename,
)
from src.datasets.pt_dataset import PtWindowDataset
from src.datasets.collate import collate_batch
from src.models.build import build_model, build_scheduler
from src.engine import train_one_epoch, evaluate
from src.scenarios import load_window_labels_csv, build_sample_weights


def _expected_dims(
    use_ego_static: bool,
    use_nb_static: bool,
    use_lc: bool,
    use_lead: bool,
) -> tuple[int, int]:
    """
    Based on your latest schema:
      ego_hist = 13
      ego_safety = 5 (use_lead)
      ego_static = 10 (use_ego_static)

      nb_hist kin = 6
      nb_lc = 3 (use_lc)
      nb_static = 10 (use_nb_static)

    -> ego_dim = 13 + (5 if use_lead else 0) + (10 if use_ego_static else 0)
    -> nb_dim  =  6 + (3 if use_lc else 0) + (10 if use_nb_static else 0)
    """
    ego_dim = 13 + (5 if use_lead else 0) + (10 if use_ego_static else 0)
    nb_dim = 6 + (3 if use_lc else 0) + (10 if use_nb_static else 0)
    return ego_dim, nb_dim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg: Dict[str, Any] = yaml.safe_load(cfg_path.read_text())

    # -------------------------
    # mode / seed / device
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
    # feature toggles
    # -------------------------
    feat_cfg = cfg.get("features", {})
    use_ego_static = bool(feat_cfg.get("use_ego_static", True))
    use_nb_static = bool(feat_cfg.get("use_nb_static", True))
    use_lc = bool(feat_cfg.get("use_lc", True))
    use_lead = bool(feat_cfg.get("use_lead", True))

    print("==== Feature Toggles ====")
    print(f"use_ego_static = {use_ego_static}")
    print(f"use_nb_static  = {use_nb_static}")
    print(f"use_lc         = {use_lc}")
    print(f"use_lead       = {use_lead}")

    # -------------------------
    # expected dims check (IMPORTANT)
    # -------------------------
    ego_dim_exp, nb_dim_exp = _expected_dims(use_ego_static, use_nb_static, use_lc, use_lead)

    ego_dim_cfg = int(cfg.get("model", {}).get("ego_dim", ego_dim_exp))
    nb_dim_cfg = int(cfg.get("model", {}).get("nb_dim", nb_dim_exp))

    if ego_dim_cfg != ego_dim_exp or nb_dim_cfg != nb_dim_exp:
        raise ValueError(
            "[DIM MISMATCH]\n"
            f"  expected ego_dim={ego_dim_exp}, nb_dim={nb_dim_exp} from toggles\n"
            f"  but cfg.model has ego_dim={ego_dim_cfg}, nb_dim={nb_dim_cfg}\n"
            "Fix: update YAML model.ego_dim / model.nb_dim to match toggles."
        )

    # -------------------------
    # paths
    # -------------------------
    paths = resolve_data_paths(cfg)
    tag = str(paths.get("tag", "unknown"))

    exid_pt_dir = paths.get("exid_pt_dir", Path(f"./data/exiD/data_pt/exid_{tag}"))
    highd_pt_dir = paths.get("highd_pt_dir", Path(f"./data/highD/data_pt/highd_{tag}"))

    exid_splits_dir = paths.get("exid_splits_dir", Path("./data/exiD/splits"))
    highd_splits_dir = paths.get("highd_splits_dir", Path("./data/highD/splits"))

    if mode == "exid":
        splits_index_dir = Path("./data/exiD/splits")
    elif mode == "highd":
        splits_index_dir = Path("./data/highD/splits")
    else:
        splits_index_dir = Path("./data/combined/splits")
    print(f"[INFO] Split indices will be loaded from: {splits_index_dir}")

    ckpt_dir = resolve_path(cfg.get("train", {}).get("ckpt_dir", "ckpts"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    event_csv = ckpt_dir / "val_stratified_event.csv"
    state_csv = ckpt_dir / "val_stratified_state.csv"

    batch_size = int(cfg.get("data", {}).get("batch_size", 128))
    num_workers = int(cfg.get("data", {}).get("num_workers", 8))

    # -------------------------
    # stats per-toggle
    # -------------------------
    stats_fname = make_stats_filename(tag, use_ego_static, use_nb_static, use_lc, use_lead)

    if mode == "exid":
        stats_path = Path("./data/exiD/stats") / stats_fname
        compute_stats_if_needed(
            stats_path=stats_path,
            data_dir=exid_pt_dir,
            splits_dir=exid_splits_dir,
            stats_split="train",
            batch_size=batch_size,
            num_workers=num_workers,
            use_ego_static=use_ego_static,
            use_nb_static=use_nb_static,
            use_lc=use_lc,
            use_lead=use_lead,
        )
    elif mode == "highd":
        stats_path = Path("./data/highD/stats") / stats_fname
        compute_stats_if_needed(
            stats_path=stats_path,
            data_dir=highd_pt_dir,
            splits_dir=highd_splits_dir,
            stats_split="train",
            batch_size=batch_size,
            num_workers=num_workers,
            use_ego_static=use_ego_static,
            use_nb_static=use_nb_static,
            use_lc=use_lc,
            use_lead=use_lead,
        )
    else:
        stats_path = Path("./data/combined/stats") / stats_fname
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        compute_stats_if_needed(
            stats_path=stats_path,
            data_dir=[exid_pt_dir, highd_pt_dir],
            splits_dir=[exid_splits_dir, highd_splits_dir],
            stats_split="train",
            batch_size=batch_size,
            num_workers=num_workers,
            use_ego_static=use_ego_static,
            use_nb_static=use_nb_static,
            use_lc=use_lc,
            use_lead=use_lead,
        )

    print(f"[INFO] Loading stats: {stats_path}")
    stats: Optional[Dict[str, torch.Tensor]] = load_stats_npz_strict(stats_path)
    assert_stats_match_batch_dims(stats, ego_dim_cfg, nb_dim_cfg, stats_path)

    # -------------------------
    # strat labels / scenario sampling
    # -------------------------
    labels_lut = None
    labels_cfg = cfg.get("data", {}).get("scenario_labels", None)
    if labels_cfg:
        if isinstance(labels_cfg, str):
            labels_lut = load_window_labels_csv(Path(labels_cfg))
        elif isinstance(labels_cfg, dict):
            merged: Dict[Any, Any] = {}
            if "exid" in labels_cfg:
                merged.update(load_window_labels_csv(Path(labels_cfg["exid"])))
            if "highd" in labels_cfg:
                merged.update(load_window_labels_csv(Path(labels_cfg["highd"])))
            labels_lut = merged

    sam_cfg = cfg.get("data", {}).get("scenario_sampling", None)
    use_scenario_sampling = bool(sam_cfg and labels_lut is not None)

    # -------------------------
    # dataset builder
    # -------------------------
    def make_ds(pt_dir: Path, split_txt: Optional[Path], ds_name: str):
        return PtWindowDataset(
            data_dir=pt_dir,
            split_txt=split_txt,
            stats=stats,
            return_meta=True,
            use_ego_static=use_ego_static,
            use_nb_static=use_nb_static,
            use_lc=use_lc,
            use_lead=use_lead,
            dataset_name=ds_name,
        )

    train_ds = None
    val_ds = None

    if use_scenario_sampling:
        print("\n[DATA-MODE] Scenario Sampling ON -> Loading ALL files and splitting by Index (.npy)")

        ds_exid_full = None
        if mode in ("exid", "combined"):
            ds_exid_full = make_ds(exid_pt_dir, split_txt=None, ds_name="exid")

        ds_highd_full = None
        if mode in ("highd", "combined"):
            ds_highd_full = make_ds(highd_pt_dir, split_txt=None, ds_name="highd")

        if mode == "combined":
            assert ds_exid_full is not None and ds_highd_full is not None
            full_ds = ConcatDataset([ds_exid_full, ds_highd_full])
        elif mode == "exid":
            assert ds_exid_full is not None
            full_ds = ds_exid_full
        else:
            assert ds_highd_full is not None
            full_ds = ds_highd_full

        try:
            train_idx = np.load(splits_index_dir / "train_indices.npy")
            val_idx = np.load(splits_index_dir / "val_indices.npy")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Scenario sampling is ON, but index files not found in {splits_index_dir}. "
                f"Run create_splits.py."
            ) from e

        train_ds = Subset(full_ds, train_idx)
        val_ds = Subset(full_ds, val_idx)
        print(f"[INFO] Subset Split Applied -> Train: {len(train_ds):,}, Val: {len(val_ds):,}")

    else:
        print("\n[DATA-MODE] Scenario Sampling OFF -> Loading files via train.txt / val.txt")

        def txt(split_dir: Path, split_name: str) -> Path:
            return split_dir / f"{split_name}.txt"

        if mode == "exid":
            train_ds = make_ds(exid_pt_dir, split_txt=txt(exid_splits_dir, "train"), ds_name="exid")
            val_ds = make_ds(exid_pt_dir, split_txt=txt(exid_splits_dir, "val"), ds_name="exid")
        elif mode == "highd":
            train_ds = make_ds(highd_pt_dir, split_txt=txt(highd_splits_dir, "train"), ds_name="highd")
            val_ds = make_ds(highd_pt_dir, split_txt=txt(highd_splits_dir, "val"), ds_name="highd")
        else:
            exid_tr = make_ds(exid_pt_dir, split_txt=txt(exid_splits_dir, "train"), ds_name="exid")
            highd_tr = make_ds(highd_pt_dir, split_txt=txt(highd_splits_dir, "train"), ds_name="highd")
            train_ds = ConcatDataset([exid_tr, highd_tr])

            exid_val = make_ds(exid_pt_dir, split_txt=txt(exid_splits_dir, "val"), ds_name="exid")
            highd_val = make_ds(highd_pt_dir, split_txt=txt(highd_splits_dir, "val"), ds_name="highd")
            val_ds = ConcatDataset([exid_val, highd_val])

        print(f"[INFO] File-based Load -> Train: {len(train_ds):,}, Val: {len(val_ds):,}")

    assert train_ds is not None and val_ds is not None

    # -------------------------
    # loaders
    # -------------------------
    mode_key = str(sam_cfg.get("mode", "event")).lower() if sam_cfg else "event"
    alpha = float(sam_cfg.get("alpha", 0.5)) if sam_cfg else 0.5
    unknown_w = float(sam_cfg.get("unknown_weight", 0.0)) if sam_cfg else 0.0
    clip_max = sam_cfg.get("clip_max", None) if sam_cfg else None
    if clip_max is not None:
        clip_max = float(clip_max)

    if use_scenario_sampling:
        train_weights = build_sample_weights(
            train_ds,
            labels_lut,
            mode=("event" if mode_key == "event" else "state"),
            alpha=alpha,
            unknown_weight=unknown_w,
            clip_max=clip_max,
            log=True,
        )
        train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=8,
            persistent_workers=(num_workers > 0),
            collate_fn=collate_batch,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=8,
            persistent_workers=(num_workers > 0),
            collate_fn=collate_batch,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        prefetch_factor=8,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_batch,
    )

    # -------------------------
    # model / optim
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

    monitor = str(cfg.get("train", {}).get("monitor", "val_loss")).lower()
    best = float("inf")
    global_step = 0

    print("\n==== Train ====")
    print(f"mode: {mode} | tag: {tag}")
    print(f"stats: {stats_path.name}")
    print(f"epochs={epochs} bs={batch_size} monitor={monitor}\n")

    # -------------------------
    # training loop
    # -------------------------
    for ep in range(1, epochs + 1):
        print(f"===== Epoch {ep}/{epochs} =====")
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
            w_ade=w_ade,
            w_fde=w_fde,
            w_rmse=w_rmse,
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
            w_ade=w_ade,
            w_fde=w_fde,
            w_rmse=w_rmse,
            w_cls=w_cls,
            labels_lut=(labels_lut if do_strat else None),
            save_event_path=(event_csv if (do_strat and labels_lut is not None) else None),
            save_state_path=(state_csv if (do_strat and labels_lut is not None) else None),
            epoch=(ep if (do_strat and labels_lut is not None) else None),
            data_hz=float(cfg.get("data", {}).get("hz", 0.0)),
        )

        print(
            f"[epoch {ep}] train loss={tr['loss']:.4f} ADE={tr['ade']:.3f} | "
            f"val loss={va['loss']:.4f} ADE={va['ade']:.3f} RMSE={va['rmse']:.3f} FDE={va['fde']:.3f}\n"
        )

        score = va["loss"] if monitor == "val_loss" else va.get(monitor.replace("val_", ""), va["loss"])
        is_best = score < best
        if is_best: 
            best = score

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
            print(f"✅[CKPT] best -> {best_path} ({monitor}={best:.4f})\n")

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
