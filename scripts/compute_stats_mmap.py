# scripts/compute_stats_mmap.py
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from src.datasets.mmap_dataset import MmapDataset


def _as_list_str(xs: Sequence[str]) -> List[str]:
    return list(xs) if isinstance(xs, (list, tuple)) else [str(xs)]


def load_split_indices(splits_dir: Path, split: str) -> np.ndarray:
    """
    Try common naming conventions for split indices.
    Preferred: {split}_indices.npy  (e.g., train_indices.npy)
    Fallbacks are supported for convenience.
    """
    candidates = [
        splits_dir / f"{split}_indices.npy",
        splits_dir / f"indices_{split}.npy",
        splits_dir / f"{split}.npy",
    ]
    for p in candidates:
        if p.exists():
            arr = np.load(p)
            if arr.dtype != np.int64 and arr.dtype != np.int32:
                arr = arr.astype(np.int64)
            return arr
    raise FileNotFoundError(
        f"[compute_stats_mmap] Cannot find split indices for '{split}' in {splits_dir}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def welford_merge(count: int, mean: np.ndarray, m2: np.ndarray, x: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Merge batch statistics into running Welford stats.
    x: (N, D)
    """
    if x.size == 0:
        return count, mean, m2

    x = x.astype(np.float64, copy=False)
    n = int(x.shape[0])
    bmean = x.mean(axis=0)
    bm2 = ((x - bmean) ** 2).sum(axis=0)

    if count == 0:
        return n, bmean, bm2

    delta = bmean - mean
    new_count = count + n
    new_mean = mean + delta * (n / new_count)
    new_m2 = m2 + bm2 + (delta ** 2) * (count * n / new_count)
    return new_count, new_mean, new_m2


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=4)

    # allow repeated args for combined
    ap.add_argument("--data_dir", action="append", required=True)
    ap.add_argument("--splits_dir", action="append", required=True)

    # toggles
    ap.add_argument("--use_neighbors", action="store_true")
    ap.add_argument("--use_ego_static", action="store_true")
    ap.add_argument("--use_nb_static", action="store_true")
    ap.add_argument("--use_lead", action="store_true")      # kept for interface consistency (MmapDataset doesn't use it directly)

    ap.add_argument("--use_lc_state", action="store_true")
    ap.add_argument("--use_dxtime", action="store_true")
    ap.add_argument("--use_gate", action="store_true")

    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data_dirs = [Path(p) for p in _as_list_str(args.data_dir)]
    splits_dirs = [Path(p) for p in _as_list_str(args.splits_dir)]
    if len(data_dirs) != len(splits_dirs):
        raise ValueError(
            f"[compute_stats_mmap] --data_dir and --splits_dir must have same count. "
            f"got data_dir={len(data_dirs)} splits_dir={len(splits_dirs)}"
        )

    # Build dataset(s)
    datasets = []
    for dd, sd in zip(data_dirs, splits_dirs):
        split_idx = load_split_indices(sd, args.split)
        ds = MmapDataset(
            data_dir=dd,
            split_indices=split_idx,
            stats=None,  # IMPORTANT: raw stats
            return_meta=False,
            use_ego_static=args.use_ego_static,
            use_nb_static=args.use_nb_static,
            use_neighbors=args.use_neighbors,
            use_lead=args.use_lead,
            use_lc_state=args.use_lc_state,
            use_dxtime=args.use_dxtime,
            use_gate=args.use_gate,
            dataset_name=None,
            is_pre_normalized=False,
        )
        datasets.append(ds)

    full_ds = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    dl = DataLoader(
        full_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
    )

    # Infer dims from first batch
    b0 = next(iter(dl))
    ego_dim = int(b0["x_ego"].shape[-1])
    nb_dim = int(b0["x_nb"].shape[-1])

    # Running stats init
    ego_count = 0
    nb_count = 0
    ego_mean = np.zeros((ego_dim,), dtype=np.float64)
    ego_m2 = np.zeros((ego_dim,), dtype=np.float64)
    nb_mean = np.zeros((nb_dim,), dtype=np.float64)
    nb_m2 = np.zeros((nb_dim,), dtype=np.float64)

    # Iterate
    for batch in dl:
        x_ego = batch["x_ego"].numpy()      # (B, T, De)
        x_nb = batch["x_nb"].numpy()        # (B, T, K, Dn)
        nb_mask = batch["nb_mask"].numpy()  # (B, T, K)

        # ego: always include all time steps
        ego_flat = x_ego.reshape(-1, ego_dim)
        ego_count, ego_mean, ego_m2 = welford_merge(ego_count, ego_mean, ego_m2, ego_flat)

        # nb: mask-based (only valid neighbors)
        if nb_dim > 0:
            B, T, K, Dn = x_nb.shape
            nb_flat = x_nb.reshape(B * T * K, Dn)
            mask_flat = nb_mask.reshape(B * T * K)

            # IMPORTANT: use mask to exclude padded neighbors
            nb_valid = nb_flat[mask_flat]
            if nb_valid.shape[0] > 0:
                nb_count, nb_mean, nb_m2 = welford_merge(nb_count, nb_mean, nb_m2, nb_valid)

    # Finalize ego std
    ego_var = ego_m2 / max(ego_count - 1, 1)
    ego_std = np.sqrt(np.maximum(ego_var, 1e-12)).astype(np.float32)

    # Finalize nb std
    if nb_count == 0:
        # Happens when use_neighbors=False (mask all false) or no valid neighbors in the entire split
        nb_mean_out = np.zeros((nb_dim,), dtype=np.float32)
        nb_std_out = np.ones((nb_dim,), dtype=np.float32)
    else:
        nb_var = nb_m2 / max(nb_count - 1, 1)
        nb_std_out = np.sqrt(np.maximum(nb_var, 1e-12)).astype(np.float32)
        nb_mean_out = nb_mean.astype(np.float32)

    # Save
    np.savez(
        str(out_path),
        ego_mean=ego_mean.astype(np.float32),
        ego_std=ego_std,
        nb_mean=nb_mean_out,
        nb_std=nb_std_out,
    )

    print(f"[OK] saved stats: {out_path}")
    print(f"     ego_dim={ego_dim}, nb_dim={nb_dim}, ego_count={ego_count}, nb_count={nb_count}")


if __name__ == "__main__":
    main()