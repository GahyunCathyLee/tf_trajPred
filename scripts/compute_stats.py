#!/usr/bin/env python3
# scripts/compute_stats.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PtWindowDatasetNoNorm(Dataset):
    """
    Minimal dataset for stats computation.
    Loads .pt files listed in split txt. No normalization applied.
    Supports optional concatenation of ego_static/nb_static if present.

    Required keys per .pt:
      x_hist  : (N,T,De)
      nb_hist : (N,T,K,Dn)
      nb_mask : (N,T,K) bool

    Optional:
      ego_static : (N,Ds_ego)
      nb_static  : (N,T,K,Ds_nb)  or (N,K,Ds_nb)
    """

    def __init__(self, data_dir: Path, split_txt: Path, use_ego_static: bool, use_nb_static: bool):
        self.data_dir = Path(data_dir)
        self.use_ego_static = use_ego_static
        self.use_nb_static = use_nb_static

        names = [ln.strip() for ln in split_txt.read_text().splitlines() if ln.strip()]
        self.files = [self.data_dir / n for n in names]
        missing = [str(p) for p in self.files if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing files in split:\n" + "\n".join(missing))

        self.recs: List[Dict[str, torch.Tensor]] = []
        self.sizes: List[int] = []
        self.prefix: List[int] = [0]

        for p in self.files:
            d = torch.load(p, map_location="cpu", weights_only=False)
            for k in ["x_hist", "nb_hist", "nb_mask"]:
                if k not in d:
                    raise KeyError(f"{p.name} missing key '{k}'. keys={list(d.keys())}")
            n = int(d["x_hist"].shape[0])
            self.recs.append(d)
            self.sizes.append(n)
            self.prefix.append(self.prefix[-1] + n)

    def __len__(self) -> int:
        return self.prefix[-1]

    def _locate(self, idx: int) -> Tuple[int, int]:
        lo, hi = 0, len(self.sizes) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.prefix[mid] <= idx < self.prefix[mid + 1]:
                return mid, idx - self.prefix[mid]
            if idx < self.prefix[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        raise IndexError(idx)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec_i, local_i = self._locate(idx)
        d = self.recs[rec_i]
        
        file_path = self.files[rec_i]

        x = d["x_hist"][local_i].to(torch.float32)
        nb = d["nb_hist"][local_i].to(torch.float32)
        mask = d["nb_mask"][local_i].bool()

        if not torch.isfinite(x).all():
            print(f"[BAD DATA] NaN/Inf found in x_hist! File: {file_path}, Index: {local_i}")
            
        if not torch.isfinite(nb[mask]).all():
            print(f"[BAD DATA] NaN/Inf found in nb_hist! File: {file_path}, Index: {local_i}")
        # ------------------------

        if self.use_ego_static and ("ego_static" in d):
            es = d["ego_static"][local_i].to(torch.float32).view(1, -1)
            if not torch.isfinite(es).all():
                print(f"[BAD DATA] NaN/Inf in ego_static! File: {file_path}")
            x = torch.cat([x, es.expand(x.shape[0], -1)], dim=-1)

        if self.use_nb_static and ("nb_static" in d):
            ns = d["nb_static"][local_i].to(torch.float32)

            if ns.dim() == 2:
                ns = ns.unsqueeze(0).expand(nb.shape[0], -1, -1)
            elif ns.dim() == 3:
                pass
            elif ns.dim() == 4:
                if ns.shape[0] == 1:
                    ns = ns.squeeze(0)
                else:
                    raise RuntimeError(f"Unexpected nb_static 4D shape: {tuple(ns.shape)}")
            else:
                raise RuntimeError(f"Unexpected nb_static shape: {tuple(ns.shape)}")

            nb = torch.cat([nb, ns], dim=-1)

        return {"x_ego": x, "x_nb": nb, "nb_mask": mask}


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "x_ego": torch.stack([b["x_ego"] for b in batch], dim=0),       # (B,T,De)
        "x_nb": torch.stack([b["x_nb"] for b in batch], dim=0),         # (B,T,K,Dn)
        "nb_mask": torch.stack([b["nb_mask"] for b in batch], dim=0),   # (B,T,K)
    }


def welford_merge(count: int, mean: np.ndarray, m2: np.ndarray, x: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Vectorized Welford update for a batch x of shape (N, D)
    """
    if x.size == 0:
        return count, mean, m2

    x = x.astype(np.float64)
    n = x.shape[0]
    batch_mean = x.mean(axis=0)
    batch_m2 = ((x - batch_mean) ** 2).sum(axis=0)

    if count == 0:
        return n, batch_mean, batch_m2

    delta = batch_mean - mean
    new_count = count + n
    new_mean = mean + delta * (n / new_count)
    new_m2 = m2 + batch_m2 + (delta ** 2) * (count * n / new_count)
    return new_count, new_mean, new_m2


def _ensure_pairs(data_dirs: List[str], splits_dirs: List[str]) -> List[Tuple[Path, Path]]:
    if len(data_dirs) != len(splits_dirs):
        raise ValueError(
            f"--data_dir and --splits_dir must have same count. "
            f"got data_dir={len(data_dirs)}, splits_dir={len(splits_dirs)}"
        )
    pairs: List[Tuple[Path, Path]] = []
    for d, s in zip(data_dirs, splits_dirs):
        pairs.append((Path(d), Path(s)))
    return pairs


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()

    # allow multi-use: --data_dir A --data_dir B ...
    ap.add_argument("--data_dir", type=str, action="append", required=True)
    ap.add_argument("--splits_dir", type=str, action="append", required=True)

    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--use_ego_static", action="store_true")
    ap.add_argument("--use_nb_static", action="store_true")

    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    pairs = _ensure_pairs(args.data_dir, args.splits_dir)

    ego_dim: int | None = None
    nb_dim: int | None = None

    ego_count = 0
    ego_mean = None
    ego_m2 = None

    nb_count = 0
    nb_mean = None
    nb_m2 = None

    total_files = 0
    total_windows = 0

    for data_dir, splits_dir in pairs:
        split_file = Path(splits_dir) / f"{args.split}.txt"
        ds = PtWindowDatasetNoNorm(
            data_dir=Path(data_dir),
            split_txt=split_file,
            use_ego_static=args.use_ego_static,
            use_nb_static=args.use_nb_static,
        )

        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate,
            persistent_workers=(args.num_workers > 0),
        )

        # infer dims (and ensure non-empty)
        try:
            b0 = next(iter(dl))
        except StopIteration:
            raise RuntimeError(f"Empty split loader: data_dir={data_dir}, split={split_file}")

        cur_ego_dim = int(b0["x_ego"].shape[-1])
        cur_nb_dim = int(b0["x_nb"].shape[-1])

        if ego_dim is None:
            ego_dim = cur_ego_dim
            nb_dim = cur_nb_dim
            ego_mean = np.zeros((ego_dim,), dtype=np.float64)
            ego_m2 = np.zeros((ego_dim,), dtype=np.float64)
            nb_mean = np.zeros((nb_dim,), dtype=np.float64)
            nb_m2 = np.zeros((nb_dim,), dtype=np.float64)
        else:
            if cur_ego_dim != ego_dim or cur_nb_dim != nb_dim:
                raise RuntimeError(
                    f"[STATS DIM MISMATCH] got ego_dim={cur_ego_dim}, nb_dim={cur_nb_dim} "
                    f"but expected ego_dim={ego_dim}, nb_dim={nb_dim} "
                    f"(data_dir={data_dir}, split={split_file})"
                )

        total_files += len(ds.files)
        total_windows += len(ds)

        for batch in dl:
            x_ego = batch["x_ego"].numpy()       # (B,T,De)
            x_nb = batch["x_nb"].numpy()         # (B,T,K,Dn)
            nb_mask = batch["nb_mask"].numpy()   # (B,T,K)

            # Ego: all timesteps
            ego_flat = x_ego.reshape(-1, ego_dim)
            ego_count, ego_mean, ego_m2 = welford_merge(ego_count, ego_mean, ego_m2, ego_flat)

            # Neighbors: only masked valid
            B, T, K, Dn = x_nb.shape
            nb_flat = x_nb.reshape(B * T * K, Dn)
            mask_flat = nb_mask.reshape(B * T * K)
            nb_valid = nb_flat[mask_flat]
            nb_count, nb_mean, nb_m2 = welford_merge(nb_count, nb_mean, nb_m2, nb_valid)

    assert ego_dim is not None and nb_dim is not None
    assert ego_mean is not None and ego_m2 is not None
    assert nb_mean is not None and nb_m2 is not None

    ego_var = ego_m2 / max(ego_count - 1, 1)
    nb_var = nb_m2 / max(nb_count - 1, 1)

    ego_std = np.sqrt(np.maximum(ego_var, 1e-12)).astype(np.float32)
    nb_std = np.sqrt(np.maximum(nb_var, 1e-12)).astype(np.float32)

    print("\n[INSPECTION] Checking for dangerous low-variance features...")
    
    threshold = 1e-3 
    
    # ---------------------------------------------------------
    # Low Variance Feature 처리
    # ---------------------------------------------------------
    print("\n[INSPECTION] Checking and FIXING dangerous low-variance features...")
    
    # 1. Ego Features 처리
    low_std_ego_indices = np.where(ego_std < threshold)[0]
    if len(low_std_ego_indices) > 0:
        print(f"⚠️  WARNING: Found {len(low_std_ego_indices)} Ego features with low std. Forcing mean=0, std=1.")
        for idx in low_std_ego_indices:
            # 학습 데이터에 값이 거의 없거나(all zeros), one-hot이라 분산이 작음
            # -> 정규화를 끄기 위해 mean=0, std=1로 덮어씌움
            old_m, old_s = ego_mean[idx], ego_std[idx]
            ego_mean[idx] = 0.0
            ego_std[idx]  = 1.0
            print(f"    - Fixed Ego Dim {idx}: (mean={old_m:.6f}, std={old_s:.6f}) -> (mean=0.0, std=1.0)")
            
    # 2. Neighbor Features 처리
    low_std_nb_indices = np.where(nb_std < threshold)[0]
    if len(low_std_nb_indices) > 0:
        print(f"⚠️  WARNING: Found {len(low_std_nb_indices)} Nb features with low std. Forcing mean=0, std=1.")
        for idx in low_std_nb_indices:
            old_m, old_s = nb_mean[idx], nb_std[idx]
            nb_mean[idx] = 0.0
            nb_std[idx]  = 1.0
            print(f"    - Fixed Nb Dim {idx}: (mean={old_m:.6f}, std={old_s:.6f}) -> (mean=0.0, std=1.0)")

    if len(low_std_ego_indices) == 0 and len(low_std_nb_indices) == 0:
        print("✅ All features have safe variance levels.")
    else:
        print("✅ Fixed low-variance features to skip normalization.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        ego_mean=ego_mean.astype(np.float32),
        ego_std=ego_std,
        nb_mean=nb_mean.astype(np.float32),
        nb_std=nb_std,
        ego_count=np.array([ego_count], dtype=np.int64),
        nb_count=np.array([nb_count], dtype=np.int64),
    )

    print(f"[DONE] saved stats -> {out_path}")
    print(f"  inputs={len(pairs)} roots, files={total_files}, windows={total_windows}")
    print(f"  ego_count={ego_count}, nb_count={nb_count}, ego_dim={ego_dim}, nb_dim={nb_dim}")


if __name__ == "__main__":
    main()