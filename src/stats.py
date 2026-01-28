#src/stats.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union

import numpy as np
import torch
import sys


def load_stats_npz(stats_path: Path) -> Dict[str, torch.Tensor]:
    d = np.load(str(stats_path))
    return {k: torch.from_numpy(d[k]) for k in d.files}


def load_stats_npz_strict(stats_path: Path) -> Dict[str, torch.Tensor]:
    stats = load_stats_npz(stats_path)
    for k in ["ego_mean", "ego_std", "nb_mean", "nb_std"]:
        if k not in stats:
            raise RuntimeError(f"[STATS] missing key '{k}' in {stats_path}")
    return stats

def load_stats_for_ablation(
    stats_dir: Path, 
    use_ego_static: bool, 
    use_nb_static: bool, 
    use_neighbors: bool
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Ablation Config에 따라 Dynamic/Static 통계를 조립하여 반환.
    """
    stats_path = stats_dir / "stats.npz"
    if not stats_path.exists():
        return None

    d = np.load(str(stats_path))
    
    # 1. Ego Stats
    ego_mean = torch.from_numpy(d["dyn_ego_mean"])
    ego_std = torch.from_numpy(d["dyn_ego_std"])
    
    if use_ego_static and "stat_ego_mean" in d:
        es_mean = torch.from_numpy(d["stat_ego_mean"])
        es_std = torch.from_numpy(d["stat_ego_std"])
        ego_mean = torch.cat([ego_mean, es_mean], dim=0)
        ego_std = torch.cat([ego_std, es_std], dim=0)

    # 2. Neighbor Stats
    nb_mean = None
    nb_std = None
    
    if use_neighbors:
        nb_mean = torch.from_numpy(d["dyn_nb_mean"])
        nb_std = torch.from_numpy(d["dyn_nb_std"])
        
        if use_nb_static and "stat_nb_mean" in d:
            ns_mean = torch.from_numpy(d["stat_nb_mean"])
            ns_std = torch.from_numpy(d["stat_nb_std"])
            nb_mean = torch.cat([nb_mean, ns_mean], dim=0)
            nb_std = torch.cat([nb_std, ns_std], dim=0)

    return {
        "ego_mean": ego_mean,
        "ego_std": ego_std,
        "nb_mean": nb_mean,
        "nb_std": nb_std
    }

def make_stats_filename(tag: str, use_ego_static: bool, use_nb_static: bool) -> str:
    """
    Naming rule:
      - es=1 & ns=1 -> {tag}.npz
      - es=0 & ns=1 -> {tag}_e0.npz
      - es=1 & ns=0 -> {tag}_n0.npz
      - es=0 & ns=0 -> {tag}_e0n0.npz
    """
    if use_ego_static and use_nb_static:
        return f"{tag}.npz"
    if (not use_ego_static) and use_nb_static:
        return f"{tag}_e0.npz"
    if use_ego_static and (not use_nb_static):
        return f"{tag}_n0.npz"
    return f"{tag}_e0n0.npz"


def _as_list(x: Union[Path, Sequence[Path]]) -> List[Path]:
    if isinstance(x, (list, tuple)):
        return [Path(p) for p in x]
    return [Path(x)]

def compute_stats_if_needed(
    *,
    stats_path: Path,
    data_dir: Union[Path, Sequence[Path]],
    splits_dir: Union[Path, Sequence[Path]],
    stats_split: str,
    batch_size: int,
    num_workers: int,
    use_ego_static: bool,
    use_nb_static: bool,
) -> None:

    if stats_path.exists():
        return

    print(f"[WARN] Stats not found: {stats_path}")

    data_dirs = _as_list(data_dir)
    splits_dirs = _as_list(splits_dir)
    if len(data_dirs) != len(splits_dirs):
        raise ValueError(
            f"[STATS] data_dir and splits_dir must have same count. "
            f"got data_dir={len(data_dirs)}, splits_dir={len(splits_dirs)}"
        )

    root = Path(__file__).resolve().parents[1]
    compute_stats_py = root / "scripts" / "compute_stats.py"
    if not compute_stats_py.exists():
        raise FileNotFoundError(f"Missing: {compute_stats_py}")

    cmd: List[str] = [
        sys.executable, str(compute_stats_py),
        "--split", str(stats_split),
        "--out", str(stats_path),
        "--batch_size", str(int(batch_size)),
        "--num_workers", str(int(num_workers)),
    ]

    for dd, sd in zip(data_dirs, splits_dirs):
        cmd += ["--data_dir", str(dd)]
        cmd += ["--splits_dir", str(sd)]

    if use_ego_static:
        cmd.append("--use_ego_static")
    if use_nb_static:
        cmd.append("--use_nb_static")

    print("[INFO] Auto-computing stats with command:")
    print("  " + " ".join(cmd))

    stats_path.parent.mkdir(parents=True, exist_ok=True)

    import subprocess
    r = subprocess.run(cmd, cwd=str(root))
    if r.returncode != 0:
        raise RuntimeError(f"compute_stats failed with return code {r.returncode}")

    if not stats_path.exists():
        raise RuntimeError(f"compute_stats finished but stats file not found: {stats_path}")

    print(f"[INFO] Stats generated: {stats_path}")


def assert_stats_match_batch_dims(
    stats: Dict[str, torch.Tensor],
    ego_dim: int,
    nb_dim: int,
    stats_path: Path,
) -> None:
    ego_mean = stats["ego_mean"]
    nb_mean = stats["nb_mean"]
    if int(ego_mean.numel()) != int(ego_dim):
        raise RuntimeError(
            f"[STATS MISMATCH] ego_mean dim={ego_mean.numel()} but ego_dim={ego_dim} "
            f"(stats={stats_path})"
        )
    if int(nb_mean.numel()) != int(nb_dim):
        raise RuntimeError(
            f"[STATS MISMATCH] nb_mean dim={nb_mean.numel()} but nb_dim={nb_dim} "
            f"(stats={stats_path})"
        )
