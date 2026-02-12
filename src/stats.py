#src/stats.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union, Optional

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
    use_neighbors: bool,
    use_lc_state: bool = True,
    use_dxtime: bool = True,
    use_gate: bool = True,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Load master stats and slice them according to active toggles.
    """
    stats_path = stats_dir / "stats.npz"
    if not stats_path.exists():
        return None

    d = np.load(str(stats_path))
    
    # 1. Ego Stats (Assuming standard 13 dims)
    # If your stats were computed with use_lead (safety), verify indices. 
    # Usually dataset handles safety concat, but here we assume stats match base feature set.
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
        # Raw Loaded (Expected 9 dims: 6 kin + 3 extra)
        raw_n_mean = torch.from_numpy(d["dyn_nb_mean"])
        raw_n_std = torch.from_numpy(d["dyn_nb_std"])
        
        # Slicing Indices
        # Always take 0-6 (Kinematics)
        indices = list(range(6))
        
        # Check dims and append indices
        current_dim = raw_n_mean.shape[0]
        
        # Assuming saved order: [Kin(6), LC(1), DxTime(1), Gate(1)]
        if use_lc_state and current_dim > 6:
            indices.append(6)
        if use_dxtime and current_dim > 7:
            indices.append(7)
        if use_gate and current_dim > 8:
            indices.append(8)
            
        nb_mean = raw_n_mean[indices]
        nb_std = raw_n_std[indices]
        
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

def make_stats_filename(
    *,
    tag: str,
    use_neighbors: bool,
    use_ego_static: bool,
    use_nb_static: bool,
    use_lead: bool,
    use_lc_state: bool,
    use_dxtime: bool,
    use_gate: bool,
    nb_kin_mode: str = "pva",
) -> str:
    suffix = ""
    if not use_ego_static:
        suffix += "_e0"
    if not use_nb_static:
        suffix += "_n0"
    if not use_lead:
        suffix += "_ld0"

    if not use_neighbors:
        suffix += "_nbr0"

    if not use_lc_state:
        suffix += "_lcs0"
    if not use_dxtime:
        suffix += "_dxt0"
    if not use_gate:
        suffix += "_gt0"

    nb_kin_mode = str(nb_kin_mode).lower().strip()
    allowed = {"p","v","a","pv","pa","va","pva"}
    if nb_kin_mode not in allowed:
        raise ValueError(f"nb_kin_mode must be one of {sorted(allowed)}, got: {nb_kin_mode}")
    suffix += f"_{nb_kin_mode}"

    return f"{tag}{suffix}.npz"


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
    tag: str,        
    use_neighbors: bool,
    use_ego_static: bool,
    use_nb_static: bool,
    use_lead: bool,
    use_lc_state: bool,
    use_dxtime: bool,
    use_gate: bool,
    nb_kin_mode: str,
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

    compute_stats_py = root / "scripts" / "compute_stats_mmap.py"
    if not compute_stats_py.exists():
        raise FileNotFoundError(f"Missing: {compute_stats_py}")

    cmd: List[str] = [
        sys.executable, str(compute_stats_py),
        "--split", str(stats_split),
        "--out", str(stats_path),
        "--tag", str(tag),           
        "--batch_size", str(int(batch_size)),
        "--num_workers", str(int(num_workers)),
    ]

    for dd, sd in zip(data_dirs, splits_dirs):
        cmd += ["--data_dir", str(dd)]
        cmd += ["--splits_dir", str(sd)]    

    if use_neighbors:
        cmd.append("--use_neighbors")

    if use_ego_static:
        cmd.append("--use_ego_static")
    if use_nb_static:
        cmd.append("--use_nb_static")
    if use_lead:
        cmd.append("--use_lead")

    # Granular toggles
    if use_lc_state:
        cmd.append("--use_lc_state")
    if use_dxtime:
        cmd.append("--use_dxtime")
    if use_gate:
        cmd.append("--use_gate")

    cmd += ["--nb_kin_mode", str(nb_kin_mode)]

    print("[INFO] Auto-computing MMAP stats with command:")
    print("  " + " ".join(cmd))

    stats_path.parent.mkdir(parents=True, exist_ok=True)

    import subprocess
    r = subprocess.run(cmd, cwd=str(root))
    if r.returncode != 0:
        raise RuntimeError(f"compute_stats_mmap failed with return code {r.returncode}")

    if not stats_path.exists():
        raise RuntimeError(f"compute_stats_mmap finished but stats file not found: {stats_path}")

    print(f"[INFO] MMAP Stats generated: {stats_path}")


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
