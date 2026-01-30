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

def make_stats_filename(
    tag: str,
    use_ego_static: bool,
    use_nb_static: bool,
    use_lc: bool,
    use_lead: bool,
) -> str:
    """
    Naming rule:
      - all on -> {tag}.npz
      - otherwise append suffix for disabled toggles in fixed order:
          _e0  (ego_static off)
          _n0  (nb_static off)
          _lc0 (lane-change features off)
          _ld0 (lead/safety features off)
    """
    suffix = ""
    if not use_ego_static:
        suffix += "_e0"
    if not use_nb_static:
        suffix += "_n0"
    if not use_lc:
        suffix += "_lc0"
    if not use_lead:
        suffix += "_ld0"
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
    use_ego_static: bool,
    use_nb_static: bool,
    use_lc: bool,
    use_lead: bool,
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
    if use_lc:
        cmd.append("--use_lc")
    if use_lead:
        cmd.append("--use_lead")

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