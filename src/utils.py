# src/utils.py
from __future__ import annotations
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from typing import Callable, Dict, Any, Optional
import time

def make_ttag(T: int, Tf: int, hz: int) -> str:
    return f"T{T}_Tf{Tf}_hz{hz}"

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def resolve_data_paths(cfg: dict) -> dict:
    T  = int(cfg["data"]["T"])
    Tf = int(cfg["data"]["Tf"])
    hz = int(cfg["data"]["hz"])
    tag = make_ttag(T, Tf, hz)

    roots = cfg["data"]["roots"]
    exid_dir = Path(roots["exid_root"])
    exid_pt_dir  = Path(roots["exid_pt_root"])  / f"exid_{tag}"
    exid_splits_dir = Path(roots["exid_root"]) / "splits"
    exid_stats_dir = exid_dir / "stats"
    
    highd_dir = Path(roots["highd_root"])
    highd_pt_dir = Path(roots["highd_pt_root"]) / f"highd_{tag}"
    highd_splits_dir = Path(roots["highd_root"]) / "splits"
    highd_stats_dir = highd_dir / "stats"

    combined_dir = Path(roots["combined_root"])
    combined_splits_dir = combined_dir / "splits"
    combined_stats_dir  = combined_dir / "stats"

    return {
        "tag": tag,
        "exid_dir": exid_dir,
        "exid_pt_dir": Path(exid_pt_dir),
        "exid_splits_dir": Path(exid_splits_dir),
        "exid_stats_dir": Path(exid_stats_dir),
        "highd_dir": highd_dir,
        "highd_pt_dir": Path(highd_pt_dir),
        "highd_splits_dir": Path(highd_splits_dir),
        "high_stats_dir": Path(highd_stats_dir),
        "combined_dir": combined_dir,
        "combined_splits_dir": Path(combined_splits_dir),
        "combined_stats_dir": Path(combined_stats_dir),
    }

def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    project_root = Path(__file__).resolve().parents[1]
    return (project_root / p).resolve()

def read_split_stems(split_file: Path) -> tuple[list[str], list[str]]:
    """
    Returns (exid_stems, highd_stems), where each stem is WITHOUT '.pt'
    """
    exid, highd = [], []
    for ln in split_file.read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if s.endswith(".pt"):
            s = s[:-3]  # drop ".pt"
        if s.startswith("exid_"):
            exid.append(s)
        elif s.startswith("highd_"):
            highd.append(s)
        else:
            raise ValueError(f"Unknown prefix in split line: {ln}")
    return exid, highd

def _to_int(x):
    try:
        if isinstance(x, torch.Tensor):
            return int(x.item())
    except Exception:
        pass
    try:
        if isinstance(x, np.generic):
            return int(x)
    except Exception:
        pass
    return int(x)

@torch.no_grad()
def measure_latency_ms(
    fn: Callable[[], Any],
    device: torch.device,
    iters: int = 200,
    warmup: int = 30,
) -> Dict[str, float]:
    """
    Measures latency of `fn()` in milliseconds.
    - `fn` should run the exact work you want to time (e.g., model forward + postprocess)
    - For CUDA uses torch.cuda.Event for accurate timing.
    Returns avg/p50/p90/p99 in ms (per-call).
    """
    # warmup
    for _ in range(max(0, warmup)):
        _ = fn()

    times_ms = []

    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

        for _ in range(max(1, iters)):
            starter.record()
            _ = fn()
            ender.record()
            torch.cuda.synchronize()
            times_ms.append(starter.elapsed_time(ender))
    else:
        for _ in range(max(1, iters)):
            t0 = time.perf_counter()
            _ = fn()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "avg_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
        "iters": float(len(arr)),
    }