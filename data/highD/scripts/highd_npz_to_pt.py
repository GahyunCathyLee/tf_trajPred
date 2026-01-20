#!/usr/bin/env python3
"""
Convert exiD window NPZs -> PT.

- Reads *.npz produced by exid_raw_to_npz.py
- Saves *.pt with the same keys (arrays -> torch tensors)
- Keeps object/string arrays (e.g., class_names) as-is (not converted to tensor)
- Optionally checks expected shapes

Example:
  python3 utils/exid_npz_to_pt.py \
    --npz_dir data/exid_npz \
    --pt_dir  data_pt/exid_T20_Tf50_hz10
"""

import argparse
from pathlib import Path
import numpy as np
import torch


def _to_tensor(v):
    # Keep string/object arrays as python objects
    if isinstance(v, np.ndarray) and v.dtype.kind in ("U", "S", "O"):
        return v
    if isinstance(v, np.ndarray):
        # np scalar arrays also handled here
        return torch.from_numpy(v)
    # numpy scalar
    if isinstance(v, (np.generic,)):
        return torch.tensor(v)
    return v


def convert_one(npz_path: Path, pt_path: Path, check_shapes: bool = True):
    data = np.load(npz_path, allow_pickle=True)

    out = {k: _to_tensor(data[k]) for k in data.files}

    if check_shapes:
        # Basic sanity checks (skip if missing)
        if "x_hist" in out and "y_fut" in out:
            x_hist = out["x_hist"]
            y_fut = out["y_fut"]
            assert x_hist.ndim == 3, f"x_hist must be (N,T,D), got {tuple(x_hist.shape)}"
            assert y_fut.ndim == 3, f"y_fut must be (N,Tf,2), got {tuple(y_fut.shape)}"
            assert y_fut.shape[-1] == 2, f"y_fut last dim must be 2, got {y_fut.shape[-1]}"

        if "nb_hist" in out:
            nb_hist = out["nb_hist"]
            assert nb_hist.ndim == 4, f"nb_hist must be (N,T,K,Dn), got {tuple(nb_hist.shape)}"

        if "nb_mask" in out:
            nb_mask = out["nb_mask"]
            assert nb_mask.ndim == 3, f"nb_mask must be (N,T,K), got {tuple(nb_mask.shape)}"

        if "ego_static" in out:
            ego_static = out["ego_static"]
            assert ego_static.ndim == 2, f"ego_static must be (N,Ds), got {tuple(ego_static.shape)}"

        if "nb_static" in out:
            nb_static = out["nb_static"]
            # exiD version: (N,T,K,Ds)
            assert nb_static.ndim == 4, f"nb_static must be (N,T,K,Ds), got {tuple(nb_static.shape)}"

    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, pt_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", type=str, required=True)
    ap.add_argument(
        "--pt_root",
        type=str,
        default="data_pt",
        help="Root directory for pt outputs",
    )
    ap.add_argument("--glob", type=str, default="*.npz")
    ap.add_argument("--no_check", action="store_true", help="Disable shape sanity checks")
    args = ap.parse_args()

    npz_dir = Path(args.npz_dir)
    pt_root = Path(args.pt_root)

    pt_dir = pt_root / npz_dir.name
    pt_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(npz_dir.glob(args.glob))
    if not npz_files:
        raise RuntimeError(f"No npz files found in {npz_dir} with pattern {args.glob}")

    for npz_path in npz_files:
        pt_path = pt_dir / (npz_path.stem + ".pt")
        convert_one(npz_path, pt_path, check_shapes=(not args.no_check))
        print(f"Converted: {npz_path.name} -> {pt_path.name}")

    print(f"[DONE] converted_files={len(npz_files)}")


if __name__ == "__main__":
    main()