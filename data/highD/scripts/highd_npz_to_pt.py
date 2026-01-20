#!/usr/bin/env python3
"""highd_npz_to_pt.py

Convert highD window NPZs -> PT.

This is a good place to move *per-sample* work out of the training DataLoader.
With the options below, you can:
  - cast arrays to the final dtypes once (float32 / bool)
  - concatenate static features into x_hist / nb_hist once
  - precompute x_last_abs once

These changes make the Dataset __getitem__ mostly slicing (much faster).

Typical usage (recommended for faster training):
  python3 scripts/highd_npz_to_pt.py \
    --npz_dir data_npz/highd_T2_Tf5_hz5_flipXY_all \
    --pt_root data_pt \
    --concat_static --save_x_last_abs
"""

import argparse
from pathlib import Path
import numpy as np
import torch


def _to_tensor(v):
    if isinstance(v, np.ndarray) and v.dtype.kind in ("U", "S", "O"):
        return v
    if isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    if isinstance(v, (np.generic,)):
        return torch.tensor(v)
    return v


FLOAT_KEYS_DEFAULT = {
    "x_hist",
    "y_fut",
    "nb_hist",
    "ego_static",
    "nb_static",
    "y_fut_vel",
    "y_fut_acc",
}


def _as_float32(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.float32 else x.to(torch.float32)


def _as_bool(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.bool else x.to(torch.bool)


def _maybe_concat_static(out: dict) -> None:
    if "x_hist" in out and "ego_static" in out:
        x_hist = out["x_hist"]
        ego_static = out["ego_static"]
        if not (isinstance(x_hist, torch.Tensor) and isinstance(ego_static, torch.Tensor)):
            raise RuntimeError("x_hist/ego_static must be tensors to concat")
        if x_hist.ndim != 3 or ego_static.ndim != 2:
            raise RuntimeError(
                f"Bad shapes for concat: x_hist={tuple(x_hist.shape)}, ego_static={tuple(ego_static.shape)}"
            )
        n, t, _ = x_hist.shape
        ego_static_t = ego_static.unsqueeze(1).expand(n, t, -1)
        out["x_hist"] = torch.cat([x_hist, ego_static_t], dim=-1)

    if "nb_hist" in out and "nb_static" in out:
        nb_hist = out["nb_hist"]
        nb_static = out["nb_static"]
        if not (isinstance(nb_hist, torch.Tensor) and isinstance(nb_static, torch.Tensor)):
            raise RuntimeError("nb_hist/nb_static must be tensors to concat")
        if nb_hist.ndim != 4:
            raise RuntimeError(f"nb_hist must be (N,T,K,Dn), got {tuple(nb_hist.shape)}")
        n, t, k, _ = nb_hist.shape
        if nb_static.ndim == 3:
            nb_static_t = nb_static.unsqueeze(1).expand(n, t, k, -1)
        elif nb_static.ndim == 4:
            nb_static_t = nb_static
        else:
            raise RuntimeError(f"nb_static must be (N,K,Ds) or (N,T,K,Ds), got {tuple(nb_static.shape)}")
        out["nb_hist"] = torch.cat([nb_hist, nb_static_t], dim=-1)


def _maybe_make_x_last_abs(out: dict) -> None:
    if "x_hist" not in out:
        return
    x_hist = out["x_hist"]
    if not isinstance(x_hist, torch.Tensor) or x_hist.ndim != 3 or x_hist.shape[-1] < 2:
        return
    out["x_last_abs"] = x_hist[:, -1, 0:2].clone()


def convert_one(
    npz_path: Path,
    pt_path: Path,
    check_shapes: bool = True,
    cast_float32: bool = True,
    cast_mask_bool: bool = True,
    concat_static: bool = True,
    remove_static_keys: bool = True,
    save_x_last_abs: bool = True,
):
    data = np.load(npz_path, allow_pickle=True)
    out = {k: _to_tensor(data[k]) for k in data.files}

    if cast_float32:
        for k in list(out.keys()):
            if k in FLOAT_KEYS_DEFAULT and isinstance(out[k], torch.Tensor) and out[k].is_floating_point():
                out[k] = _as_float32(out[k])
    if cast_mask_bool and "nb_mask" in out and isinstance(out["nb_mask"], torch.Tensor):
        out["nb_mask"] = _as_bool(out["nb_mask"])

    if concat_static:
        _maybe_concat_static(out)
        if remove_static_keys:
            out.pop("ego_static", None)
            out.pop("nb_static", None)

    if save_x_last_abs:
        _maybe_make_x_last_abs(out)

    if check_shapes:
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
            assert nb_static.ndim in (3, 4), f"nb_static must be (N,K,Ds) or (N,T,K,Ds), got {tuple(nb_static.shape)}"

    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, pt_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", type=str, required=True)
    ap.add_argument("--pt_root", type=str, default="data_pt")
    ap.add_argument("--glob", type=str, default="*.npz")
    ap.add_argument("--no_check", action="store_true")

    ap.add_argument("--no_cast_float32", action="store_true")
    ap.add_argument("--no_cast_mask_bool", action="store_true")
    ap.add_argument("--concat_static", action="store_true")
    ap.add_argument("--keep_static_keys", action="store_true")
    ap.add_argument("--save_x_last_abs", action="store_true")
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
        convert_one(
            npz_path,
            pt_path,
            check_shapes=(not args.no_check),
            cast_float32=(not args.no_cast_float32),
            cast_mask_bool=(not args.no_cast_mask_bool),
            concat_static=args.concat_static,
            remove_static_keys=(not args.keep_static_keys),
            save_x_last_abs=args.save_x_last_abs,
        )
        print(f"Converted: {npz_path.name} -> {pt_path.name}")

    print(f"[DONE] converted_files={len(npz_files)}")


if __name__ == "__main__":
    main()