#!/usr/bin/env python3
"""
Sanity-check NPZ outputs from exiD_raw_to_npz.py and highD_raw_to_npz.py.

Checks
- Required keys exist
- Shapes match declared meta (T, Tf, K, ego_dim, nb_dim, static_dim) when present
- x_hist last-dim == 18
- nb_mask is boolean and consistent with nb_hist masking (heuristic)
- Lead feature semantics:
  - If lead_exists == 0, leadDHW/leadDV/leadTHW/leadTTC are ~0
  - If lead_exists == 1, lead metrics may be valid values or -1 (invalid)
- Optional strict checks for highD metrics (TTC/THW/DHW caps & invalid ranges)

Usage
  python3 sanity_check_npz.py --exid exid_00.npz --highd highd_101.npz
  python3 sanity_check_npz.py --exid_dir path/to/exid_npz --highd_dir path/to/highd_npz --max_files 5
  python3 sanity_check_npz.py --highd_dir path/to/highd_npz --strict_highd_metrics
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


# Agreed x_hist feature indices (ego_dim=18)
IDX = {
    "x": 0, "y": 1, "xV": 2, "yV": 3, "xA": 4, "yA": 5,
    "latLaneCenterOffset": 6,
    "laneChange": 7,
    "norm_off": 8,
    "leadDHW": 9,
    "leadDV": 10,
    "leadTHW": 11,
    "leadTTC": 12,
    "lead_exists": 13,
    "ramp_0": 14, "ramp_1": 15, "ramp_2": 16, "ramp_3": 17,
}


REQUIRED_KEYS = [
    "x_hist", "y_fut", "nb_hist", "nb_mask", "ego_static", "nb_static",
    "recordingId", "trackId", "t0_frame",
]

META_KEYS = ["class_names", "T", "Tf", "K", "ego_dim", "nb_dim", "static_dim", "origin_min_xy"]


def _npz_keys(npz: np.lib.npyio.NpzFile) -> List[str]:
    return sorted(list(npz.files))


def _as_scalar_int(arr: np.ndarray) -> Optional[int]:
    a = np.asarray(arr)
    if a.size == 1:
        return int(a.reshape(-1)[0])
    return None


def _fail(msg: str, errors: List[str]) -> None:
    errors.append(msg)


def check_one_npz(path: Path, dataset_tag: str, strict_highd_metrics: bool) -> Tuple[bool, Dict]:
    errors: List[str] = []
    info: Dict = {"path": str(path), "dataset": dataset_tag}

    if not path.exists():
        return False, {"path": str(path), "error": "file not found"}

    with np.load(path, allow_pickle=True) as z:
        keys = _npz_keys(z)
        info["keys"] = keys

        for k in REQUIRED_KEYS:
            if k not in keys:
                _fail(f"[{dataset_tag}] missing required key: {k}", errors)

        if errors:
            return False, {"path": str(path), "errors": errors, "keys": keys}

        x_hist = z["x_hist"]
        y_fut = z["y_fut"]
        nb_hist = z["nb_hist"]
        nb_mask = z["nb_mask"]
        ego_static = z["ego_static"]
        nb_static = z["nb_static"]

        # dtype checks
        if nb_mask.dtype != np.bool_:
            _fail(f"[{dataset_tag}] nb_mask dtype should be bool, got {nb_mask.dtype}", errors)

        # shape checks
        if x_hist.ndim != 3:
            _fail(f"[{dataset_tag}] x_hist should be (N,T,ego_dim) but got {x_hist.shape}", errors)
        if y_fut.ndim != 3 or y_fut.shape[-1] != 2:
            _fail(f"[{dataset_tag}] y_fut should be (N,Tf,2) but got {y_fut.shape}", errors)
        if nb_hist.ndim != 4 or nb_hist.shape[-1] != 6:
            _fail(f"[{dataset_tag}] nb_hist should be (N,T,K,6) but got {nb_hist.shape}", errors)
        if nb_mask.ndim != 3:
            _fail(f"[{dataset_tag}] nb_mask should be (N,T,K) but got {nb_mask.shape}", errors)
        if ego_static.ndim != 2:
            _fail(f"[{dataset_tag}] ego_static should be (N,static_dim) but got {ego_static.shape}", errors)
        if nb_static.ndim != 4:
            _fail(f"[{dataset_tag}] nb_static should be (N,T,K,static_dim) but got {nb_static.shape}", errors)

        if errors:
            return False, {"path": str(path), "errors": errors, "keys": keys}

        N, T, ego_dim = x_hist.shape
        N2, Tf, _ = y_fut.shape
        if N2 != N:
            _fail(f"[{dataset_tag}] N mismatch: x_hist N={N} vs y_fut N={N2}", errors)

        if ego_dim != 18:
            _fail(f"[{dataset_tag}] ego_dim expected 18 but got {ego_dim}", errors)

        # meta consistency (if present)
        meta_present = []
        if "T" in keys:
            meta_present.append("T")
            t_meta = _as_scalar_int(z["T"])
            if t_meta is not None and t_meta != T:
                _fail(f"[{dataset_tag}] meta T={t_meta} but x_hist T={T}", errors)
        if "Tf" in keys:
            meta_present.append("Tf")
            tf_meta = _as_scalar_int(z["Tf"])
            if tf_meta is not None and tf_meta != Tf:
                _fail(f"[{dataset_tag}] meta Tf={tf_meta} but y_fut Tf={Tf}", errors)
        if "K" in keys:
            meta_present.append("K")
            k_meta = _as_scalar_int(z["K"])
            if k_meta is not None and k_meta != nb_hist.shape[2]:
                _fail(f"[{dataset_tag}] meta K={k_meta} but nb_hist K={nb_hist.shape[2]}", errors)
        if "ego_dim" in keys:
            meta_present.append("ego_dim")
            ed_meta = _as_scalar_int(z["ego_dim"])
            if ed_meta is not None and ed_meta != ego_dim:
                _fail(f"[{dataset_tag}] meta ego_dim={ed_meta} but x_hist ego_dim={ego_dim}", errors)
        for mk in ["class_names","nb_dim","static_dim","origin_min_xy"]:
            if mk in keys:
                meta_present.append(mk)
        info["meta_present"] = sorted(meta_present)

        # lead gating semantics
        lead_exists = x_hist[:, :, IDX["lead_exists"]]
        le0 = lead_exists < 0.5

        for name in ["leadDHW", "leadDV", "leadTHW", "leadTTC"]:
            arr = x_hist[:, :, IDX[name]]
            bad = np.abs(arr[le0]) > 1e-5
            if np.any(bad):
                _fail(f"[{dataset_tag}] {name}: non-zero where lead_exists==0 (count={int(bad.sum())})", errors)

        # ramp onehot should not be negative
        ramp = x_hist[:, :, IDX["ramp_0"]:IDX["ramp_3"]+1]
        if np.any(ramp < -1e-6):
            _fail(f"[{dataset_tag}] ramp values contain negatives", errors)

        # strict checks for highD lead metrics
        if dataset_tag.lower().startswith("highd") and strict_highd_metrics:
            le1 = lead_exists > 0.5
            dhw = x_hist[:, :, IDX["leadDHW"]][le1]
            thw = x_hist[:, :, IDX["leadTHW"]][le1]
            ttc = x_hist[:, :, IDX["leadTTC"]][le1]
            if np.any((ttc != -1.0) & ((ttc <= 1.0) | (ttc > 90.0))):
                _fail("[highD] leadTTC violates rule (-1 or (1,90])", errors)
            if np.any((thw != -1.0) & ((thw <= 0.5) | (thw > 20.0))):
                _fail("[highD] leadTHW violates rule (-1 or (0.5,20])", errors)
            if np.any((dhw != -1.0) & ((dhw <= 10.0) | (dhw > 150.0))):
                _fail("[highD] leadDHW violates rule (-1 or (10,150])", errors)

        # neighbor mask consistency (heuristic)
        if nb_hist.shape[:2] != (N, T):
            _fail(f"[{dataset_tag}] nb_hist (N,T) mismatch vs x_hist: nb_hist {nb_hist.shape[:2]} vs {(N,T)}", errors)
        if nb_mask.shape != (N, T, nb_hist.shape[2]):
            _fail(f"[{dataset_tag}] nb_mask shape mismatch: {nb_mask.shape} vs {(N,T,nb_hist.shape[2])}", errors)

        # Heuristic: masked-out neighbors should be near-zero in nb_hist.
        # Boolean indexing with a (N,T,K) mask selects rows in the last-dim (6) correctly.
        masked_vals = np.abs(nb_hist[~nb_mask])  # shape: (M, 6)
        if masked_vals.size > 0:
            frac = float(np.mean(masked_vals > 1e-4))
            if frac > 0.01:
                _fail(f"[{dataset_tag}] nb_hist has values where nb_mask==False (frac={frac:.3f} > 0.01)", errors)

    ok = (len(errors) == 0)
    info["ok"] = ok
    if not ok:
        info["errors"] = errors
    else:
        info["summary"] = {
            "N": int(N),
            "T": int(T),
            "Tf": int(Tf),
            "K": int(nb_hist.shape[2]),
            "ego_dim": int(ego_dim),
            "static_dim": int(ego_static.shape[-1]),
        }
    return ok, info


def iter_npz_files(dir_path: Path, max_files: int) -> List[Path]:
    return sorted(dir_path.glob("*.npz"))[:max_files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exid", type=str, default=None)
    ap.add_argument("--highd", type=str, default=None)
    ap.add_argument("--exid_dir", type=str, default=None)
    ap.add_argument("--highd_dir", type=str, default=None)
    ap.add_argument("--max_files", type=int, default=3)
    ap.add_argument("--strict_highd_metrics", action="store_true")
    args = ap.parse_args()

    exid_files: List[Path] = []
    highd_files: List[Path] = []

    if args.exid:
        exid_files = [Path(args.exid)]
    elif args.exid_dir:
        exid_files = iter_npz_files(Path(args.exid_dir), args.max_files)

    if args.highd:
        highd_files = [Path(args.highd)]
    elif args.highd_dir:
        highd_files = iter_npz_files(Path(args.highd_dir), args.max_files)

    if not exid_files and not highd_files:
        raise SystemExit("Provide --exid/--highd or --exid_dir/--highd_dir")

    all_ok = True

    for p in exid_files:
        ok, info = check_one_npz(p, "exiD", strict_highd_metrics=False)
        if ok:
            s = info["summary"]
            print(f"[OK][exiD] {p.name}: N={s['N']} T={s['T']} Tf={s['Tf']} K={s['K']} ego_dim={s['ego_dim']} static_dim={s['static_dim']} meta={info['meta_present']}")
        else:
            all_ok = False
            print(f"[FAIL][exiD] {p}")
            for e in info.get("errors", []):
                print("  -", e)

    for p in highd_files:
        ok, info = check_one_npz(p, "highD", strict_highd_metrics=args.strict_highd_metrics)
        if ok:
            s = info["summary"]
            print(f"[OK][highD] {p.name}: N={s['N']} T={s['T']} Tf={s['Tf']} K={s['K']} ego_dim={s['ego_dim']} static_dim={s['static_dim']} meta={info['meta_present']}")
        else:
            all_ok = False
            print(f"[FAIL][highD] {p}")
            for e in info.get("errors", []):
                print("  -", e)

    if not all_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()