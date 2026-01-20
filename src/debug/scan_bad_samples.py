#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import pandas as pd
from tqdm import tqdm


def as_3d(x: torch.Tensor) -> torch.Tensor:
    """(T,D) -> (1,T,D), (N,T,D) 그대로"""
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() == 3:
        return x
    raise ValueError(f"Unexpected tensor dim={x.dim()} shape={tuple(x.shape)}")


def as_2d(x: torch.Tensor) -> torch.Tensor:
    """(T,2) -> (1,T,2), (N,T,2) 그대로"""
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() == 3:
        return x
    raise ValueError(f"Unexpected tensor dim={x.dim()} shape={tuple(x.shape)}")


def try_get_meta(obj: Dict[str, Any], i: int) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    # meta list[dict]
    if "meta" in obj and isinstance(obj["meta"], list) and len(obj["meta"]) > i:
        m = obj["meta"][i]
        if isinstance(m, dict):
            for k, v in m.items():
                if torch.is_tensor(v) and v.numel() == 1:
                    meta[k] = int(v.item())
                else:
                    meta[k] = v
    # common fields
    for k in ["trackId", "t0_frame", "t0Frame", "recordingId", "recId"]:
        if k in obj and torch.is_tensor(obj[k]):
            v = obj[k]
            if v.dim() == 0:
                meta[k] = int(v.item())
            elif v.dim() == 1 and len(v) > i:
                meta[k] = int(v[i].item())
    return meta


def update_feat_counts(
    counts: Dict[str, torch.Tensor],
    totals: Dict[str, torch.Tensor],
    x: torch.Tensor,
    name: str,
    sentinel_vals: List[float],
):
    """
    counts[name]: (D, K+2) where columns = [nonfinite, zero, sentinel1, sentinel2, ...]
    totals[name]: (D,) total number of elements seen per feature (T*N)
    """
    x3 = as_3d(x)  # (N,T,D)
    N, T, D = x3.shape
    xf = x3.reshape(-1, D)  # (N*T, D)

    if name not in counts:
        K = len(sentinel_vals)
        counts[name] = torch.zeros((D, 2 + K), dtype=torch.long)
        totals[name] = torch.zeros((D,), dtype=torch.long)

    totals[name] += xf.shape[0]

    nonfinite = ~torch.isfinite(xf)
    counts[name][:, 0] += nonfinite.sum(dim=0).to(torch.long)

    # finite mask for exact comparisons (avoid nan==0 false anyway, but keep clean)
    finite = torch.isfinite(xf)

    zero = finite & (xf == 0.0)
    counts[name][:, 1] += zero.sum(dim=0).to(torch.long)

    for j, sv in enumerate(sentinel_vals):
        m = finite & (xf == sv)
        counts[name][:, 2 + j] += m.sum(dim=0).to(torch.long)


def sample_flags(
    x: torch.Tensor,
    sentinel_vals: List[float],
    zero_ratio_thresh: float,
    sentinel_ratio_thresh: float,
) -> Dict[str, Any]:
    """
    샘플 하나(또는 (1,T,D)) 기준으로:
    - nonfinite 존재 여부
    - feature별 zero 비율이 임계값 이상인 dim 목록
    - feature별 sentinel(-1,-1000) 비율이 임계값 이상인 dim 목록
    """
    x3 = as_3d(x)  # (N,T,D) (여기서는 N=1 기대)
    assert x3.shape[0] == 1, "call sample_flags with batch size 1"
    T, D = x3.shape[1], x3.shape[2]
    xi = x3[0]  # (T,D)

    finite = torch.isfinite(xi)
    nonfinite_any = bool((~finite).any().item())

    out: Dict[str, Any] = {"nonfinite": nonfinite_any}

    # zero dominance
    zero_ratio = ((finite & (xi == 0.0)).sum(dim=0).float() / max(1, T)).cpu()
    zero_dims = torch.nonzero(zero_ratio >= zero_ratio_thresh, as_tuple=False).flatten().tolist()
    out["zero_dims"] = zero_dims
    out["zero_max_ratio"] = float(zero_ratio.max().item())

    # sentinel dominance (any of sentinel values)
    sent_mask = torch.zeros_like(xi, dtype=torch.bool)
    for sv in sentinel_vals:
        sent_mask |= (finite & (xi == sv))
    sent_ratio = (sent_mask.sum(dim=0).float() / max(1, T)).cpu()
    sent_dims = torch.nonzero(sent_ratio >= sentinel_ratio_thresh, as_tuple=False).flatten().tolist()
    out["sentinel_dims"] = sent_dims
    out["sentinel_max_ratio"] = float(sent_ratio.max().item())

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/exiD/data_pt/exid_T2_Tf5_hz5")
    ap.add_argument("--glob", type=str, default="**/*.pt")
    ap.add_argument("--out_dir", type=str, default="pt_scan_report")
    ap.add_argument("--sentinels", type=str, default="-1,-1000", help="comma-separated sentinel values to count")
    ap.add_argument("--zero_ratio_thresh", type=float, default=0.95, help="flag dims if >= this fraction are zeros")
    ap.add_argument("--sentinel_ratio_thresh", type=float, default=0.95, help="flag dims if >= this fraction are sentinel")
    ap.add_argument("--max_bad_samples", type=int, default=2000, help="cap for saving sample-level flags")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    pt_paths = sorted(data_dir.glob(args.glob))
    if not pt_paths:
        raise FileNotFoundError(f"No .pt found under {data_dir} with glob={args.glob}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sentinel_vals = [float(x.strip()) for x in args.sentinels.split(",") if x.strip() != ""]

    # feature-level aggregated counts
    counts: Dict[str, torch.Tensor] = {}
    totals: Dict[str, torch.Tensor] = {}

    # sample-level flags
    bad_rows: List[Dict[str, Any]] = []

    # 어떤 키를 볼지 (있으면 스캔)
    keys_to_scan = ["x_ego", "x_nb", "y", "x_last_abs"]

    total_files = 0
    total_samples = 0

    for pt_path in tqdm(pt_paths, desc="scan pt", dynamic_ncols=True):
        obj = torch.load(pt_path, map_location="cpu", weights_only=False)
        if not isinstance(obj, dict):
            continue

        # 파일에 어떤 텐서가 있는지 확인
        available = [k for k in keys_to_scan if k in obj and torch.is_tensor(obj[k])]
        if not available:
            continue

        total_files += 1

        # 샘플 수 추정 (x_ego 기준)
        n_in_file = None
        if "x_ego" in obj and torch.is_tensor(obj["x_ego"]):
            x3 = as_3d(obj["x_ego"])
            n_in_file = x3.shape[0]
        else:
            # fallback: 첫 번째 available tensor로 추정
            x_any = as_3d(obj[available[0]])
            n_in_file = x_any.shape[0]

        total_samples += int(n_in_file)

        # aggregated counts
        for k in available:
            update_feat_counts(counts, totals, obj[k], name=k, sentinel_vals=sentinel_vals)

        # sample-level flags (배치 1씩 뽑아서 검사)
        # 파일이 (N,T,D)면 샘플 단위로 보기 위해 슬라이스
        for i in range(int(n_in_file)):
            if len(bad_rows) >= args.max_bad_samples:
                break

            row_base = {"pt": str(pt_path), "i": i}
            meta = try_get_meta(obj, i)
            if meta:
                row_base.update({f"meta_{k}": v for k, v in meta.items()})

            any_flag = False

            for k in available:
                xk = obj[k]
                # (N,T,D) -> take i-th sample, keep shape (1,T,D) or (1,T,2)
                if xk.dim() == 3:
                    xk_i = xk[i : i + 1]
                elif xk.dim() == 2:
                    # single sample in file
                    xk_i = xk.unsqueeze(0)
                else:
                    continue

                flags = sample_flags(
                    xk_i,
                    sentinel_vals=sentinel_vals,
                    zero_ratio_thresh=args.zero_ratio_thresh,
                    sentinel_ratio_thresh=args.sentinel_ratio_thresh,
                )

                if flags["nonfinite"] or flags["zero_dims"] or flags["sentinel_dims"]:
                    any_flag = True
                    row = dict(row_base)
                    row["tensor"] = k
                    row["nonfinite"] = flags["nonfinite"]
                    row["zero_max_ratio"] = flags["zero_max_ratio"]
                    row["zero_dims"] = ",".join(map(str, flags["zero_dims"])) if flags["zero_dims"] else ""
                    row["sentinel_max_ratio"] = flags["sentinel_max_ratio"]
                    row["sentinel_dims"] = ",".join(map(str, flags["sentinel_dims"])) if flags["sentinel_dims"] else ""
                    bad_rows.append(row)

            # optional: if you only want truly “bad” ones, keep as above

    # ---- write feature summary ----
    feat_rows: List[Dict[str, Any]] = []
    for name, c in counts.items():
        D = c.shape[0]
        for d in range(D):
            tot = int(totals[name][d].item())
            nonfinite = int(c[d, 0].item())
            zero = int(c[d, 1].item())
            row = {
                "tensor": name,
                "dim": d,
                "total": tot,
                "nonfinite": nonfinite,
                "zero": zero,
                "nonfinite_ratio": (nonfinite / tot) if tot > 0 else 0.0,
                "zero_ratio": (zero / tot) if tot > 0 else 0.0,
            }
            for j, sv in enumerate(sentinel_vals):
                cnt = int(c[d, 2 + j].item())
                row[f"sentinel_{sv}"] = cnt
                row[f"sentinel_{sv}_ratio"] = (cnt / tot) if tot > 0 else 0.0
            feat_rows.append(row)

    feat_df = pd.DataFrame(feat_rows).sort_values(["tensor", "dim"])
    feat_csv = out_dir / "feature_value_stats.csv"
    feat_df.to_csv(feat_csv, index=False)

    # ---- write sample flags ----
    bad_df = pd.DataFrame(bad_rows)
    bad_csv = out_dir / "flagged_samples.csv"
    bad_df.to_csv(bad_csv, index=False)

    print("\n==== DONE ====")
    print(f"data_dir: {data_dir}")
    print(f"pt_files_scanned: {total_files} / {len(pt_paths)}")
    print(f"total_samples_est: {total_samples}")
    print(f"sentinels: {sentinel_vals}")
    print(f"feature_stats: {feat_csv}")
    print(f"flagged_samples: {bad_csv}")
    print(f"flagged_rows: {len(bad_rows)}")
    print("\nTip: feature_value_stats.csv에서 sentinel(-1/-1000)/zero 비율이 높은 dim을 먼저 확인하세요.")


if __name__ == "__main__":
    main()