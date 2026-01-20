# src/datasets/pt_dataset.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class PtWindowDataset(Dataset):
    """
    Loads a list of .pt files, each containing N windows.

    Required keys in each .pt:
      x_hist (N,T,De), y_fut (N,Tf,2), nb_hist (N,T,K,Dn), nb_mask (N,T,K)

    Optional:
      ego_static (N,Ds_ego) -> repeated over T and concatenated to x_hist
      nb_static  (N,K,Ds_nb) or (N,T,K,Ds_nb) -> repeated over T and concatenated to nb_hist
      meta fields (recordingId, trackId, t0_frame, ...)
    """

    def __init__(
        self,
        data_dir: Path,
        split_txt: Path,
        stats: Optional[Dict[str, torch.Tensor]] = None,
        return_meta: bool = False,
        use_ego_static: bool = True,
        use_nb_static: bool = True,
        dataset_name: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.return_meta = return_meta
        self.stats = stats
        self.use_ego_static = use_ego_static
        self.use_nb_static = use_nb_static
        self.dataset_name = dataset_name

        names = [ln.strip() for ln in split_txt.read_text().splitlines() if ln.strip()]
        self.files = [self.data_dir / n for n in names]
        missing = [str(p) for p in self.files if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing files in split:\n" + "\n".join(missing))

        self.recs: List[Dict[str, torch.Tensor]] = []
        self.rec_sizes: List[int] = []
        self.prefix: List[int] = [0]

        for p in self.files:
            d = torch.load(p, map_location="cpu", weights_only=False)
            for k in ["x_hist", "y_fut", "nb_hist", "nb_mask"]:
                if k not in d:
                    raise KeyError(f"{p.name} missing key '{k}'. Keys: {list(d.keys())}")

            n = int(d["x_hist"].shape[0])
            self.recs.append(d)
            self.rec_sizes.append(n)
            self.prefix.append(self.prefix[-1] + n)

    def __len__(self) -> int:
        return self.prefix[-1]

    def _locate(self, idx: int) -> Tuple[int, int]:
        lo, hi = 0, len(self.rec_sizes) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.prefix[mid] <= idx < self.prefix[mid + 1]:
                return mid, idx - self.prefix[mid]
            if idx < self.prefix[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        raise IndexError(idx)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec_i, local_i = self._locate(idx)
        d = self.recs[rec_i]

        x_hist = d["x_hist"][local_i].to(torch.float32)     # (T,De)
        y_fut = d["y_fut"][local_i].to(torch.float32)       # (Tf,2)
        y_fut_vel = d["y_fut_vel"][local_i].to(torch.float32) if "y_fut_vel" in d else None
        y_fut_acc = d["y_fut_acc"][local_i].to(torch.float32) if "y_fut_acc" in d else None
        nb_hist = d["nb_hist"][local_i].to(torch.float32)   # (T,K,Dn)
        nb_mask = d["nb_mask"][local_i].bool()              # (T,K)

        x_last_abs = x_hist[-1, 0:2].clone()

        if self.use_ego_static and ("ego_static" in d):
            ego_static = d["ego_static"][local_i].to(torch.float32).view(1, -1)
            x_hist = torch.cat([x_hist, ego_static.expand(x_hist.shape[0], -1)], dim=-1)

        if self.use_nb_static and ("nb_static" in d):
            nb_static = d["nb_static"][local_i].to(torch.float32)

            if nb_static.dim() == 2:
                nb_static = nb_static.unsqueeze(0).expand(nb_hist.shape[0], -1, -1)
            elif nb_static.dim() == 3:
                pass
            elif nb_static.dim() == 4:
                if nb_static.shape[0] == 1:
                    nb_static = nb_static.squeeze(0)
                else:
                    raise RuntimeError(f"Unexpected nb_static 4D shape: {tuple(nb_static.shape)}")
            else:
                raise RuntimeError(f"Unexpected nb_static shape: {tuple(nb_static.shape)}")

            nb_hist = torch.cat([nb_hist, nb_static], dim=-1)

        if self.stats is not None:
            ego_mean = self.stats["ego_mean"].to(torch.float32)
            ego_std = self.stats["ego_std"].to(torch.float32)
            if ego_mean.numel() != x_hist.shape[-1]:
                raise RuntimeError(f"[STATS MISMATCH] ego {ego_mean.numel()} vs {x_hist.shape[-1]}")
            x_hist = (x_hist - ego_mean) / ego_std.clamp_min(1e-6)

            nb_mean = self.stats["nb_mean"].to(torch.float32)
            nb_std = self.stats["nb_std"].to(torch.float32)
            if nb_mean.numel() != nb_hist.shape[-1]:
                raise RuntimeError(f"[STATS MISMATCH] nb {nb_mean.numel()} vs {nb_hist.shape[-1]}")
            nb_hist = (nb_hist - nb_mean) / nb_std.clamp_min(1e-6)

        out: Dict[str, Any] = {
            "x_ego": x_hist,
            "x_nb": nb_hist,
            "nb_mask": nb_mask,
            "y": y_fut,
            "x_last_abs": x_last_abs,
        }
        if y_fut_vel is not None:
            out["y_vel"] = y_fut_vel
        if y_fut_acc is not None:
            out["y_acc"] = y_fut_acc

        if self.return_meta:
            meta: Dict[str, Any] = {}
            if self.dataset_name is not None:
                meta["dataset"] = self.dataset_name
            for k in ["recordingId", "trackId", "t0_frame"]:
                if k in d:
                    meta[k] = d[k][local_i]
            out["meta"] = meta

        return out