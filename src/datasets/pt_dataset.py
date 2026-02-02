# src/datasets/pt_dataset.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from tqdm import tqdm
import torch
from torch.utils.data import Dataset


class PtWindowDataset(Dataset):
    """
    In-Memory Version: Loads ALL .pt files into RAM at startup.
    """

    def __init__(
        self,
        data_dir: Path,
        split_txt: Optional[Path] = None,
        stats: Optional[Dict[str, torch.Tensor]] = None,
        return_meta: bool = False,
        use_ego_static: bool = True,
        use_nb_static: bool = True,
        use_lead: bool = True,
        use_lc_state: bool = True,
        use_dxtime: bool = True,
        use_gate: bool = True,
        dataset_name: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.return_meta = return_meta
        self.stats = stats

        self.use_ego_static = use_ego_static
        self.use_nb_static = use_nb_static
        self.use_lead = use_lead
        
        self.use_lc_state = use_lc_state
        self.use_dxtime = use_dxtime
        self.use_gate = use_gate

        self.dataset_name = dataset_name

        if self.stats is not None:
            self._ego_mean = self.stats["ego_mean"].to(torch.float32)
            self._ego_std = self.stats["ego_std"].to(torch.float32)
            self._nb_mean = self.stats["nb_mean"].to(torch.float32)
            self._nb_std = self.stats["nb_std"].to(torch.float32)
        else:
            self._ego_mean = self._ego_std = None
            self._nb_mean = self._nb_std = None

        if split_txt is None:
            self.file_paths = sorted(list(self.data_dir.glob("*.pt")))
        else:
            split_txt = Path(split_txt)
            names = [ln.strip() for ln in split_txt.read_text().splitlines() if ln.strip()]
            self.file_paths = [self.data_dir / n for n in names]

        self.recs: List[Dict[str, torch.Tensor]] = []
        self.prefix: List[int] = [0]

        print(f"[INFO] Loading {len(self.file_paths)} files into RAM... ")
        for p in tqdm(self.file_paths, desc="Loading Dataset"):
            d = torch.load(p, map_location="cpu", weights_only=False)

            # --- 1. Ego History (Ego-Safety) ---
            x_hist = d["x_hist"]
            if x_hist.shape[-1] > 13:
                x_hist = x_hist[..., :13]

            if self.use_lead and "ego_safety" in d:
                ego_safety = d["ego_safety"]
                if ego_safety.dim() == 2:
                    ego_safety = ego_safety.unsqueeze(1).expand(x_hist.shape[0], x_hist.shape[1], -1)
                x_hist = torch.cat([x_hist, ego_safety], dim=-1)

            # nb_hist: [dx, dy, dvx, dvy, dax, day, lc_state, dx_time, gate]
            raw_nb = d["nb_hist"]
            
            # (1) 기본 6차원 kinematics
            nb_parts = [raw_nb[..., :6]]
            
            if self.use_lc_state:
                nb_parts.append(raw_nb[..., 6:7])
            if self.use_dxtime:
                nb_parts.append(raw_nb[..., 7:8])
            if self.use_gate:
                nb_parts.append(raw_nb[..., 8:9])
            
            nb_hist = torch.cat(nb_parts, dim=-1)

            if self.use_ego_static and ("ego_static" in d):
                ego_static = d["ego_static"]
                if ego_static.dim() == 2:
                    ego_static = ego_static.unsqueeze(1).expand(x_hist.shape[0], x_hist.shape[1], -1)
                x_hist = torch.cat([x_hist, ego_static], dim=-1)

            if self.use_nb_static and ("nb_static" in d):
                nb_static = d["nb_static"]
                if nb_static.dim() == 3:
                    nb_static = nb_static.unsqueeze(1).expand(nb_hist.shape[0], nb_hist.shape[1], -1, -1)
                nb_hist = torch.cat([nb_hist, nb_static], dim=-1)

            # Stats Shape 검증
            if self.stats is not None:
                if self._ego_mean.numel() != x_hist.shape[-1]:
                    raise ValueError(f"[Stats mismatch] ego_mean dim={self._ego_mean.numel()} but x_hist={x_hist.shape[-1]} in {p.name}")
                if self._nb_mean.numel() != nb_hist.shape[-1]:
                    raise ValueError(f"[Stats mismatch] nb_mean dim={self._nb_mean.numel()} but nb_hist={nb_hist.shape[-1]} in {p.name}")

            d["x_hist"] = x_hist
            d["nb_hist"] = nb_hist

            n = int(d["x_hist"].shape[0])
            self.recs.append(d)
            self.prefix.append(self.prefix[-1] + n)

    def __len__(self) -> int:
        return self.prefix[-1]

    def _locate(self, idx: int) -> Tuple[int, int]:
        lo, hi = 0, len(self.prefix) - 2
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.prefix[mid] <= idx < self.prefix[mid + 1]:
                return mid, idx - self.prefix[mid]
            if idx < self.prefix[mid]: hi = mid - 1
            else: lo = mid + 1
        raise IndexError(f"Index {idx} out of range")

    def _get_meta_from_rec(self, d: Dict[str, Any], local_i: int) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        if self.dataset_name is not None: meta["dataset"] = self.dataset_name
        for k in ["recordingId", "trackId", "t0_frame"]:
            if k in d:
                val = d[k][local_i]
                meta[k] = val.item() if hasattr(val, "item") else val
        return meta

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec_i, local_i = self._locate(idx)
        d = self.recs[rec_i]

        x_hist = d["x_hist"][local_i]
        y_fut = d["y_fut"][local_i]
        nb_hist = d["nb_hist"][local_i]
        nb_mask = d["nb_mask"][local_i]

        if self.stats is not None:
            x_hist = (x_hist - self._ego_mean) / self._ego_std.clamp_min(1e-2)
            nb_hist = (nb_hist - self._nb_mean) / self._nb_std.clamp_min(1e-2)

        out: Dict[str, Any] = {
            "x_ego": x_hist, "x_nb": nb_hist, "nb_mask": nb_mask, "y": y_fut,
            "x_last_abs": d["x_last_abs"][local_i] if "x_last_abs" in d else x_hist[-1, 0:2].clone(),
        }
        if "y_fut_vel" in d: out["y_vel"] = d["y_fut_vel"][local_i]
        if "y_fut_acc" in d: out["y_acc"] = d["y_fut_acc"][local_i]
        if self.return_meta: out["meta"] = self._get_meta_from_rec(d, local_i)

        return out