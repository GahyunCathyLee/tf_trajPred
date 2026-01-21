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
    Best for servers with huge RAM (e.g., 500GB+).
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

        # ---- cache stats as float32 ONCE ----
        if self.stats is not None:
            self._ego_mean = self.stats["ego_mean"].to(torch.float32)
            self._ego_std = self.stats["ego_std"].to(torch.float32)
            self._nb_mean = self.stats["nb_mean"].to(torch.float32)
            self._nb_std = self.stats["nb_std"].to(torch.float32)
        else:
            self._ego_mean = self._ego_std = None
            self._nb_mean = self._nb_std = None

        if not split_txt.exists():
             raise FileNotFoundError(f"Split file not found: {split_txt}")
             
        names = [ln.strip() for ln in split_txt.read_text().splitlines() if ln.strip()]
        self.file_paths = [self.data_dir / n for n in names]

        missing = [str(p) for p in self.file_paths if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing files in split:\n" + "\n".join(missing))

        # ---- Load All Data into RAM ----
        self.recs: List[Dict[str, torch.Tensor]] = []
        self.prefix: List[int] = [0]
        
        print(f"[INFO] Loading {len(self.file_paths)} files into RAM... ")
        
        # tqdm으로 로딩 진행 상황 표시
        for p in tqdm(self.file_paths, desc="Loading Dataset"):
            # 여기서 데이터를 전부 RAM에 올림
            d = torch.load(p, map_location="cpu", weights_only=False)
            
            # Shape Check & Pre-processing
            if "x_hist" not in d:
                 raise KeyError(f"{p.name} missing key 'x_hist'")
            
            # Static Features Concatenation (미리 합쳐둠)
            x_hist = d["x_hist"]
            nb_hist = d["nb_hist"]

            if self.use_ego_static and ("ego_static" in d):
                ego_static = d["ego_static"].view(1, -1)
                x_hist = torch.cat([x_hist, ego_static.expand(x_hist.shape[0], -1)], dim=-1)
                
            if self.use_nb_static and ("nb_static" in d):
                nb_static = d["nb_static"]
                if nb_static.dim() == 2:
                    nb_static = nb_static.unsqueeze(0).expand(nb_hist.shape[0], -1, -1)
                elif nb_static.dim() == 4 and nb_static.shape[0] == 1:
                    nb_static = nb_static.squeeze(0)
                nb_hist = torch.cat([nb_hist, nb_static], dim=-1)

            # 덮어쓰기 (메모리 절약)
            d["x_hist"] = x_hist
            d["nb_hist"] = nb_hist

            n = int(d["x_hist"].shape[0])
            self.recs.append(d)
            self.prefix.append(self.prefix[-1] + n)

        print(f"[INFO] Dataset loaded. Total samples: {self.prefix[-1]}")


    def __len__(self) -> int:
        return self.prefix[-1]

    def _locate(self, idx: int) -> Tuple[int, int]:
        lo, hi = 0, len(self.prefix) - 2
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.prefix[mid] <= idx < self.prefix[mid + 1]:
                return mid, idx - self.prefix[mid]
            if idx < self.prefix[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        raise IndexError(f"Index {idx} out of range")

    def _get_meta_from_rec(self, d: Dict[str, Any], local_i: int) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        if self.dataset_name is not None:
            meta["dataset"] = self.dataset_name
        for k in ["recordingId", "trackId", "t0_frame"]:
            if k in d:
                val = d[k][local_i]
                if hasattr(val, "item"):
                    meta[k] = val.item()
                else:
                    meta[k] = val
        return meta

    def get_meta(self, idx: int) -> Dict[str, Any]:
        rec_i, local_i = self._locate(idx)
        d = self.recs[rec_i]
        return self._get_meta_from_rec(d, local_i)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # RAM에서 즉시 가져옴 (No Disk I/O)
        rec_i, local_i = self._locate(idx)
        d = self.recs[rec_i]

        # 이미 Static이 합쳐져 있음
        x_hist = d["x_hist"][local_i]
        y_fut = d["y_fut"][local_i]
        nb_hist = d["nb_hist"][local_i]
        nb_mask = d["nb_mask"][local_i]

        # Optional fields
        y_fut_vel = d["y_fut_vel"][local_i] if "y_fut_vel" in d else None
        y_fut_acc = d["y_fut_acc"][local_i] if "y_fut_acc" in d else None

        if "x_last_abs" in d:
            x_last_abs = d["x_last_abs"][local_i]
        else:
            x_last_abs = x_hist[-1, 0:2].clone()

        # Normalize (using cached stats)
        # 중요: clamp_min(1e-2) 적용됨
        if self.stats is not None:
            x_hist = (x_hist - self._ego_mean) / self._ego_std.clamp_min(1e-2)
            nb_hist = (nb_hist - self._nb_mean) / self._nb_std.clamp_min(1e-2)

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
            out["meta"] = self._get_meta_from_rec(d, local_i)
        
        return out