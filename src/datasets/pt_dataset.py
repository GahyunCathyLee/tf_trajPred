# src/datasets/pt_dataset.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

class PtWindowDataset(Dataset):
    """
    Lazy Loading version of PtWindowDataset.
    Loads .pt files on-the-fly in __getitem__ to save RAM.
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

        # Read split file
        if not split_txt.exists():
             raise FileNotFoundError(f"Split file not found: {split_txt}")
             
        names = [ln.strip() for ln in split_txt.read_text().splitlines() if ln.strip()]
        self.file_paths = [self.data_dir / n for n in names]

        missing = [str(p) for p in self.file_paths if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing files in split:\n" + "\n".join(missing))

        # ---- Index Building (Lazy Mode) ----
        # We iterate files ONLY to get the number of samples in each file.
        # We do NOT store the content.
        self.rec_sizes: List[int] = []
        self.prefix: List[int] = [0]
        
        # print(f"[INFO] Scanning {len(self.file_paths)} files for indexing... (Lazy Loading)")
        
        for p in self.file_paths:
            # We must load to check size. 
            # map_location="cpu" is essential.
            # We delete 'd' immediately to free RAM.
            try:
                # weights_only=False is needed for older pytorch/structures, 
                # but if your .pt is simple tensors, weights_only=True is safer/faster in new torch.
                # using weights_only=False to be safe based on your previous code.
                d = torch.load(p, map_location="cpu", weights_only=False)
                
                if "x_hist" not in d:
                     raise KeyError(f"{p.name} missing key 'x_hist'")
                
                n = int(d["x_hist"].shape[0])
                self.rec_sizes.append(n)
                self.prefix.append(self.prefix[-1] + n)
                
                del d  # CRITICAL: Free memory
            except Exception as e:
                print(f"[ERROR] Failed to load/index {p}: {e}")
                raise e

    def __len__(self) -> int:
        return self.prefix[-1]

    def _locate(self, idx: int) -> Tuple[int, int]:
        """Binary search to find which file contains the global index."""
        lo, hi = 0, len(self.rec_sizes) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.prefix[mid] <= idx < self.prefix[mid + 1]:
                return mid, idx - self.prefix[mid]
            if idx < self.prefix[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        raise IndexError(f"Index {idx} out of range")

    def _get_meta_from_rec(self, d: Dict[str, torch.Tensor], local_i: int) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        if self.dataset_name is not None:
            meta["dataset"] = self.dataset_name
        for k in ["recordingId", "trackId", "t0_frame"]:
            if k in d:
                val = d[k][local_i]
                # Convert tensor scalar to int/float if possible
                if hasattr(val, "item"):
                    meta[k] = val.item()
                else:
                    meta[k] = val
        return meta

    def get_meta(self, idx: int) -> Dict[str, Any]:
        """
        In Lazy mode, this forces a file load, so it is SLOW if called repeatedly
        outside of the main loop (e.g. for building sampler weights).
        """
        rec_i, local_i = self._locate(idx)
        path = self.file_paths[rec_i]
        
        # Load just for meta
        d = torch.load(path, map_location="cpu", weights_only=False)
        meta = self._get_meta_from_rec(d, local_i)
        del d
        return meta

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 1. Locate file
        rec_i, local_i = self._locate(idx)
        path = self.file_paths[rec_i]

        # 2. Load file on demand (Lazy)
        # This is the trade-off: IO overhead per sample vs RAM savings
        d = torch.load(path, map_location="cpu", weights_only=False)

        # 3. Extract tensors
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

        # 4. Handle static features
        if self.use_ego_static and ("ego_static" in d):
            ego_static = d["ego_static"][local_i].view(1, -1)
            x_hist = torch.cat([x_hist, ego_static.expand(x_hist.shape[0], -1)], dim=-1)

        if self.use_nb_static and ("nb_static" in d):
            nb_static = d["nb_static"][local_i]
            # dimension handling
            if nb_static.dim() == 2:
                # (K, D) -> expand to (T, K, D) if nb_hist is (T,K,D)
                nb_static = nb_static.unsqueeze(0).expand(nb_hist.shape[0], -1, -1)
            elif nb_static.dim() == 3:
                pass
            elif nb_static.dim() == 4:
                if nb_static.shape[0] == 1:
                    nb_static = nb_static.squeeze(0)
            
            nb_hist = torch.cat([nb_hist, nb_static], dim=-1)

        # 5. Normalize (using cached stats)
        if self.stats is not None:
            # We assume stats are already float32 on CPU
            # x_hist: (T, De)
            x_hist = (x_hist - self._ego_mean) / self._ego_std.clamp_min(1e-6)
            
            # nb_hist: (T, K, Dn)
            nb_hist = (nb_hist - self._nb_mean) / self._nb_std.clamp_min(1e-6)

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
        
        del d

        return out