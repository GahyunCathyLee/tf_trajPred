# src/datasets/mmap_dataset.py

from __future__ import annotations
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Literal

class MmapDataset(Dataset):
    def __init__(
        self,
        tag: str,
        data_dir: Path,
        split_indices: Optional[np.ndarray] = None,
        stats: Optional[Dict[str, torch.Tensor]] = None,
        return_meta: bool = False,
        use_ego_static: bool = True,
        use_nb_static: bool = True,
        use_lead: bool = False, 
        use_neighbors: bool = True,
        use_lc_state: bool = True,
        use_dxtime: bool = True,
        use_gate: bool = True,
        nb_kin_mode: str = "pva",  # "p" | "pv" | "pva"
        is_pre_normalized: bool = False, 
    ):
        self.data_dir = Path(data_dir)
        self.tag = tag
        self.stats = stats
        self.return_meta = return_meta
        self.use_ego_static = use_ego_static
        self.use_nb_static = use_nb_static
        self.use_neighbors = use_neighbors
        self.use_lead = use_lead
        
        self.use_lc_state = use_lc_state
        self.use_dxtime = use_dxtime
        self.use_gate = use_gate

        nb_kin_mode = str(nb_kin_mode).lower().strip()
        allowed = {"p","v","a","pv","pa","va","pva", "none"}
        if nb_kin_mode not in allowed:
            raise ValueError(f"nb_kin_mode must be one of {sorted(allowed)}, got: {nb_kin_mode}")
        self.nb_kin_mode = nb_kin_mode

        self.is_pre_normalized = is_pre_normalized

        # 1. Main Ego File (Dimension: 13)
        self.x_ego = np.load(self.data_dir / f"{tag}_x_ego.npy", mmap_mode='r')
        
        # 2. Safety File (Dimension: 5) - v0 스크립트가 생성한 파일
        self.x_safe = None
        # use_lead가 True일 때만 로드 시도
        if self.use_lead:
            safe_path = self.data_dir / f"{tag}_ego_safety.npy"
            if safe_path.exists():
                self.x_safe = np.load(safe_path, mmap_mode='r')
            else:
                print(f"[WARN] use_lead=True but {safe_path.name} not found. Safety features will be zeros.")

        # 3. Neighbors (Dimension: 9)
        self.x_nb = np.load(self.data_dir / f"{tag}_x_nb.npy", mmap_mode='r')
        self.mask = np.load(self.data_dir / f"{tag}_nb_mask.npy", mmap_mode='r')
        
        # 4. Targets & Others
        self.y = np.load(self.data_dir / f"{tag}_y.npy", mmap_mode='r')
        self.x_last = np.load(self.data_dir / f"{tag}_x_last_abs.npy", mmap_mode='r')
        
        # Optional Files
        self.y_vel = None
        if (self.data_dir / f"{tag}_y_vel.npy").exists():
            self.y_vel = np.load(self.data_dir / f"{tag}_y_vel.npy", mmap_mode='r')
            
        self.y_acc = None
        if (self.data_dir / f"{tag}_y_acc.npy").exists():
            self.y_acc = np.load(self.data_dir / f"{tag}_y_acc.npy", mmap_mode='r')

        self.ego_static = None
        if (self.data_dir / f"{tag}_ego_static.npy").exists():
            self.ego_static = np.load(self.data_dir / f"{tag}_ego_static.npy", mmap_mode='r')
            
        self.nb_static = None
        if (self.data_dir / f"{tag}_nb_static.npy").exists():
            self.nb_static = np.load(self.data_dir / f"{tag}_nb_static.npy", mmap_mode='r')

        # Meta Load 
        self.meta_rec = None
        self.meta_track = None
        self.meta_frame = None
        if return_meta:
            self.meta_rec = np.load(self.data_dir / f"{tag}_meta_recordingId.npy", mmap_mode='r')
            self.meta_track = np.load(self.data_dir / f"{tag}_meta_trackId.npy", mmap_mode='r')
            self.meta_frame = np.load(self.data_dir / f"{tag}_meta_frame.npy", mmap_mode='r')

        self.indices = split_indices if split_indices is not None else np.arange(len(self.x_ego))

        # Stats Load
        if self.stats:
            self._ego_mean = self.stats.get("ego_mean")
            self._ego_std = self.stats.get("ego_std")
            self._nb_mean = self.stats.get("nb_mean")
            self._nb_std = self.stats.get("nb_std")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        # 1. Core Data (Ego) - 13차원
        x_hist = torch.from_numpy(self.x_ego[real_idx].copy())
        y_fut = torch.from_numpy(self.y[real_idx].copy())
        x_last_abs = torch.from_numpy(self.x_last[real_idx].copy())

        if self.use_lead:
            if self.x_safe is not None:
                x_safe = torch.from_numpy(self.x_safe[real_idx].copy())
                x_hist = torch.cat([x_hist, x_safe], dim=-1)
            else:
                T = x_hist.shape[0]
                zeros = torch.zeros((T, 5), dtype=x_hist.dtype)
                x_hist = torch.cat([x_hist, zeros], dim=-1)

        # 2. Ego Static (뒤에 붙임)
        if self.use_ego_static and self.ego_static is not None:
            estat = torch.from_numpy(self.ego_static[real_idx].copy())
            estat = estat.unsqueeze(0).expand(x_hist.shape[0], -1)
            x_hist = torch.cat([x_hist, estat], dim=-1)

        # 3. Neighbors
        if self.use_neighbors:
            raw_nb = torch.from_numpy(self.x_nb[real_idx].copy()) 
            nb_mask = torch.from_numpy(self.mask[real_idx].copy())
            
            # (1) 기본 Kinematics (0~5): [dx,dy,dvx,dvy,dax,day]
            kin6 = raw_nb[..., :6]
            m = str(self.nb_kin_mode).lower().strip()
            if m == "p":
                kin = kin6[..., 0:2]                 # dx, dy
            elif m == "v":
                kin = kin6[..., 2:4]                 # dvx, dvy
            elif m == "a":
                kin = kin6[..., 4:6]                 # dax, day
            elif m == "pv":
                kin = kin6[..., 0:4]                 # dx, dy, dvx, dvy
            elif m == "pa":
                kin = torch.cat([kin6[..., 0:2], kin6[..., 4:6]], dim=-1)  # dx, dy, dax, day
            elif m == "va":
                kin = kin6[..., 2:6]                 # dvx, dvy, dax, day
            elif m == "pva":
                kin = kin6                           # dx, dy, dvx, dvy, dax, day
            elif m == "none":
                kin = None
            else:
                raise ValueError(f"Unknown nb_kin_mode: {self.nb_kin_mode}")

            nb_parts = [kin] if kin is not None else []
            
            # (2) 추가 Feature (Index 6: LC, 7: DxTime, 8: Gate)
            if self.use_lc_state:
                nb_parts.append(raw_nb[..., 6:7])
            if self.use_dxtime:
                nb_parts.append(raw_nb[..., 7:8])
            if self.use_gate:
                nb_parts.append(raw_nb[..., 8:9])
            
            x_nb = torch.cat(nb_parts, dim=-1)
            
            # (3) Nb Static
            if self.use_nb_static and self.nb_static is not None:
                nstat = torch.from_numpy(self.nb_static[real_idx].copy())
                if nstat.ndim == 2:  # (K, D) -> (T, K, D)
                    nstat = nstat.unsqueeze(0).expand(x_nb.shape[0], -1, -1)
                x_nb = torch.cat([x_nb, nstat], dim=-1)

            if len(nb_parts) == 0:
                raise RuntimeError(
                    "Neighbor feature is empty: nb_kin_mode='none' and all aux features disabled."
                )
        else:
            # Neighbor 사용 안함
            nb_shape = self.x_nb[real_idx].shape
            T, K, _ = nb_shape
            x_nb = torch.zeros((T, K, 0), dtype=torch.float32)
            nb_mask = torch.zeros((T, K), dtype=torch.bool)

        # 4. Normalize
        if (not self.is_pre_normalized) and self.stats:
            if self._ego_mean is not None:
                x_hist = (x_hist - self._ego_mean) / self._ego_std.clamp_min(1e-2)
            if self.use_neighbors and self._nb_mean is not None:
                x_nb = (x_nb - self._nb_mean) / self._nb_std.clamp_min(1e-2)

        out = {
            "x_ego": x_hist,
            "x_nb": x_nb,
            "nb_mask": nb_mask,
            "y": y_fut,
            "x_last_abs": x_last_abs
        }
        
        # [Vel/Acc]
        if self.y_vel is not None:
            out["y_vel"] = torch.from_numpy(self.y_vel[real_idx].copy())
        else:
            out["y_vel"] = torch.zeros_like(y_fut)

        if self.y_acc is not None:
            out["y_acc"] = torch.from_numpy(self.y_acc[real_idx].copy())
        else:
            out["y_acc"] = torch.zeros_like(y_fut)

        if self.return_meta:
            out["meta"] = {
                "recordingId": int(self.meta_rec[real_idx]),
                "trackId": int(self.meta_track[real_idx]),
                "dataset_name": str(self.tag),
                "t0_frame": int(self.meta_frame[real_idx]),
            }
        return out