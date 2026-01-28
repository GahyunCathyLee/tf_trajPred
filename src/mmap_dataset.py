from __future__ import annotations
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

class MmapDataset(Dataset):
    """
    Memory-efficient dataset using memory mapping.
    Includes y_vel and y_acc for evaluation.
    """
    def __init__(
        self,
        data_dir: Path,
        tag: str,
        split_indices: Optional[np.ndarray] = None,
        stats: Optional[Dict[str, torch.Tensor]] = None,
        return_meta: bool = False,
        use_ego_static: bool = True,
        use_nb_static: bool = True,
        use_neighbors: bool = True,
        dataset_name: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.tag = tag
        self.stats = stats
        self.return_meta = return_meta
        self.use_ego_static = use_ego_static
        self.use_nb_static = use_nb_static
        self.use_neighbors = use_neighbors
        self.dataset_name = dataset_name

        # Mmap Load
        self.x_ego = np.load(self.data_dir / f"{tag}_x_ego.npy", mmap_mode='r')
        self.x_nb = np.load(self.data_dir / f"{tag}_x_nb.npy", mmap_mode='r')
        self.y = np.load(self.data_dir / f"{tag}_y.npy", mmap_mode='r')
        self.mask = np.load(self.data_dir / f"{tag}_nb_mask.npy", mmap_mode='r')
        self.x_last = np.load(self.data_dir / f"{tag}_x_last_abs.npy", mmap_mode='r')
        
        # [추가] Velocity / Acceleration Load
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

        # Meta
        self.meta_rec = None
        if return_meta:
            self.meta_rec = np.load(self.data_dir / f"{tag}_meta_recordingId.npy", mmap_mode='r')
            self.meta_track = np.load(self.data_dir / f"{tag}_meta_trackId.npy", mmap_mode='r')
            self.meta_frame = np.load(self.data_dir / f"{tag}_meta_frame.npy", mmap_mode='r')

        self.indices = split_indices if split_indices is not None else np.arange(len(self.x_ego))

        if self.stats:
            self._ego_mean = self.stats.get("ego_mean")
            self._ego_std = self.stats.get("ego_std")
            self._nb_mean = self.stats.get("nb_mean")
            self._nb_std = self.stats.get("nb_std")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        # 1. Core Data
        x_hist = torch.from_numpy(self.x_ego[real_idx].copy())
        x_hist = torch.nan_to_num(x_hist, nan=0.0, posinf=1e6, neginf=-1e6)
        
        y_fut = torch.from_numpy(self.y[real_idx].copy())
        y_fut = torch.nan_to_num(y_fut, nan=0.0, posinf=1e6, neginf=-1e6)
        
        x_last_abs = torch.from_numpy(self.x_last[real_idx].copy())
        x_last_abs = torch.nan_to_num(x_last_abs, nan=0.0, posinf=1e6, neginf=-1e6)

        # 2. Static
        if self.use_ego_static and self.ego_static is not None:
            estat = torch.from_numpy(self.ego_static[real_idx].copy())
            estat = torch.nan_to_num(estat, nan=0.0)
            estat = estat.unsqueeze(0).expand(x_hist.shape[0], -1)
            x_hist = torch.cat([x_hist, estat], dim=-1)

        # 3. Neighbors
        if self.use_neighbors:
            x_nb = torch.from_numpy(self.x_nb[real_idx].copy())
            x_nb = torch.nan_to_num(x_nb, nan=0.0, posinf=1e6, neginf=-1e6)
            nb_mask = torch.from_numpy(self.mask[real_idx].copy())
            
            if self.use_nb_static and self.nb_static is not None:
                nstat = torch.from_numpy(self.nb_static[real_idx].copy())
                nstat = torch.nan_to_num(nstat, nan=0.0)
                x_nb = torch.cat([x_nb, nstat], dim=-1)
        else:
            # Ablation
            nb_shape = self.x_nb[real_idx].shape
            mask_shape = self.mask[real_idx].shape
            if self.use_nb_static and self.nb_static is not None:
                d_dyn = nb_shape[-1]
                d_stat = self.nb_static[real_idx].shape[-1]
                x_nb = torch.zeros((nb_shape[0], nb_shape[1], d_dyn + d_stat), dtype=torch.float32)
            else:
                x_nb = torch.zeros(nb_shape, dtype=torch.float32)
            nb_mask = torch.zeros(mask_shape, dtype=torch.bool)

        # 4. Normalize
        if self.stats:
            if self._ego_mean is not None:
                x_hist = (x_hist - self._ego_mean) / self._ego_std.clamp_min(1e-2)
            if self.use_neighbors and self._nb_mean is not None:
                x_nb = (x_nb - self._nb_mean) / self._nb_std.clamp_min(1e-2)
        
        x_hist = torch.nan_to_num(x_hist, nan=0.0)
        x_nb = torch.nan_to_num(x_nb, nan=0.0)

        out = {
            "x_ego": x_hist,
            "x_nb": x_nb,
            "nb_mask": nb_mask,
            "y": y_fut,
            "x_last_abs": x_last_abs
        }
        
        # [추가] Vel/Acc return
        if self.y_vel is not None:
            out["y_vel"] = torch.nan_to_num(torch.from_numpy(self.y_vel[real_idx].copy()), nan=0.0)
        else:
            # Fallback if file missing
            out["y_vel"] = torch.zeros_like(y_fut)

        if self.y_acc is not None:
            out["y_acc"] = torch.nan_to_num(torch.from_numpy(self.y_acc[real_idx].copy()), nan=0.0)
        else:
            out["y_acc"] = torch.zeros_like(y_fut)

        if self.return_meta:
            out["meta"] = {
                "recordingId": int(self.meta_rec[real_idx]),
                "trackId": int(self.meta_track[real_idx]),
                "t0_frame": int(self.meta_frame[real_idx]),
                "dataset": self.dataset_name
            }
        return out