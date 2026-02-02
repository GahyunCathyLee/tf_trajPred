from __future__ import annotations
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

class MmapDataset(Dataset):
    """
    Memory-efficient dataset using memory mapping.
    Supports pre-normalized data and granular feature toggles.
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
        # [추가 1] 세부 Feature Toggle
        use_lc_state: bool = True,
        use_dxtime: bool = True,
        use_gate: bool = True,
        dataset_name: Optional[str] = None,
        # [추가 2] 미리 정규화된 데이터인지 여부 (서버에서는 기본값 True 권장)
        is_pre_normalized: bool = True, 
    ):
        self.data_dir = Path(data_dir)
        self.tag = tag
        self.stats = stats
        self.return_meta = return_meta
        self.use_ego_static = use_ego_static
        self.use_nb_static = use_nb_static
        self.use_neighbors = use_neighbors
        
        # Feature Toggle 저장
        self.use_lc_state = use_lc_state
        self.use_dxtime = use_dxtime
        self.use_gate = use_gate

        self.dataset_name = dataset_name
        self.is_pre_normalized = is_pre_normalized

        # Mmap Load (변동 없음)
        self.x_ego = np.load(self.data_dir / f"{tag}_x_ego.npy", mmap_mode='r')
        self.x_nb = np.load(self.data_dir / f"{tag}_x_nb.npy", mmap_mode='r')
        self.y = np.load(self.data_dir / f"{tag}_y.npy", mmap_mode='r')
        self.mask = np.load(self.data_dir / f"{tag}_nb_mask.npy", mmap_mode='r')
        self.x_last = np.load(self.data_dir / f"{tag}_x_last_abs.npy", mmap_mode='r')
        
        # Optional Files Load (변동 없음)
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

        # Meta Load (변동 없음)
        self.meta_rec = None
        if return_meta:
            self.meta_rec = np.load(self.data_dir / f"{tag}_meta_recordingId.npy", mmap_mode='r')
            self.meta_track = np.load(self.data_dir / f"{tag}_meta_trackId.npy", mmap_mode='r')
            self.meta_frame = np.load(self.data_dir / f"{tag}_meta_frame.npy", mmap_mode='r')

        self.indices = split_indices if split_indices is not None else np.arange(len(self.x_ego))

        # Stats Load (변동 없음)
        if self.stats:
            self._ego_mean = self.stats.get("ego_mean")
            self._ego_std = self.stats.get("ego_std")
            self._nb_mean = self.stats.get("nb_mean")
            self._nb_std = self.stats.get("nb_std")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        # 1. Core Data (Ego)
        x_hist = torch.from_numpy(self.x_ego[real_idx].copy())
        y_fut = torch.from_numpy(self.y[real_idx].copy())
        x_last_abs = torch.from_numpy(self.x_last[real_idx].copy())

        # 2. Ego Static
        if self.use_ego_static and self.ego_static is not None:
            estat = torch.from_numpy(self.ego_static[real_idx].copy())
            estat = estat.unsqueeze(0).expand(x_hist.shape[0], -1)
            x_hist = torch.cat([x_hist, estat], dim=-1)

        # 3. Neighbors (Slicing 로직 적용)
        if self.use_neighbors:
            # 전체 Feature(최대 9차원)를 가져온 뒤 필요한 부분만 Slicing
            raw_nb = torch.from_numpy(self.x_nb[real_idx].copy()) 
            nb_mask = torch.from_numpy(self.mask[real_idx].copy())
            
            # (1) 기본 Kinematics (0~5)
            nb_parts = [raw_nb[..., :6]]
            
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
                x_nb = torch.cat([x_nb, nstat], dim=-1)
        else:
            # Neighbors 미사용 시 (Shape만 유지)
            d_dyn = 6 + int(self.use_lc_state) + int(self.use_dxtime) + int(self.use_gate)
            
            nb_shape = self.x_nb[real_idx].shape
            T, K, _ = nb_shape
            
            if self.use_nb_static and self.nb_static is not None:
                d_stat = self.nb_static[real_idx].shape[-1]
                x_nb = torch.zeros((T, K, d_dyn + d_stat), dtype=torch.float32)
            else:
                x_nb = torch.zeros((T, K, d_dyn), dtype=torch.float32)
            nb_mask = torch.zeros((T, K), dtype=torch.bool)

        # 4. Normalize (수정됨: is_pre_normalized 체크)
        # 데이터가 미리 정규화되어 있다면 이 단계를 건너뛰어 CPU 연산을 절약합니다.
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
            # 없는 경우 0으로 채움 (평가 시 에러 방지)
            out["y_vel"] = torch.zeros_like(y_fut)

        if self.y_acc is not None:
            out["y_acc"] = torch.from_numpy(self.y_acc[real_idx].copy())
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
