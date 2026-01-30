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

    Expected keys in each .pt (typical):
      - x_hist:     (N, T, D_ego_base)   # e.g., ego_hist only (13)
      - y_fut:      (N, Tf, 2)
      - nb_hist:    (N, T, K, D_nb_base) # e.g., 9 = 6 kin + 3 lc-feats
      - nb_mask:    (N, T, K) bool
      - (optional) ego_safety: (N, T, 5) # lead-related features (ego-side)
      - (optional) ego_static: (N, S_ego) or (N, T, S_ego)
      - (optional) nb_static:  (N, K, S_nb) or (N, T, K, S_nb)
      - (optional) y_fut_vel, y_fut_acc, x_last_abs, recordingId/trackId/t0_frame ...

    Toggles:
      - use_lead: if True, concatenates ego_safety to x_hist
      - use_lc:   if True, keeps lc features in nb_hist (assumed last 3 dims)
                  if False, drops nb_hist[..., 6:9] -> uses nb_hist[..., :6]
      - use_ego_static / use_nb_static: same as before
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
        use_lc: bool = True,
        dataset_name: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.return_meta = return_meta
        self.stats = stats

        self.use_ego_static = use_ego_static
        self.use_nb_static = use_nb_static
        self.use_lead = use_lead
        self.use_lc = use_lc

        self.dataset_name = dataset_name

        if self.stats is not None:
            self._ego_mean = self.stats["ego_mean"].to(torch.float32)
            self._ego_std = self.stats["ego_std"].to(torch.float32)
            self._nb_mean = self.stats["nb_mean"].to(torch.float32)
            self._nb_std = self.stats["nb_std"].to(torch.float32)
        else:
            self._ego_mean = self._ego_std = None
            self._nb_mean = self._nb_std = None

        # --- resolve file paths ---
        if split_txt is None:
            print(f"[INFO] split_txt is None. Loading all .pt files from {self.data_dir}...")
            self.file_paths = sorted(list(self.data_dir.glob("*.pt")))
            if not self.file_paths:
                raise FileNotFoundError(f"No .pt files found in {self.data_dir}")
        else:
            split_txt = Path(split_txt)
            if not split_txt.exists():
                raise FileNotFoundError(f"Split file not found: {split_txt}")

            names = [ln.strip() for ln in split_txt.read_text().splitlines() if ln.strip()]
            self.file_paths = [self.data_dir / n for n in names]

            missing = [str(p) for p in self.file_paths if not p.exists()]
            if missing:
                raise FileNotFoundError("Missing files in split:\n" + "\n".join(missing))

        # ---- load all data into RAM ----
        self.recs: List[Dict[str, torch.Tensor]] = []
        self.prefix: List[int] = [0]

        print(f"[INFO] Loading {len(self.file_paths)} files into RAM... ")
        for p in tqdm(self.file_paths, desc="Loading Dataset"):
            d = torch.load(p, map_location="cpu", weights_only=False)

            if "x_hist" not in d:
                raise KeyError(f"{p.name} missing key 'x_hist'")
            if "nb_hist" not in d:
                raise KeyError(f"{p.name} missing key 'nb_hist'")
            if "nb_mask" not in d:
                raise KeyError(f"{p.name} missing key 'nb_mask'")
            if "y_fut" not in d:
                raise KeyError(f"{p.name} missing key 'y_fut'")

            x_hist = d["x_hist"]  # (N,T,D)
            nb_hist = d["nb_hist"]  # (N,T,K,Dnb)

            # --- use_lead: concat ego_safety to x_hist ---
            if self.use_lead:
                if "ego_safety" in d:
                    ego_safety = d["ego_safety"]
                    # expect (N,T,5); if (N,5) expand to (N,T,5)
                    if ego_safety.dim() == 2:
                        ego_safety = ego_safety.unsqueeze(1).expand(x_hist.shape[0], x_hist.shape[1], -1)
                    x_hist = torch.cat([x_hist, ego_safety], dim=-1)
                else:
                    # If you want strictness, raise; but keeping it permissive helps mixed datasets.
                    # raise KeyError(f"{p.name} missing key 'ego_safety' while use_lead=True")
                    pass

            # --- use_lc: drop lc features from nb_hist if disabled ---
            # assumes nb_hist last dim is [dx,dy,dvx,dvy,dax,day, lc_state, dx_time, gate]
            if not self.use_lc:
                if nb_hist.shape[-1] >= 6:
                    nb_hist = nb_hist[..., :6]
                else:
                    raise ValueError(f"{p.name} nb_hist has dim {nb_hist.shape[-1]} < 6; cannot slice [:6].")

            # --- static feature concatenation ---
            if self.use_ego_static and ("ego_static" in d):
                ego_static = d["ego_static"]
                # allow (N,S) or (N,T,S)
                if ego_static.dim() == 2:
                    ego_static = ego_static.unsqueeze(1).expand(x_hist.shape[0], x_hist.shape[1], -1)
                x_hist = torch.cat([x_hist, ego_static], dim=-1)

            if self.use_nb_static and ("nb_static" in d):
                nb_static = d["nb_static"]
                # allow (N,K,S) or (N,T,K,S)
                if nb_static.dim() == 3:
                    nb_static = nb_static.unsqueeze(1).expand(nb_hist.shape[0], nb_hist.shape[1], -1, -1)
                nb_hist = torch.cat([nb_hist, nb_static], dim=-1)

            # --- stats shape sanity (optional) ---
            if self.stats is not None:
                if self._ego_mean.numel() != x_hist.shape[-1]:
                    raise ValueError(
                        f"[Stats mismatch] ego_mean dim={self._ego_mean.numel()} "
                        f"but x_hist dim={x_hist.shape[-1]} in {p.name}. "
                        f"(Check your stats file matches toggles: use_lead/use_ego_static)"
                    )
                if self._nb_mean.numel() != nb_hist.shape[-1]:
                    raise ValueError(
                        f"[Stats mismatch] nb_mean dim={self._nb_mean.numel()} "
                        f"but nb_hist dim={nb_hist.shape[-1]} in {p.name}. "
                        f"(Check your stats file matches toggles: use_lc/use_nb_static)"
                    )

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
                meta[k] = val.item() if hasattr(val, "item") else val
        return meta

    def get_meta(self, idx: int) -> Dict[str, Any]:
        rec_i, local_i = self._locate(idx)
        d = self.recs[rec_i]
        return self._get_meta_from_rec(d, local_i)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec_i, local_i = self._locate(idx)
        d = self.recs[rec_i]

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