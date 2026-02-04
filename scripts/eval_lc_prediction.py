#!/usr/bin/env python3
# scripts/eval_lc_performance.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pickle
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset, ConcatDataset
from tqdm import tqdm
from matplotlib.path import Path as MplPath
from pyproj import CRS, Transformer

# 기존 프로젝트 모듈
from src.datasets.mmap_dataset import MmapDataset
from src.datasets.collate import collate_batch
from src.models.build import build_model
from src.utils import _to_int, resolve_data_paths, set_seed
from src.stats import load_stats_npz_strict, make_stats_filename, compute_stats_if_needed
from src.scenarios import load_window_labels_csv

# =============================================================================
# 0. Offset Loader
# =============================================================================
def load_offsets_from_npz(npz_dir: Path) -> Dict[int, np.ndarray]:
    offsets = {}
    if not npz_dir.exists():
        print(f"[WARN] NPZ directory not found: {npz_dir}. Coordinates might be misaligned.")
        return offsets

    print(f"[INFO] Loading coordinate offsets from {npz_dir} ...")
    npz_files = list(npz_dir.glob("*.npz"))
    
    if not npz_files:
        print(f"[WARN] No .npz files found in {npz_dir}.")
        return offsets

    for p in npz_files:
        try:
            with np.load(p) as data:
                if 'recordingId' in data and 'origin_min_xy' in data:
                    rids = data['recordingId']
                    rid = int(rids[0]) if rids.ndim > 0 else int(rids)
                    origin = data['origin_min_xy']
                    offsets[rid] = origin
        except Exception:
            pass
            
    print(f"[INFO] Loaded offsets for {len(offsets)} recordings.")
    return offsets

# =============================================================================
# 1. HighD Geometry Engine (Frame-by-Frame)
# =============================================================================
def parse_semicolon_floats(s: str) -> List[float]:
    if not isinstance(s, str): return []
    out = []
    for p in s.split(";"):
        p = p.strip()
        if p: out.append(float(p))
    return out

class HighDGeometry:
    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir
        self.cache: Dict[int, Tuple[np.ndarray, float]] = {}

    def get_lane_boundaries(self, rid: int, meta: Dict[str, Any], offset_lut: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
        if rid in self.cache:
            return self.cache[rid][0]

        raw_id = rid - 100 if rid > 100 else rid
        meta_path = self.raw_dir / f"{raw_id:02d}_recordingMeta.csv"
        
        if not meta_path.exists():
            return None

        try:
            df = pd.read_csv(meta_path)
            upper_str = str(df["upperLaneMarkings"].iloc[0])
            lower_str = str(df["lowerLaneMarkings"].iloc[0])
            
            upper = np.array(parse_semicolon_floats(upper_str), dtype=np.float32)
            lower = np.array(parse_semicolon_floats(lower_str), dtype=np.float32)
            
            C_y = float(upper[-1] + lower[0]) if (len(upper) > 0 and len(lower) > 0) else 0.0

            if rid in offset_lut:
                y_min = float(offset_lut[rid][1])
            else:
                origin = meta.get("origin_min_xy", np.array([0, 0]))
                y_min = float(origin[1])

            boundaries = []
            if len(upper) > 0:
                upper_trans = (C_y - upper) - y_min
                boundaries.extend(upper_trans.tolist())
            if len(lower) > 0:
                lower_trans = lower - y_min
                boundaries.extend(lower_trans.tolist())

            boundaries = np.sort(np.unique(np.array(boundaries)))
            self.cache[rid] = (boundaries, y_min)
            return boundaries

        except Exception as e:
            print(f"[WARN] Failed to load HighD meta for {rid}: {e}")
            return None

    def check_lc_trajectory(self, traj_y: np.ndarray, boundaries: np.ndarray) -> bool:
        """
        궤적 전체(traj_y)를 검사하여 Lane Index가 바뀌는지 확인 (Frame-by-Frame)
        """
        if boundaries is None or len(boundaries) < 2:
            return False
        
        # np.digitize: 각 포인트가 몇 번째 차선(bin)에 있는지 반환
        lane_indices = np.digitize(traj_y, boundaries)
        
        # 초기 차선(t=0)과 다른 차선이 등장하는지 검사
        start_lane = lane_indices[0]
        is_changed = np.any(lane_indices != start_lane)
        return bool(is_changed)

# =============================================================================
# 2. ExiD Geometry Engine (Frame-by-Frame)
# =============================================================================
class ExiDGeometry:
    def __init__(self, maps_dir: Path, adj_pkl: Path, origins_dir: Path):
        self.maps_dir = maps_dir
        self.origins_dir = origins_dir
        self.polygons: Dict[int, Dict[int, MplPath]] = {} 
        self.utm_origins: Dict[int, Tuple[float, float]] = {} 
        self.rec_to_map: Dict[int, int] = {} 
        
        if adj_pkl.exists():
            with open(adj_pkl, "rb") as f:
                data = pickle.load(f)
                self.adj_db = data.get("adj_by_map", data)
        else:
            self.adj_db = {}
            print(f"[WARN] Adjacency pickle not found: {adj_pkl}")

        self._preload_meta()

    def _preload_meta(self):
        for meta_file in self.origins_dir.glob("*_recordingMeta.csv"):
            try:
                rid = int(meta_file.stem.split("_")[0])
                df = pd.read_csv(meta_file, nrows=1)
                self.utm_origins[rid] = (float(df["xUtmOrigin"].iloc[0]), float(df["yUtmOrigin"].iloc[0]))
                if "locationId" in df.columns:
                    self.rec_to_map[rid] = int(df["locationId"].iloc[0])
            except: pass

    def _load_map(self, map_id: int):
        if map_id in self.polygons: return
        osm_path = self.maps_dir / f"location{map_id}.osm"
        if not osm_path.exists(): return
        
        tree = ET.parse(osm_path); root = tree.getroot()
        lats, lons = [], []
        for n in root.findall("node"):
            lats.append(float(n.attrib["lat"])); lons.append(float(n.attrib["lon"]))
        if not lats: return
        
        lat0, lon0 = np.mean(lats), np.mean(lons)
        zone = int((lon0 + 180) // 6) + 1
        epsg = (32600 + zone) if lat0 >= 0 else (32700 + zone)
        tf = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)
        
        utm_nodes = {}
        for n in root.findall("node"):
            utm_nodes[int(n.attrib["id"])] = tf.transform(float(n.attrib["lon"]), float(n.attrib["lat"]))
            
        ways = {int(w.attrib["id"]): [int(nd.attrib["ref"]) for nd in w.findall("nd")] for w in root.findall("way")}
        poly_map = {}
        for r in root.findall("relation"):
            if not r.findall("tag"): continue # Safe check
            # tag check logic simplified
            is_lanelet = False
            for t in r.findall("tag"):
                if t.attrib.get("k") == "type" and t.attrib.get("v") == "lanelet":
                    is_lanelet = True; break
            if not is_lanelet: continue

            lid = int(r.attrib["id"])
            left, right = None, None
            for m in r.findall("member"):
                if m.attrib["role"] == "left": left = int(m.attrib["ref"])
                elif m.attrib["role"] == "right": right = int(m.attrib["ref"])
            if left in ways and right in ways:
                pts = [utm_nodes[n] for n in ways[left] if n in utm_nodes] + \
                      [utm_nodes[n] for n in ways[right][::-1] if n in utm_nodes]
                if len(pts) > 2:
                    poly_map[lid] = MplPath(pts)
        self.polygons[map_id] = poly_map

    def get_lane_id(self, x: float, y: float, map_id: int) -> int:
        if map_id not in self.polygons: return -1
        for lid, poly in self.polygons[map_id].items():
            if poly.contains_point((x, y)):
                return lid
        return -1

    def check_lc_trajectory(self, rid: int, traj: np.ndarray, min_xy: np.ndarray) -> bool:
        map_id = self.rec_to_map.get(rid, -1)
        if map_id == -1: return False
        self._load_map(map_id)
        
        u_org = self.utm_origins.get(rid, (0.0, 0.0))
        adj_set = self.adj_db.get(map_id, {})
        
        # Convert all points to UTM
        traj_utm = traj + min_xy + np.array(u_org)
        
        # t=0의 차선 ID
        start_lid = self.get_lane_id(traj_utm[0,0], traj_utm[0,1], map_id)
        if start_lid == -1: return False
        
        # 궤적 전체를 돌면서 차선 변경 확인
        for t in range(1, len(traj_utm)):
            curr_lid = self.get_lane_id(traj_utm[t,0], traj_utm[t,1], map_id)
            if curr_lid != -1 and curr_lid != start_lid:
                if start_lid in adj_set and curr_lid in adj_set[start_lid]:
                    return True
        return False

# =============================================================================
# 3. Main Logic
# =============================================================================
LC_EVENT_TYPES = {"cut_in", "merging", "diverging", "simple_lane_change", "lane_change_other", "lane_change"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    
    # Custom Args
    ap.add_argument("--window_labels_exid", type=str, default=None)
    ap.add_argument("--window_labels_highd", type=str, default=None)
    ap.add_argument("--highd_raw", type=str, default="data/highD/raw")
    ap.add_argument("--exid_raw", type=str, default="data/exiD/raw")
    ap.add_argument("--exid_maps", type=str, default="data/exiD/maps")
    ap.add_argument("--exid_adj", type=str, default="data/exiD/maps/lanelet_adj_allmaps.pkl")
    ap.add_argument("--output_csv", type=str, default="results/predict_lc_performance.csv")
    ap.add_argument("--npz_dir_exid", type=str, default="data/exiD/data_npz_lc/exid_T2_Tf5_hz3")
    ap.add_argument("--npz_dir_highd", type=str, default="data/highD/data_npz/highd_T2_Tf5_hz3")

    args = ap.parse_args()
    
    # 1. Config
    cfg_path = Path(args.config)
    cfg: Dict[str, Any] = yaml.safe_load(cfg_path.read_text())
    
    cfg.setdefault("data", {})
    if args.batch_size: cfg["data"]["batch_size"] = args.batch_size
    if args.num_workers: cfg["data"]["num_workers"] = args.num_workers

    mode = str(cfg.get("data", {}).get("mode", "combined")).lower()
    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    # Features
    feat_cfg = cfg.get("features", {})
    use_ego_static = bool(feat_cfg.get("use_ego_static", True))
    use_nb_static = bool(feat_cfg.get("use_nb_static", True))
    use_neighbors = bool(cfg.get("model", {}).get("use_neighbors", True))
    use_lead = bool(feat_cfg.get("use_lead", False))
    use_lc_state = bool(feat_cfg.get("use_lc_state", True))
    use_dxtime = bool(feat_cfg.get("use_dxtime", True))
    use_gate = bool(feat_cfg.get("use_gate", True))

    paths = resolve_data_paths(cfg)
    tag = str(paths.get("tag", "T2_Tf5_hz3"))

    exid_dir = paths.get("exid_pt_dir", Path(f"./data/exiD/data_mmap/exid_{tag}"))
    highd_dir = paths.get("highd_pt_dir", Path(f"./data/highD/data_mmap/highd_{tag}"))
    splits_dir = Path("./data/combined/splits") if mode == "combined" else Path(f"./data/{'exiD' if mode=='exid' else 'highD'}/splits")

    # 2. Stats
    print("[INFO] Checking stats...")
    stats_fname = make_stats_filename(
        tag=tag, use_ego_static=use_ego_static, use_nb_static=use_nb_static, 
        use_neighbors=use_neighbors, use_lead=use_lead, use_lc_state=use_lc_state, 
        use_dxtime=use_dxtime, use_gate=use_gate
    )
    
    if mode == "exid": stats_dir = Path("./data/exiD/stats")
    elif mode == "highd": stats_dir = Path("./data/highD/stats")
    else: stats_dir = Path("./data/combined/stats")
    stats_path = stats_dir / stats_fname

    if mode == "exid": data_dirs, splits_dirs = [exid_dir], [Path("./data/exiD/splits")]
    elif mode == "highd": data_dirs, splits_dirs = [highd_dir], [Path("./data/highD/splits")]
    else: data_dirs, splits_dirs = [exid_dir, highd_dir], [Path("./data/exiD/splits"), Path("./data/highD/splits")]

    compute_stats_if_needed(
        stats_path=stats_path, data_dir=data_dirs, splits_dir=splits_dirs,
        stats_split="train", batch_size=args.batch_size, num_workers=args.num_workers,
        data_tag=tag, use_neighbors=use_neighbors, use_ego_static=use_ego_static,
        use_nb_static=use_nb_static, use_lead=use_lead, use_lc_state=use_lc_state,
        use_dxtime=use_dxtime, use_gate=use_gate
    )
    
    print(f"[INFO] Loading stats from {stats_path}")
    stats = load_stats_npz_strict(stats_path)
    split_indices = np.load(splits_dir / f"{args.split}_indices.npy")

    # 3. Load Offsets
    offset_lut = {}
    if args.npz_dir_exid: exid_npz = Path(args.npz_dir_exid)
    else: exid_npz = exid_dir.parent.parent / "data_npz" / exid_dir.name
        
    if args.npz_dir_highd: highd_npz = Path(args.npz_dir_highd)
    else: highd_npz = highd_dir.parent.parent / "data_npz" / highd_dir.name

    if mode in ["exid", "combined"]: offset_lut.update(load_offsets_from_npz(exid_npz))
    if mode in ["highd", "combined"]: offset_lut.update(load_offsets_from_npz(highd_npz))

    # 4. Dataset
    ds_kwargs = {
        "use_ego_static": use_ego_static, "use_nb_static": use_nb_static,
        "use_neighbors": use_neighbors, "use_lc_state": use_lc_state,
        "use_dxtime": use_dxtime, "use_gate": use_gate,
        "stats": stats, "return_meta": True, "is_pre_normalized": False,
    }

    if mode == "exid":
        full = MmapDataset(exid_dir, tag, dataset_name="exid", **ds_kwargs)
        target_ds = Subset(full, split_indices)
    elif mode == "highd":
        full = MmapDataset(highd_dir, tag, dataset_name="highd", **ds_kwargs)
        target_ds = Subset(full, split_indices)
    else:
        exid_full = MmapDataset(exid_dir, tag, dataset_name="exid", **ds_kwargs)
        highd_full = MmapDataset(highd_dir, tag, dataset_name="highd", **ds_kwargs)
        combined_full = ConcatDataset([exid_full, highd_full])
        target_ds = Subset(combined_full, split_indices)

    loader = DataLoader(target_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch)

    # 5. Geometry Engines
    highd_geo = HighDGeometry(Path(args.highd_raw))
    exid_geo = ExiDGeometry(Path(args.exid_maps), Path(args.exid_adj), Path(args.exid_raw))

    labels_lut = {}
    if args.window_labels_exid:
        print(f"[INFO] Loading ExiD labels from CLI: {args.window_labels_exid}")
        labels_lut.update(load_window_labels_csv(Path(args.window_labels_exid)) or {})
    if args.window_labels_highd:
        print(f"[INFO] Loading HighD labels from CLI: {args.window_labels_highd}")
        labels_lut.update(load_window_labels_csv(Path(args.window_labels_highd)) or {})
        
    labels_cfg = cfg.get("data", {}).get("scenario_labels", None)
    if labels_cfg:
        if isinstance(labels_cfg, dict):
            if "exid" in labels_cfg and not args.window_labels_exid:
                print(f"[INFO] Loading ExiD labels from Config: {labels_cfg['exid']}")
                labels_lut.update(load_window_labels_csv(Path(labels_cfg["exid"])) or {})
            if "highd" in labels_cfg and not args.window_labels_highd:
                print(f"[INFO] Loading HighD labels from Config: {labels_cfg['highd']}")
                labels_lut.update(load_window_labels_csv(Path(labels_cfg["highd"])) or {})
        elif isinstance(labels_cfg, str) and not labels_lut:
            print(f"[INFO] Loading labels from Config (string): {labels_cfg}")
            labels_lut.update(load_window_labels_csv(Path(labels_cfg)) or {})

    # 6. Model
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    if "state_dict" in ckpt: state_dict = ckpt["state_dict"]
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    # 7. Evaluation Loop (GT = Real Future LC from Geometry Check)
    results = []
    print(f"[INFO] Evaluating {len(target_ds)} samples (Using Future GT as Target)...")
    
    with torch.no_grad():
        for batch in tqdm(loader):
            x_ego = batch["x_ego"].to(device)
            x_nb = batch["x_nb"].to(device)
            nb_mask = batch["nb_mask"].to(device)
            x_last_abs = batch["x_last_abs"].to(device)
            y_gt = batch["y"].to(device)
            metas = batch["meta"]

            out = model(x_ego, x_nb, nb_mask)
            if isinstance(out, (tuple, list)): pred, scores = out
            else: pred, scores = out, None
            
            if pred.dim() == 4:
                idx = torch.argmax(scores, dim=1) if scores is not None else torch.zeros(len(pred), device=device, dtype=torch.long)
                pred = pred[torch.arange(len(pred)), idx]
            
            if cfg.get("model", {}).get("predict_delta", False):
                pred_abs = torch.cumsum(pred, dim=1) + x_last_abs[:, None, :]
            else:
                pred_abs = pred

            pred_np = pred_abs.cpu().numpy()
            y_gt_np = y_gt.cpu().numpy()
            start_np = x_last_abs.cpu().numpy()

            for i, meta in enumerate(metas):
                if meta is None: continue
                rid = _to_int(meta["recordingId"])
                tid = _to_int(meta["trackId"])
                t0 = _to_int(meta["t0_frame"])

                key = (rid, tid, t0)
                gt_event = str(labels_lut.get(key, {}).get("event_label", "unknown"))
                
                min_xy = offset_lut.get(rid, np.array([0., 0.]))
                
                # Create Trajectories: Start Point + Future Points
                pred_traj = np.concatenate([start_np[i:i+1], pred_np[i]], axis=0)
                gt_traj = np.concatenate([start_np[i:i+1], y_gt_np[i]], axis=0)
                
                pred_lc = False
                real_gt_lc = False
                
                if rid >= 100: # HighD
                    bounds = highd_geo.get_lane_boundaries(rid, meta, offset_lut)
                    pred_lc = highd_geo.check_lc_trajectory(pred_traj[:, 1], bounds)
                    real_gt_lc = highd_geo.check_lc_trajectory(gt_traj[:, 1], bounds)
                else: # ExiD
                    pred_lc = exid_geo.check_lc_trajectory(rid, pred_traj, min_xy)
                    real_gt_lc = exid_geo.check_lc_trajectory(rid, gt_traj, min_xy)
                
                # Logic: Target is "Did it actually LC in future?"
                target_is_lc = real_gt_lc
                
                results.append({
                    "recordingId": rid,
                    "trackId": tid,
                    "t0_frame": t0,
                    "gt_event": gt_event,       # Original Category
                    "target_is_lc": target_is_lc, # Ground Truth for Metric
                    "pred_is_lc": pred_lc,        # Prediction
                    "is_correct": (target_is_lc == pred_lc)
                })

    # 8. Stats Calculation & Saving
    df_samples = pd.DataFrame(results)
    
    # Save Samples CSV
    samples_path = Path(args.output_csv).with_name(Path(args.output_csv).stem + "_samples.csv")
    df_samples.to_csv(samples_path, index=False)
    print(f"\n[INFO] Saved sample details to {samples_path}")

    # Compute Summaries (Using real_gt_lc as Truth)
    tp = len(df_samples[(df_samples["target_is_lc"]==True) & (df_samples["pred_is_lc"]==True)])
    fn = len(df_samples[(df_samples["target_is_lc"]==True) & (df_samples["pred_is_lc"]==False)])
    fp = len(df_samples[(df_samples["target_is_lc"]==False) & (df_samples["pred_is_lc"]==True)])
    tn = len(df_samples[(df_samples["target_is_lc"]==False) & (df_samples["pred_is_lc"]==False)])
    
    total = len(df_samples)
    acc = (tp + tn) / max(1, total)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1, prec + rec)
    
    print("\n" + "="*50)
    print(f" LC INTENTION PERFORMANCE (Target: Future GT Trajectory)")
    print("="*50)
    print(f" Total Samples   : {total}")
    print(f" Actual Future LC: {tp+fn} (Geometry Verified)")
    print("-" * 50)
    print(f" Accuracy     : {acc:.4f}")
    print(f" Precision    : {prec:.4f}")
    print(f" Recall       : {rec:.4f}")
    print(f" F1 Score     : {f1:.4f}")
    print("-" * 50)
    print("[Breakdown by Original Dataset Label]")
    
    summary_rows = []
    summary_rows.append({
        "Category": "Overall", "Count": total, 
        "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn
    })
    
    for evt in sorted(df_samples["gt_event"].unique()):
        sub = df_samples[df_samples["gt_event"] == evt]
        count = len(sub)
        
        # Real positive targets in this group
        real_pos_count = sub["target_is_lc"].sum()
        real_neg_count = count - real_pos_count
        
        # Metrics
        sub_tp = len(sub[(sub["target_is_lc"]==True) & (sub["pred_is_lc"]==True)])
        sub_fp = len(sub[(sub["target_is_lc"]==False) & (sub["pred_is_lc"]==True)])
        
        sub_recall = sub_tp / max(1, real_pos_count)
        sub_fpr = sub_fp / max(1, real_neg_count)
        
        summary_rows.append({
            "Category": evt,
            "Count": count,
            "Target_Positive_Count": real_pos_count, # How many actually changed lane
            "Recall": sub_recall,
            "FPR": sub_fpr,
            "TP": sub_tp, "FP": sub_fp
        })
        
        print(f" {evt:<20s}: Total={count:<5d} | Target LC={real_pos_count:<5d} | Recall={sub_recall*100:.1f}% | FPR={sub_fpr*100:.1f}%")

    # Save Summary CSV
    df_summary = pd.DataFrame(summary_rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(args.output_csv, index=False)
    print(f"[INFO] Saved summary statistics to {args.output_csv}")

if __name__ == "__main__":
    main()