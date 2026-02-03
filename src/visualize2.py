# visualize_target.py
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

import xml.etree.ElementTree as ET
from pyproj import CRS, Transformer

# 기존 src 모듈
from src.datasets.collate import collate_batch
from src.models.build import build_model
from src.utils import set_seed, resolve_data_paths
from src.stats import load_stats_npz_strict, make_stats_filename


# ------------------------------------------------------------------
# 1. Map Loader
# ------------------------------------------------------------------
def load_lanelets_from_osm(osm_path: Path, offset_x: float, offset_y: float):
    """
    OSM -> UTM -> Model Coords 변환 후 좌표 범위를 출력합니다.
    """
    print(f"[INFO] Loading map from {osm_path} ...")
    tree = ET.parse(osm_path)
    root = tree.getroot()

    nodes = root.findall("node")
    ways = root.findall("way")
    rels = root.findall("relation")

    # 1. Lat/Lon 수집
    node_latlon = {}
    lats, lons = [], []
    for n in nodes:
        nid = int(n.attrib["id"])
        lat = float(n.attrib["lat"])
        lon = float(n.attrib["lon"])
        node_latlon[nid] = (lat, lon)
        lats.append(lat); lons.append(lon)

    if not lats:
        print("[WARN] No nodes found in OSM file.")
        return []

    # 2. UTM Projection
    lat0, lon0 = np.mean(lats), np.mean(lons)
    zone = int((lon0 + 180) // 6) + 1
    epsg = (32600 + zone) if lat0 >= 0 else (32700 + zone)
    
    tf = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)

    # 3. 좌표 변환 및 Offset 적용
    node_xy = {}
    all_map_x, all_map_y = [], []
    
    for nid, (lat, lon) in node_latlon.items():
        utm_x, utm_y = tf.transform(lon, lat)
        # 변환 식: Plot_XY = UTM_XY - Offset
        tx = utm_x - offset_x
        ty = utm_y - offset_y
        
        node_xy[nid] = (tx, ty)
        all_map_x.append(tx)
        all_map_y.append(ty)

    # [DEBUG] 변환된 지도 좌표 범위 출력
    if all_map_x:
        print(f"\n{'='*20} MAP COORDINATE DEBUG {'='*20}")
        print(f"[DEBUG] Map X Range: {min(all_map_x):.2f} ~ {max(all_map_x):.2f} (Span: {max(all_map_x)-min(all_map_x):.2f})")
        print(f"[DEBUG] Map Y Range: {min(all_map_y):.2f} ~ {max(all_map_y):.2f} (Span: {max(all_map_y)-min(all_map_y):.2f})")
        print(f"{'='*60}\n")
    else:
        print("[WARN] Map is empty after transformation.")

    # 4. Way 구성
    ways_dict = {
        int(w.attrib["id"]): [int(nd.attrib["ref"]) for nd in w.findall("nd")]
        for w in ways
    }

    lane_lines = [] 
    for r in rels:
        tags = {t.attrib["k"]: t.attrib.get("v") for t in r.findall("tag")}
        if tags.get("type") != "lanelet":
            continue

        left, right = None, None
        for m in r.findall("member"):
            if m.attrib.get("type") != "way": continue
            if m.attrib.get("role") == "left": left = int(m.attrib["ref"])
            elif m.attrib.get("role") == "right": right = int(m.attrib["ref"])

        for wid in (left, right):
            if wid is None or wid not in ways_dict: continue
            pts = [node_xy[nid] for nid in ways_dict[wid] if nid in node_xy]
            if not pts: continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            lane_lines.append((xs, ys))
            
    print(f"[INFO] Loaded {len(lane_lines)} lane boundaries.")
    return lane_lines

def get_utm_origin(meta_csv: Path) -> tuple[float, float]:
    """recordingMeta.csv에서 xUtmOrigin, yUtmOrigin을 읽어옵니다."""
    try:
        df = pd.read_csv(meta_csv)
        x_org = float(df["xUtmOrigin"].iloc[0])
        y_org = float(df["yUtmOrigin"].iloc[0])
        print(f"[INFO] Loaded UtmOrigin from meta: ({x_org:.2f}, {y_org:.2f})")
        return x_org, y_org
    except Exception as e:
        print(f"[WARN] Failed to read UtmOrigin from {meta_csv}: {e}")
        return 0.0, 0.0

# ------------------------------------------------------------------
# 1. Dataset Class (Specific File Loader)
# ------------------------------------------------------------------
class SingleFileDataset(Dataset):
    def __init__(self, pt_path: Path, stats=None):
        self.pt_path = pt_path
        self.stats = stats
        self.data = torch.load(self.pt_path, map_location="cpu", weights_only=False)
        self.num_samples = int(self.data["x_hist"].shape[0])
        
        # [CRITICAL] 전처리 시 저장된 Local Min 값을 가져옵니다.
        if "origin_min_xy" in self.data:
            self.min_xy = self.data["origin_min_xy"] # [min_x, min_y]
        else:
            print("[WARN] 'origin_min_xy' not found in .pt file. Alignment might be off.")
            self.min_xy = np.array([0.0, 0.0])

        if self.stats is not None:
            self._ego_mean = self.stats["ego_mean"].to(torch.float32)
            self._ego_std = self.stats["ego_std"].to(torch.float32)
            self._nb_mean = self.stats["nb_mean"].to(torch.float32)
            self._nb_std = self.stats["nb_std"].to(torch.float32)

    def find_index(self, vehicle_id: int, t0_frame: int) -> int:
        tracks = self.data["trackId"].numpy()
        t0s = self.data["t0_frame"].numpy()
        mask = (tracks == vehicle_id) & (t0s == t0_frame)
        idxs = np.where(mask)[0]
        
        if len(idxs) == 0:
            # [복구] 실패 시 후보군(Candidate Frames) 출력 로직
            cand_idxs = np.where(tracks == vehicle_id)[0]
            cand_frames = t0s[cand_idxs]
            cand_frames.sort()
            
            msg = f"Cannot find vehicle_id={vehicle_id}, t0_frame={t0_frame} in {self.pt_path.name}."
            if len(cand_frames) > 0:
                # 너무 많으면 앞부분 20개만 보여줌
                msg += f"\n[Hint] Found vehicle_id={vehicle_id} at frames: {cand_frames[:20].tolist()} ..."
            else:
                msg += f"\n[Hint] Vehicle ID {vehicle_id} does not exist in this file."
                
            raise ValueError(msg)
            
        return int(idxs[0])

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        x_hist = self.data["x_hist"][idx]
        y_fut = self.data["y_fut"][idx]
        nb_hist = self.data["nb_hist"][idx]
        nb_mask = self.data["nb_mask"][idx]
        
        if self.stats is not None:
            x_hist = (x_hist - self._ego_mean) / self._ego_std.clamp_min(1e-2)
            nb_hist = (nb_hist - self._nb_mean) / self._nb_std.clamp_min(1e-2)

        out = {
            "x_ego": x_hist, "x_nb": nb_hist, "nb_mask": nb_mask, "y": y_fut,
            "x_last_abs": self.data["x_last_abs"][idx] if "x_last_abs" in self.data else x_hist[-1, 0:2].clone(),
        }
        return out

# ------------------------------------------------------------------
# 3. Plotting Function
# ------------------------------------------------------------------
def plot_trajs(history_xy, gt_future_xy, pred_future_xy, title, output_path, lane_lines=None, invert_y=True):
    current_xy = history_xy[-1]
    gt_full = np.concatenate([history_xy, gt_future_xy], axis=0)
    
    if pred_future_xy.ndim == 2:
        pred_full = np.vstack([current_xy, pred_future_xy])
    else:
        pred_full = np.vstack([current_xy, pred_future_xy])
    
    all_traj_x = np.concatenate([gt_full[:, 0], pred_full[:, 0]])
    all_traj_y = np.concatenate([gt_full[:, 1], pred_full[:, 1]])

    FIG_W, FIG_H = 10, 12
    plt.figure(figsize=(FIG_W, FIG_H))

    if lane_lines:
        for xs, ys in lane_lines:
            plt.plot(xs, ys, color="black", linewidth=0.5, alpha=0.3, zorder=1)

    # GT & Prediction
    plt.plot(gt_full[:, 0], gt_full[:, 1], color="orange", linewidth=2.0, marker="o", markersize=4, alpha=0.8, label="GT", zorder=2)
    plt.plot(pred_full[:, 0], pred_full[:, 1], color="tab:blue", linewidth=2.0, linestyle="--", marker="o", markersize=4, alpha=0.9, label="Prediction", zorder=3)

    if invert_y: plt.gca().invert_yaxis()
    
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.legend(loc="best")
    plt.title(title)
    
    # 1. 데이터의 중심(Center)과 실제 범위(Span) 계산
    cx = (all_traj_x.min() + all_traj_x.max()) / 2
    cy = (all_traj_y.min() + all_traj_y.max()) / 2
    
    pad = 10 # 여백
    data_span_x = (all_traj_x.max() - all_traj_x.min()) + (pad * 2)
    data_span_y = (all_traj_y.max() - all_traj_y.min()) + (pad * 2)
    
    # 2. 피규어 비율(10:12)과 데이터 비율 비교
    target_ratio = FIG_W / FIG_H  # 10/12 = 0.833...
    data_ratio = data_span_x / data_span_y
    
    if data_ratio > target_ratio:
        # 데이터가 피규어보다 더 '납작'함 -> X축 기준(Width)으로 맞추고 Y축을 늘림
        view_span_x = data_span_x
        view_span_y = data_span_x / target_ratio
    else:
        # 데이터가 피규어보다 더 '길쭉'함 -> Y축 기준(Height)으로 맞추고 X축을 늘림
        view_span_y = data_span_y
        view_span_x = data_span_y * target_ratio
        
    # 3. 계산된 범위 적용
    plt.xlim(cx - view_span_x / 2, cx + view_span_x / 2)
    plt.ylim(cy - view_span_y / 2, cy + view_span_y / 2)
    
    # 4. 물리적 스케일 고정 (1:1)
    plt.gca().set_aspect("equal", adjustable="box")
    
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    print(f"[PLOT] Saved to {output_path}")
    plt.close()

# ------------------------------------------------------------------
# 3. Main Script
# ------------------------------------------------------------------
def _load_ckpt(ckpt_path, model):
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to config yaml")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint pt")
    
    # Target Arguments
    ap.add_argument("--pt_file", type=str, required=True, help="Path to the specific .pt file (e.g. data/exiD/data_pt/exid_test/00.pt)")
    ap.add_argument("--vehicle_id", type=int, required=True, help="Target Vehicle ID")
    ap.add_argument("--t0_frame", type=int, required=True, help="Target Start Frame")
    ap.add_argument("--map_file", type=str, default="./data/exiD/maps/location0.osm", help="Path to .osm file for lane visualization")
    ap.add_argument("--meta_file", type=str, default="./data/exiD/raw/00_recordingMeta.csv", help="Path to recordingMeta.csv (Required for map alignment)")
    
    ap.add_argument("--out_dir", type=str, default="vis_target", help="Output directory")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 1. Load Stats
    paths = resolve_data_paths(cfg)
    features_cfg = cfg.get("features", {})
    use_ego_static = features_cfg.get("use_ego_static", True)
    use_nb_static = features_cfg.get("use_nb_static", True)
    
    # Mode 확인 (exid / highd / combined)
    mode = cfg.get("data", {}).get("mode", "exid").lower()
    tag = paths.get("tag", "unknown")
    
    if mode == "exid":
        stats_dir = paths.get("exid_stats_dir", Path("./data/exiD/stats"))
    elif mode == "highd":
        stats_dir = paths.get("highd_stats_dir", Path("./data/highD/stats"))
    else: 
        stats_dir = Path("./data/combined/stats_0")

    stats_fname = "T2_Tf5_hz3.npz"
    stats_path = stats_dir / stats_fname
    
    print(f"[INFO] Loading stats from {stats_path}")
    stats = load_stats_npz_strict(stats_path)

    # 2. Setup Dataset & Find Sample
    pt_path = Path(args.pt_file)
    if not pt_path.exists():
        raise FileNotFoundError(f"PT file not found: {pt_path}")

    ds = SingleFileDataset(pt_path, stats=stats)
    min_x, min_y = ds.min_xy

    # Auto-detect meta file if not provided
    meta_path = Path(args.meta_file) if args.meta_file else None
    if meta_path is None:
        rec_id_str = pt_path.stem.split("_")[0] if "_" in pt_path.stem else pt_path.stem
        candidate = Path("data/exiD/data") / f"{rec_id_str}_recordingMeta.csv"
        if candidate.exists():
            meta_path = candidate
    
    if meta_path and meta_path.exists():
        utm_x, utm_y = get_utm_origin(meta_path)
    else:
        print("[WARN] No recordingMeta.csv found. Map will likely be misaligned!")
        utm_x, utm_y = 0.0, 0.0
        
    total_offset_x = utm_x + min_x
    total_offset_y = utm_y + min_y

    print(f"\n{'='*20} OFFSET DEBUG {'='*20}")
    print(f"[DEBUG] Local Min (from PT)   : ({min_x:.2f}, {min_y:.2f})")
    print(f"[DEBUG] UTM Origin (from Meta): ({utm_x:.2f}, {utm_y:.2f})")
    print(f"[DEBUG] Total Offset          : ({total_offset_x:.2f}, {total_offset_y:.2f})")
    print(f"{'='*54}\n")

    # 4. Load Map
    lane_lines = []
    if args.map_file:
        lane_lines = load_lanelets_from_osm(
            Path(args.map_file), 
            offset_x=total_offset_x, 
            offset_y=total_offset_y
        )
    
    # 타겟 인덱스 검색
    target_idx = ds.find_index(args.vehicle_id, args.t0_frame)
    print(f"[INFO] Found sample at index {target_idx} (Total samples: {len(ds)})")

    min_x, min_y = ds.min_xy

    # 3. Model Load
    model = build_model(cfg).to(device)
    model = _load_ckpt(args.ckpt, model)
    model.eval()

    # 4. Inference
    # Create a batch of size 1
    sample = ds[target_idx]
    batch = collate_batch([sample]) # 리스트로 감싸서 collate
    
    predict_delta = cfg.get("model", {}).get("predict_delta", False)
    
    with torch.no_grad():
        x_ego = batch["x_ego"].to(device)
        x_nb = batch["x_nb"].to(device)
        nb_mask = batch["nb_mask"].to(device)
        x_last_abs = batch["x_last_abs"].to(device) 
        y_gt = batch["y"].to(device) # (1, Tf, 2)

        out = model(x_ego, x_nb, nb_mask)
        if isinstance(out, (tuple, list)):
            pred, scores = out
        else:
            pred, scores = out, None
        
        # Best Mode Selection (MinADE 기준)
        if pred.dim() == 4: # (1, Modes, T, 2)
            # GT와 가장 가까운 모드 선택
            if predict_delta:
                # Delta -> Abs 변환 후 비교
                pred_abs_candidates = torch.cumsum(pred, dim=2) + x_last_abs[:, None, None, :]
            else:
                pred_abs_candidates = pred
            
            diff = pred_abs_candidates - y_gt.unsqueeze(1) # (1, M, T, 2)
            ade_per_mode = torch.norm(diff, dim=-1).mean(dim=-1) # (1, M)
            best_mode_idx = torch.argmin(ade_per_mode, dim=1)[0].item()
            
            best_pred = pred[:, best_mode_idx] # (1, T, 2)
            print(f"[INFO] Selected Mode {best_mode_idx} (MinADE)")
        else:
            best_pred = pred

        # Delta to Abs
        if predict_delta:
            best_pred_abs = torch.cumsum(best_pred, dim=1) + x_last_abs[:, None, :]
        else:
            best_pred_abs = best_pred

    # 5. Visualize
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_filename = f"rec{pt_path.stem}_id{args.vehicle_id}_frame{args.t0_frame}.png"
    out_path = Path(args.out_dir) / out_filename

    # History Data (Raw)
    # 정규화된 x_ego가 아니라 원본 데이터에서 좌표만 추출
    # Legacy 구조상 x_hist는 [x, y, vx, vy, ax, ay, ...] 형태임
    hist_raw = ds.data["x_hist"][target_idx][:, 0:2].numpy() # (T, 2)
    gt_fut_raw = ds.data["y_fut"][target_idx].numpy() # (Tf, 2)
    pred_fut_raw = best_pred_abs[0].cpu().numpy() # (Tf, 2)

    plot_trajs(
        history_xy=hist_raw,
        gt_future_xy=gt_fut_raw,
        pred_future_xy=pred_fut_raw,
        title=f"File: {pt_path.name} | ID: {args.vehicle_id} | Frame: {args.t0_frame}",
        output_path=Path(args.out_dir) / out_filename,
        lane_lines=lane_lines,
        invert_y=True 
    )

if __name__ == "__main__":
    main()