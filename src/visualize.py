# visualize_custom.py
"""
Visualize history + GT vs model prediction for a chosen (recording .pt, vehicleId, start frame).
Adapted to work with the NEW data format (13-dim x_hist) while keeping ORIGINAL plot style.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

# 프로젝트 루트 경로 설정 (필요시 수정)
TP_BASELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TP_BASELINE_DIR))

# [FIX] Updated imports for current codebase
from src.utils import set_seed, resolve_path, resolve_data_paths
from src.stats import load_stats_npz_strict, make_stats_filename
from src.models.build import build_model
from src.losses import delta_to_abs, multimodal_loss


def _load_ckpt_state_dict(ckpt_path: Path) -> Dict[str, Any]:
    """
    Supports:
      - raw state_dict
      - dict checkpoint with 'model' key
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict) and any(isinstance(k, str) and (k.endswith(".weight") or k.endswith(".bias")) for k in ckpt.keys()):
        return ckpt
    for k in ["state_dict", "net", "network"]:
        if isinstance(ckpt, dict) and k in ckpt:
            return ckpt[k]
    raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")


def build_inputs_from_pt(
    pt_dict: Dict[str, torch.Tensor],
    idx: int,
    stats: Dict[str, torch.Tensor],
    *,
    # [FIX] New feature flags matching PtWindowDataset
    use_ego_static: bool = True,
    use_nb_static: bool = True,
    use_lead: bool = True,
    use_lc_state: bool = True,
    use_dxtime: bool = True,
    use_gate: bool = True,
    use_neighbors: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Constructs model inputs from the new .pt format (13-dim x_hist, separate safety, etc.)
    """
    # 1. Load Raw Data
    # x_hist is now (T, 13) in the new format
    x_hist_raw = pt_dict["x_hist"][idx] 
    y_fut  = pt_dict["y_fut"][idx]
    
    # 2. Ego Features Construction
    # Base: x_hist (13 dims)
    if x_hist_raw.shape[-1] > 13:
        x_hist_c = x_hist_raw[..., :13]
    else:
        x_hist_c = x_hist_raw

    feats = [x_hist_c.to(torch.float32)]

    # Append Ego Safety (Lead info)
    if use_lead and "ego_safety" in pt_dict:
        ego_safety = pt_dict["ego_safety"][idx].to(torch.float32) # (T, 5)
        feats.append(ego_safety)

    # Append Ego Static
    if use_ego_static and "ego_static" in pt_dict:
        # ego_static is (StaticDim,) -> repeat to (T, StaticDim)
        e_stat = pt_dict["ego_static"][idx].to(torch.float32)
        T = x_hist_c.shape[0]
        e_stat_rep = e_stat.unsqueeze(0).expand(T, -1)
        feats.append(e_stat_rep)

    # Concatenate
    x_ego = torch.cat(feats, dim=-1) # (T, TotalEgoDim)

    # Normalize Ego
    ego_mean = stats["ego_mean"].to(x_ego.device).view(1, -1)
    ego_std  = stats["ego_std"].to(x_ego.device).view(1, -1).clamp_min(1e-6)
    
    # Shape check
    if ego_mean.shape[-1] != x_ego.shape[-1]:
        # Try to fix mismatch if simple (e.g. static dim issue)
        print(f"[WARN] Ego dim mismatch: Input {x_ego.shape[-1]} vs Stats {ego_mean.shape[-1]}")
    
    x_ego = (x_ego - ego_mean) / ego_std
    x_ego = x_ego.unsqueeze(0)  # (1,T,ego_dim)

    # 3. Neighbor Features Construction
    if use_neighbors:
        raw_nb = pt_dict["nb_hist"][idx].to(torch.float32) # (T, 8, 9)
        # raw_nb: [dx, dy, dvx, dvy, dax, day, lc_state, dx_time, gate]
        
        nb_parts = [raw_nb[..., :6]] # Basic kinematics (6)
        
        if use_lc_state:
            nb_parts.append(raw_nb[..., 6:7])
        if use_dxtime:
            nb_parts.append(raw_nb[..., 7:8])
        if use_gate:
            nb_parts.append(raw_nb[..., 8:9])
            
        nb_hist = torch.cat(nb_parts, dim=-1)
        
        # Append NB Static
        if use_nb_static and "nb_static" in pt_dict:
            # nb_static: (8, StaticDim) -> (T, 8, StaticDim)
            n_stat = pt_dict["nb_static"][idx].to(torch.float32)
            T = nb_hist.shape[0]
            n_stat_rep = n_stat.unsqueeze(0).expand(T, -1, -1)
            nb_hist = torch.cat([nb_hist, n_stat_rep], dim=-1)

        nb_mask = pt_dict["nb_mask"][idx].to(torch.bool)

        # Normalize NB
        nb_mean = stats["nb_mean"].to(nb_hist.device).view(1, 1, -1)
        nb_std  = stats["nb_std"].to(nb_hist.device).view(1, 1, -1).clamp_min(1e-6)
        
        x_nb = (nb_hist - nb_mean) / nb_std
        x_nb = x_nb.unsqueeze(0)        # (1,T,8,nb_dim)
        nb_mask = nb_mask.unsqueeze(0)  # (1,T,8)
    else:
        nb_dim = int(stats["nb_mean"].numel())
        x_nb = torch.zeros((1, x_hist_c.shape[0], 8, nb_dim), dtype=torch.float32)
        nb_mask = torch.zeros((1, x_hist_c.shape[0], 8), dtype=torch.bool)

    # 4. Targets
    y_abs = y_fut.to(torch.float32).unsqueeze(0)
    # x_last_abs logic: prefer pre-calculated, fallback to x_hist last point
    if "x_last_abs" in pt_dict:
        x_last_abs = pt_dict["x_last_abs"][idx].to(torch.float32).unsqueeze(0)
    else:
        # Assuming first 2 dims of x_hist_raw are x, y absolute
        x_last_abs = x_hist_raw[-1, 0:2].to(torch.float32).unsqueeze(0)

    return x_ego, x_nb, nb_mask, y_abs, x_last_abs


def pick_sample_index(pt_dict: Dict[str, torch.Tensor], vehicle_id: int, t0_frame: int) -> int:
    track = pt_dict["trackId"].cpu().numpy()
    t0 = pt_dict["t0_frame"].cpu().numpy()
    mask = (track == vehicle_id) & (t0 == t0_frame)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        cand = np.where(track == vehicle_id)[0]
        if len(cand) == 0:
            raise ValueError(f"vehicle_id={vehicle_id} not found in this .pt file.")
        t0s = np.unique(t0[cand])
        nearest = t0s[np.argsort(np.abs(t0s - t0_frame))][:10]
        raise ValueError(
            f"(vehicle_id={vehicle_id}, t0_frame={t0_frame}) not found.\n"
            f"Nearest t0_frame candidates for this vehicle: {nearest.tolist()}"
        )
    return int(idxs[0])


def plot_trajs(
    history_xy: np.ndarray,
    gt_future_xy: np.ndarray,
    pred_future_xy: np.ndarray,
    title: str,
    *,
    invert_y: bool = True,
    draw_boxes: bool = True,
    show_t0_origin: bool = True,
    lane_ys: Optional[list[float]] = None,
):
    """
    [ORIGINAL PLOT FUNCTION] 
    Keeps user's hardcoded styles and markers.
    """

    def add_vehicle_box(ax, center_xy, L=5.15, W=2.32, **kwargs):
        x, y = float(center_xy[0]), float(center_xy[1])
        rect = Rectangle((x - L / 2, y - W / 2), L, W, fill=False, **kwargs)
        ax.add_patch(rect)

    gt_full = np.concatenate([history_xy, gt_future_xy], axis=0)

    plt.figure(figsize=(15, 4))

    # history + GT future
    plt.plot(
        gt_full[:, 0], gt_full[:, 1],
        color="gray", linewidth=1.5,
        marker="o", markersize=3,
        alpha=0.9,
        label="GT (hist+future)"
    )

    # predicted future
    plt.plot(
        pred_future_xy[:, 0], pred_future_xy[:, 1],
        color="tab:blue", linewidth=1.5,
        marker="o", markersize=3,
        alpha=0.95,
        label="Pred future"
    )

    # markers
    plt.scatter(history_xy[0, 0], history_xy[0, 1], s=25, c="gray", marker="o", alpha=0.9)
    plt.scatter(gt_future_xy[-1, 0], gt_future_xy[-1, 1], s=35, c="gray", marker="x", alpha=0.9, label="GT end")
    plt.scatter(pred_future_xy[-1, 0], pred_future_xy[-1, 1], s=35, c="tab:blue", marker="x", alpha=0.9, label="Pred end")

    ax = plt.gca()

    if show_t0_origin:
        # NOTE: this is only meaningful if your preprocessing set t0 to (0,0).
        plt.scatter(0.0, 0.0, s=40, c="black", marker="x", alpha=0.9, label="t0 (origin)")

    # axis limits (tight)
    all_xy = np.vstack([gt_full, pred_future_xy])
    xmin, ymin = all_xy.min(axis=0)
    xmax, ymax = all_xy.max(axis=0)
    xpad = max(5.0, 0.05 * (xmax - xmin))
    ypad = max(0.5, 0.20 * (ymax - ymin))
    plt.xlim(xmin - xpad, xmax + xpad)
    plt.ylim(ymin - ypad, ymax + ypad)

    # lane lines (optional) - [KEEPING HARDCODED VALUES]
    if lane_ys:
        x0, x1 = plt.gca().get_xlim()
        for i, y in enumerate(lane_ys):
            plt.hlines(
                y=y,
                xmin=x0,
                xmax=x1,
                colors="k",
                linestyles="dashed",
                linewidth=1.2,
                alpha=0.7,
                label="Lane" if i == 0 else None
            )

    # tighten y-range around GT to emphasize lane-change
    y_center = float(np.mean(gt_full[:, 1]))
    y_half_range = 5.0
    plt.ylim(y_center - y_half_range, y_center + y_half_range)

    if invert_y:
        plt.gca().invert_yaxis()

    plt.grid(True, alpha=0.25)
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="same yaml used for training/eval")
    ap.add_argument("--ckpt", type=str, required=True, help="checkpoint path")
    ap.add_argument("--pt", type=str, required=True, help="path to a single exid/highd .pt")
    ap.add_argument("--vehicle_id", type=int, required=True)
    ap.add_argument("--t0_frame", type=int, required=True)
    ap.add_argument("--out", type=str, default="vis", help="output root dir (default: vis)")
    ap.add_argument("--show", action="store_true", help="show interactive window")
    ap.add_argument("--no_invert_y", action="store_true", help="do not invert y-axis")
    ap.add_argument("--no_boxes", action="store_true", help="do not draw vehicle boxes")
    ap.add_argument("--no_origin", action="store_true", help="do not draw t0 origin marker (0,0)")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg_dir = cfg_path.parent
    cfg: Dict[str, Any] = yaml.safe_load(cfg_path.read_text())

    set_seed(int(cfg.get("train", {}).get("seed", 42)))

    want_cuda = (cfg.get("train", {}).get("device", "cuda") == "cuda")
    device = torch.device("cuda" if (want_cuda and torch.cuda.is_available()) else "cpu")

    # [FIX] Read features from config matching NEW structure
    feat = cfg.get("features", {})
    flags = dict(
        use_neighbors=feat.get("use_neighbors", True),
        use_ego_static=feat.get("use_ego_static", True),
        use_nb_static=feat.get("use_nb_static", True),
        use_lead=feat.get("use_lead", True),
        use_lc_state=feat.get("use_lc_state", False),
        use_dxtime=feat.get("use_dxtime", False),
        use_gate=feat.get("use_gate", False),
    )

    # stats path resolution (copied from eval.py logic)
    paths = resolve_data_paths(cfg)
    tag = str(paths.get("tag", "unknown"))
    mode = str(cfg.get("data", {}).get("mode", "exid")).lower()
    
    if mode == "exid":
        stats_dir = paths.get("exid_stats_dir", Path("./data/exiD/stats"))
    elif mode == "highd":
        stats_dir = paths.get("highd_stats_dir", Path("./data/highD/stats"))
    else:
        stats_dir = paths.get("combined_stats_dir", Path("./data/combined/stats"))

    # Construct stats filename
    stats_fname = make_stats_filename(tag, flags["use_ego_static"], flags["use_nb_static"])
    stats_path = stats_dir / stats_fname

    if not stats_path.exists():
        # Fallback to direct path in config if provided
        stats_cfg = str(cfg.get("data", {}).get("stats_path", ""))
        if stats_cfg:
            stats_path = Path(stats_cfg)
    
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    
    print(f"[INFO] Loading stats from: {stats_path}")
    # [FIX] Use load_stats_npz_strict
    stats = load_stats_npz_strict(stats_path)

    # model + ckpt
    model = build_model(cfg).to(device)
    state = _load_ckpt_state_dict(Path(args.ckpt).resolve())
    model.load_state_dict(state, strict=True)
    model.eval()

    model_type = str(cfg.get("model", {}).get("type", "baseline")).lower()
    predict_delta = bool(cfg.get("model", {}).get("predict_delta", False))

    # weights
    w_traj = float(cfg.get("train", {}).get("w_traj", 1.0))
    w_fde = float(cfg.get("train", {}).get("w_fde", 0.0))
    w_cls = float(cfg.get("train", {}).get("w_cls", 0.0))

    # load pt
    pt_dict = torch.load(str(Path(args.pt).resolve()), map_location="cpu", weights_only=False)
    idx = pick_sample_index(pt_dict, args.vehicle_id, args.t0_frame)

    x_ego, x_nb, nb_mask, y_abs, x_last_abs = build_inputs_from_pt(pt_dict, idx, stats, **flags)

    x_ego = x_ego.to(device)
    x_nb = x_nb.to(device)
    nb_mask = nb_mask.to(device)
    y_abs = y_abs.to(device)
    x_last_abs = x_last_abs.to(device)

    # Optional style tensors
    style_prob = None
    style_valid = None
    if "style_prob" in pt_dict:
        style_prob = pt_dict["style_prob"][idx].unsqueeze(0).to(torch.float32).to(device)
    if "style_valid" in pt_dict:
        style_valid = pt_dict["style_valid"][idx].unsqueeze(0).to(torch.bool).to(device)

    # forward
    if model_type == "baseline":
        out = model(x_ego, x_nb, nb_mask)
    else:
        out = model(x_ego, x_nb, nb_mask, style_prob=style_prob, style_valid=style_valid)

    if isinstance(out, (tuple, list)) and len(out) == 2:
        pred, scores = out
    else:
        pred, scores = out, None

    # pick best prediction
    if pred.dim() == 4:
        loss_val, best_idx = multimodal_loss(
            pred=pred,
            y_abs=y_abs,
            x_last_abs=x_last_abs,
            predict_delta=predict_delta,
            score_logits=scores,
            w_ade=w_traj, # Note: using w_traj as w_ade for visualization selection
            w_fde=w_fde,
            w_cls=w_cls,
        )
        best_pred = pred[0, int(best_idx[0].item())]
        pred_abs = delta_to_abs(best_pred.unsqueeze(0), x_last_abs)[0] if predict_delta else best_pred
    else:
        pred_abs = delta_to_abs(pred, x_last_abs)[0] if predict_delta else pred[0]

    # Prepare Plot Arrays (Use raw x_hist from pt, only first 2 dims are x,y)
    # Note: pt_dict["x_hist"] in new format is (T, 13), indices 0,1 are x,y
    history_xy = pt_dict["x_hist"][idx][:, 0:2].cpu().numpy()
    gt_future_xy = pt_dict["y_fut"][idx].cpu().numpy()
    pred_future_xy = pred_abs.detach().cpu().numpy()

    rec_id = int(pt_dict["recordingId"][idx].item()) if "recordingId" in pt_dict else -1
    title = f"rec={rec_id} vehicle={args.vehicle_id} t0={args.t0_frame} model={model_type}"

    # [KEEP] Hardcoded lane_ys from original script
    plot_trajs(
        history_xy,
        gt_future_xy,
        pred_future_xy,
        title=title,
        invert_y=(not args.no_invert_y),
        draw_boxes=(not args.no_boxes),
        show_t0_origin=(not args.no_origin),
        lane_ys=[24.46, 28.01, 31.38],  # Original hardcoded lanes
    )

    # auto output path
    ckpt_path = Path(args.ckpt).resolve()
    ckpt_leaf_dir = ckpt_path.parent.name

    out_root = Path(args.out).resolve()
    out_dir = out_root / ckpt_leaf_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = f"rec{rec_id}_vid{args.vehicle_id}_f{args.t0_frame}.png"
    outp = out_dir / fname

    plt.savefig(str(outp), dpi=150)
    print(f"saved: {outp}")

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()