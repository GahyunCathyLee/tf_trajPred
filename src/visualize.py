# tp_baseline/scripts/visualize.py
"""
Visualize history + GT vs model prediction for a chosen (recording .pt, vehicleId, start frame).

- history + ground truth future : gray
- predicted future trajectory   : blue

This version is aligned with the updated eval.py / losses.py behavior:
- model is built via build_model(cfg)
- checkpoint loading supports {"model": state_dict, ...} or raw state_dict
- multimodal output uses multimodal_loss() to pick best_idx (minADE in ABS space)
- delta->abs conversion uses delta_to_abs() from src.losses
- stats_path can be "auto" -> stats/<data_dir_leaf>/stats.npz
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

TP_BASELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TP_BASELINE_DIR))

from src.utils import load_stats_npz, build_model, set_seed, resolve_path, auto_stats_path_from_data_dir
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
    use_neighbors: bool = True,
    use_context: bool = True,
    use_safety: bool = True,
    use_preceding: bool = True,
    use_lane: bool = True,
    use_static: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      x_ego  : (1,T,ego_dim) normalized
      x_nb   : (1,T,8,nb_dim) normalized
      nb_mask: (1,T,8) bool
      y_abs  : (1,Tf,2) absolute
      x_last_abs: (1,2) absolute
    """
    x_hist = pt_dict["x_hist"][idx]      # (T,6) absolute: (x,y,vx,vy,ax,ay)
    y_fut  = pt_dict["y_fut"][idx]       # (Tf,2) absolute
    T = x_hist.shape[0]

    # ---- ego features ----
    feats = [x_hist.to(torch.float32)]  # (T,6)
    if use_context and "tv_context" in pt_dict:
        feats.append(pt_dict["tv_context"][idx].to(torch.float32))  # (T,2)
    if use_safety and "tv_safety" in pt_dict:
        feats.append(pt_dict["tv_safety"][idx].to(torch.float32))   # (T,3)
    if use_preceding and "tv_preceding" in pt_dict:
        feats.append(pt_dict["tv_preceding"][idx].to(torch.float32))  # (T,1)
    if use_lane and "tv_lane" in pt_dict:
        lane = pt_dict["tv_lane"][idx].to(torch.float32).unsqueeze(-1)  # (T,1)
        feats.append(lane)
    if use_static and "tv_static" in pt_dict:
        tvs = pt_dict["tv_static"][idx].to(torch.float32).view(1, -1).repeat(T, 1)  # (T,3)
        feats.append(tvs)

    x_ego = torch.cat(feats, dim=-1)  # (T,ego_dim)

    ego_mean = stats["ego_mean"].to(x_ego.device).view(1, -1)
    ego_std  = stats["ego_std"].to(x_ego.device).view(1, -1).clamp_min(1e-6)
    if ego_mean.shape[-1] != x_ego.shape[-1]:
        raise ValueError(
            f"ego_dim mismatch: x_ego has {x_ego.shape[-1]} dims but stats has {ego_mean.shape[-1]} dims.\n"
            f"Fix: make sure feature flags match the stats used in training.\n"
            f"Current flags: context={use_context}, safety={use_safety}, preceding={use_preceding}, lane={use_lane}, static={use_static}"
        )
    x_ego = (x_ego - ego_mean) / ego_std
    x_ego = x_ego.unsqueeze(0)  # (1,T,ego_dim)

    # ---- neighbors ----
    if use_neighbors:
        nb_hist = pt_dict["nb_hist"][idx].to(torch.float32)  # (T,8,6) relative-to-ego (nb - ego)
        nb_mask = pt_dict["nb_mask"][idx].to(torch.bool)     # (T,8)
        nb_feats = [nb_hist]
        if use_static and "nb_static" in pt_dict:
            nbs = pt_dict["nb_static"][idx].to(torch.float32).view(1, 8, -1).repeat(T, 1, 1)  # (T,8,3)
            nb_feats.append(nbs)
        x_nb = torch.cat(nb_feats, dim=-1)  # (T,8,nb_dim)

        nb_mean = stats["nb_mean"].to(x_nb.device).view(1, 1, -1)
        nb_std  = stats["nb_std"].to(x_nb.device).view(1, 1, -1).clamp_min(1e-6)
        if nb_mean.shape[-1] != x_nb.shape[-1]:
            raise ValueError(
                f"nb_dim mismatch: x_nb has {x_nb.shape[-1]} dims but stats has {nb_mean.shape[-1]} dims.\n"
                f"Fix: make sure use_static/use_neighbors match training stats."
            )
        x_nb = (x_nb - nb_mean) / nb_std

        x_nb = x_nb.unsqueeze(0)        # (1,T,8,nb_dim)
        nb_mask = nb_mask.unsqueeze(0)  # (1,T,8)
    else:
        # Keep shapes consistent
        nb_dim = int(stats["nb_mean"].numel())
        x_nb = torch.zeros((1, x_hist.shape[0], 8, nb_dim), dtype=torch.float32)
        nb_mask = torch.zeros((1, x_hist.shape[0], 8), dtype=torch.bool)

    # ---- targets / last abs ----
    y_abs = y_fut.to(torch.float32).unsqueeze(0)                # (1,Tf,2)
    x_last_abs = x_hist[-1, 0:2].to(torch.float32).unsqueeze(0) # (1,2)

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
    history_xy: (T,2)
    gt_future_xy: (Tf,2)
    pred_future_xy: (Tf,2)
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

    # lane lines (optional)
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

    # tighten y-range around GT to emphasize lane-change (optional but consistent with your original)
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
    ap.add_argument("--pt", type=str, required=True, help="path to a single highd_XX.pt")
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

    # feature flags must match training stats
    feat = cfg.get("features", {})
    flags = dict(
        use_neighbors=feat.get("use_neighbors", True),
        use_context=feat.get("use_context", True),
        use_safety=feat.get("use_safety", True),
        use_preceding=feat.get("use_preceding", True),
        use_lane=feat.get("use_lane", True),
        use_static=feat.get("use_static", True),
    )

    # stats path: support "auto"
    data_dir = cfg.get("data", {}).get("data_dir", "")
    data_dir_abs = str(resolve_path(cfg_dir, data_dir)) if data_dir else ""
    stats_cfg = str(cfg.get("data", {}).get("stats_path", "auto"))
    if stats_cfg.lower() in ("", "auto"):
        if not data_dir_abs:
            raise ValueError("stats_path is 'auto' but cfg.data.data_dir is missing.")
        stats_path = auto_stats_path_from_data_dir(data_dir_abs)
        # make it relative to project root (so it works when running from tp_baseline/)
        stats_path = stats_path.resolve()
    else:
        stats_path = resolve_path(cfg_dir, stats_cfg)

    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    stats = load_stats_npz(str(stats_path))

    # model + ckpt
    model = build_model(cfg).to(device)
    state = _load_ckpt_state_dict(Path(args.ckpt).resolve())
    model.load_state_dict(state, strict=True)
    model.eval()

    model_type = str(cfg.get("model", {}).get("type", "baseline")).lower()
    predict_delta = bool(cfg.get("model", {}).get("predict_delta", False))

    # weights (for multimodal_loss; best_idx is minADE regardless, but keep consistent)
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

    # optional style tensors (if present in pt)
    style_prob = None
    style_valid = None
    if "style_prob" in pt_dict:
        style_prob = pt_dict["style_prob"][idx].unsqueeze(0).to(torch.float32).to(device)
    if "style_valid" in pt_dict:
        style_valid = pt_dict["style_valid"][idx].unsqueeze(0).to(torch.bool).to(device)

    # forward (baseline does not accept style args)
    if model_type == "baseline":
        out = model(x_ego, x_nb, nb_mask)
    else:
        out = model(x_ego, x_nb, nb_mask, style_prob=style_prob, style_valid=style_valid)

    if isinstance(out, (tuple, list)) and len(out) == 2:
        pred, scores = out
    else:
        pred, scores = out, None

    # pick best prediction and convert to ABS for plotting
    if pred.dim() == 4:
        # multimodal: use same criterion as eval.py (= multimodal_loss best_idx)
        loss_val, best_idx = multimodal_loss(
            pred=pred,
            y_abs=y_abs,
            x_last_abs=x_last_abs,
            predict_delta=predict_delta,
            score_logits=scores,
            w_traj=w_traj,
            w_fde=w_fde,
            w_cls=w_cls,
        )
        best_pred = pred[0, int(best_idx[0].item())]  # (Tf,2) in delta or abs
        pred_abs = delta_to_abs(best_pred.unsqueeze(0), x_last_abs)[0] if predict_delta else best_pred
    else:
        # single-mode
        pred_abs = delta_to_abs(pred, x_last_abs)[0] if predict_delta else pred[0]

    # prepare plot arrays (use stored ABS for history/gt)
    history_xy = pt_dict["x_hist"][idx][:, 0:2].cpu().numpy()
    gt_future_xy = pt_dict["y_fut"][idx].cpu().numpy()
    pred_future_xy = pred_abs.detach().cpu().numpy()

    rec_id = int(pt_dict["recordingId"][idx].item()) if "recordingId" in pt_dict else -1
    title = f"rec={rec_id} vehicle={args.vehicle_id} t0={args.t0_frame} model={model_type}"

    plot_trajs(
        history_xy,
        gt_future_xy,
        pred_future_xy,
        title=title,
        invert_y=(not args.no_invert_y),
        draw_boxes=(not args.no_boxes),
        show_t0_origin=(not args.no_origin),
        lane_ys=[24.46, 28.01, 31.38],  # keep your default; change if needed per recording
    )

    # auto output path: vis/<ckpt_leaf_dir>/rec{rec}_vid{vid}_f{t0}.png
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