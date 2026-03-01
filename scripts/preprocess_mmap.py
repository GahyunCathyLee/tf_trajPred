import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from numpy.lib.format import open_memmap
from bisect import bisect_left

# -------------------------
# Helpers
# -------------------------
def get_shapes(valid_file_path: Path):
    d = np.load(valid_file_path, allow_pickle=True)
    shapes = {}
    shapes["x_hist"] = d["x_hist"].shape[1:]
    shapes["y_fut"] = d["y_fut"].shape[1:]

    # Optional Velocity/Acceleration Shapes
    shapes["y_fut_vel"] = d["y_fut_vel"].shape[1:] if "y_fut_vel" in d else None
    shapes["y_fut_acc"] = d["y_fut_acc"].shape[1:] if "y_fut_acc" in d else None

    shapes["nb_hist"] = d["nb_hist"].shape[1:]
    shapes["nb_mask"] = d["nb_mask"].shape[1:]

    if "ego_static" in d:
        shapes["ego_static"] = d["ego_static"].shape[1:]
    if "nb_static" in d:
        shp = list(d["nb_static"].shape)
        # If nb_static is (N, M, C), expand to (N, T, M, C) to match training-time usage
        if len(shp) == 3:
            T = shapes["x_hist"][0]
            shp.insert(1, T)
        shapes["nb_static"] = tuple(shp[1:])
    if "ego_safety" in d:
        shapes["ego_safety"] = d["ego_safety"].shape[1:]

    return shapes


def update_welford(count, mean, m2, new_data):
    n = new_data.shape[0]
    if n == 0:
        return count, mean, m2

    new_data = new_data.astype(np.float64)
    batch_mean = new_data.mean(axis=0)
    batch_m2 = ((new_data - batch_mean) ** 2).sum(axis=0)

    delta = batch_mean - mean
    new_count = count + n
    new_mean = mean + delta * (n / new_count)
    new_m2 = m2 + batch_m2 + (delta ** 2) * (count * n / new_count)
    return new_count, new_mean, new_m2


def finalize_stats(count, mean, m2, threshold=1e-3):
    if count < 2:
        return mean.astype(np.float32), np.ones_like(mean, dtype=np.float32)

    var = m2 / (count - 1)
    std = np.sqrt(np.maximum(var, 1e-12))

    low_idx = np.where(std < threshold)[0]
    if len(low_idx) > 0:
        print(f"    [Fix] Found {len(low_idx)} low-variance features. Forcing mean=0, std=1.")
        mean[low_idx] = 0.0
        std[low_idx] = 1.0

    return mean.astype(np.float32), std.astype(np.float32)


def _merge_intervals(intervals):
    """
    intervals: list of (start, end) inclusive, start<=end
    return merged sorted list
    """
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe + 1:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _intervals_overlap(sample_start, sample_end, merged_intervals):
    """
    sample_start/end inclusive.
    merged_intervals sorted by start.
    Use binary search on starts.
    """
    if not merged_intervals:
        return False
    starts = [it[0] for it in merged_intervals]
    idx = bisect_left(starts, sample_start)
    # candidate could be idx-1 (interval that starts before sample_start)
    for j in (idx - 1, idx):
        if 0 <= j < len(merged_intervals):
            s, e = merged_intervals[j]
            if not (sample_end < s or e < sample_start):
                return True
    return False


def build_keep_index(
    d,
    hz: int,
    pad_steps: int,
    drop_mode: str,
    keys_to_scan=("x_hist",),
):
    """
    Returns:
      keep_idx: (N,) bool
      debug_info: dict
    drop_mode:
      - "row_only": drop only samples whose x_hist contains NaN/Inf
      - "frame_window": find bad frames in x_hist and drop ANY sample whose frame span overlaps
                       expanded bad-frame windows (±pad_steps*hz).
    Assumption (confirmed by user):
      - t0_frame is the FIRST history frame (absolute frame index).
    """
    x = d["x_hist"]
    N = x.shape[0]
    Th = x.shape[1]
    Tf = d["y_fut"].shape[1] if "y_fut" in d else 0

    debug = {
        "drop_mode_used": drop_mode,
        "n_total": N,
        "n_bad_rows": 0,
        "n_bad_frames": 0,
        "n_drop_final": 0,
    }

    # If we can't do frame-window without t0_frame, fallback.
    if drop_mode == "frame_window" and ("t0_frame" not in d):
        drop_mode = "row_only"
        debug["drop_mode_used"] = "row_only"

    # Only floating tensors can have NaN/Inf meaningfully.
    if not np.issubdtype(x.dtype, np.floating):
        keep = np.ones(N, dtype=bool)
        return keep, debug

    # --- Mode A: row_only ---
    if drop_mode == "row_only":
        row_ok = np.isfinite(x).all(axis=(1, 2))
        debug["n_bad_rows"] = int((~row_ok).sum())
        debug["n_drop_final"] = debug["n_bad_rows"]
        return row_ok, debug

    # --- Mode B: frame_window ---
    # Find bad positions in x_hist: (i, t, c)
    bad_pos = np.argwhere(~np.isfinite(x))
    if bad_pos.size == 0:
        keep = np.ones(N, dtype=bool)
        return keep, debug

    # Identify bad rows for reference/debug
    bad_rows = np.unique(bad_pos[:, 0])
    debug["n_bad_rows"] = int(bad_rows.size)

    t0 = d["t0_frame"].astype(np.int64)  # (N,)

    # Convert each bad (i, t) to an absolute frame:
    # since t0 is FIRST history frame and history is sampled every hz:
    # frame(i, t) = t0[i] + t * hz
    bad_frames = set()
    for i, t, _c in bad_pos:
        bf = int(t0[int(i)] + int(t) * hz)
        bad_frames.add(bf)
    debug["n_bad_frames"] = int(len(bad_frames))

    pad = pad_steps * hz

    # Build expanded bad-frame intervals and merge them for fast overlap checking.
    intervals = [(bf - pad, bf + pad) for bf in bad_frames]
    merged = _merge_intervals(intervals)

    # Sample span (inclusive):
    # history: t0 .. t0 + (Th-1)*hz
    # future : (assumed) continues after history, so include up to t0 + (Th+Tf-1)*hz
    # (This is a conservative span, tends to drop more rather than less.)
    start = t0
    end = t0 + (Th + max(Tf, 0) - 1) * hz

    keep = np.ones(N, dtype=bool)
    for i in range(N):
        s = int(start[i])
        e = int(end[i])
        if _intervals_overlap(s, e, merged):
            keep[i] = False

    debug["n_drop_final"] = int((~keep).sum())
    return keep, debug


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--npz_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--calc_stats", action="store_true")

    # New filtering options
    ap.add_argument("--drop_mode", type=str, default="frame_window",
                    choices=["row_only", "frame_window"],
                    help="row_only: drop only samples containing NaN/Inf in x_hist. "
                         "frame_window: drop any sample overlapping expanded bad-frame windows.")
    ap.add_argument("--hz", type=int, default=3, help="Frame stride used when building windows (e.g., hz=3).")
    ap.add_argument("--pad_steps", type=int, default=8,
                    help="Expand each bad frame by ±(pad_steps*hz) frames.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Only scan and report how many samples would be kept/dropped. No files written.")

    args = ap.parse_args()

    npz_dir = Path(args.npz_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(npz_dir.glob("*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files in {npz_dir}")

    # 1) Scan and build per-file keep_idx
    print(f"[INFO] Scanning {len(files)} files...")
    valid_file_info = []   # list of (path, keep_idx, n_keep)
    total_samples = 0
    total_dropped = 0

    for p in tqdm(files, desc="Scanning"):
        try:
            with np.load(p, allow_pickle=True) as d:
                keep_idx, dbg = build_keep_index(
                    d,
                    hz=args.hz,
                    pad_steps=args.pad_steps,
                    drop_mode=args.drop_mode,
                )
                n_keep = int(keep_idx.sum())
                n_drop = int((~keep_idx).sum())
                total_dropped += n_drop

                if n_keep == 0:
                    print(f"⚠️ [SKIP] {p.name}: kept 0 / {dbg['n_total']} (mode={dbg['drop_mode_used']})")
                    continue

                valid_file_info.append((p, keep_idx, n_keep))

                total_samples += n_keep

        except Exception as e:
            print(f"⚠️ [SKIP] {p.name}: exception during scan: {e}")

    print(f"[INFO] Total kept samples: {total_samples}")
    print(f"[INFO] Total dropped samples (by filtering): {total_dropped}")
    if total_samples == 0:
        raise RuntimeError("No samples left after filtering!")

    if args.dry_run:
        print("[DRY RUN] Exiting without writing output.")
        return

    # 2) Allocate memmaps using the first valid file as shape reference
    shapes = get_shapes(valid_file_info[0][0])

    fp_x = open_memmap(out_dir / f"x_ego.npy", mode="w+", dtype="float32",
                       shape=(total_samples, *shapes["x_hist"]))
    fp_y = open_memmap(out_dir / f"y.npy", mode="w+", dtype="float32",
                       shape=(total_samples, *shapes["y_fut"]))
    fp_nb = open_memmap(out_dir / f"x_nb.npy", mode="w+", dtype="float32",
                        shape=(total_samples, *shapes["nb_hist"]))
    fp_mask = open_memmap(out_dir / f"nb_mask.npy", mode="w+", dtype="bool",
                          shape=(total_samples, *shapes["nb_mask"]))
    fp_last = open_memmap(out_dir / f"x_last_abs.npy", mode="w+", dtype="float32",
                          shape=(total_samples, 2))

    # Optional Velocity/Acceleration memmaps
    fp_yv = None
    if shapes["y_fut_vel"] is not None:
        fp_yv = open_memmap(out_dir / f"y_vel.npy", mode="w+", dtype="float32",
                            shape=(total_samples, *shapes["y_fut_vel"]))

    fp_ya = None
    if shapes["y_fut_acc"] is not None:
        fp_ya = open_memmap(out_dir / f"y_acc.npy", mode="w+", dtype="float32",
                            shape=(total_samples, *shapes["y_fut_acc"]))

    fp_estat = None
    if "ego_static" in shapes:
        fp_estat = open_memmap(out_dir / f"ego_static.npy", mode="w+", dtype="float32",
                               shape=(total_samples, *shapes["ego_static"]))

    fp_nstat = None
    if "nb_static" in shapes:
        fp_nstat = open_memmap(out_dir / f"nb_static.npy", mode="w+", dtype="float32",
                               shape=(total_samples, *shapes["nb_static"]))

    fp_safe = None
    if "ego_safety" in shapes:
        fp_safe = open_memmap(out_dir / f"ego_safety.npy", mode="w+", dtype="float32",
                              shape=(total_samples, *shapes["ego_safety"]))

    meta_rec = np.zeros(total_samples, dtype=np.int32)
    meta_track = np.zeros(total_samples, dtype=np.int32)
    meta_frame = np.zeros(total_samples, dtype=np.int32)

    # 3) Stats init
    if args.calc_stats:
        s_dyn_mean = np.zeros(shapes["x_hist"][-1], dtype=np.float64)
        s_dyn_m2 = np.zeros(shapes["x_hist"][-1], dtype=np.float64)
        s_dyn_cnt = 0

        s_nb_mean = np.zeros(shapes["nb_hist"][-1], dtype=np.float64)
        s_nb_m2 = np.zeros(shapes["nb_hist"][-1], dtype=np.float64)
        s_nb_cnt = 0

        s_es_mean = np.zeros(shapes["ego_static"][0], dtype=np.float64) if fp_estat is not None else None
        s_es_m2 = np.zeros(shapes["ego_static"][0], dtype=np.float64) if fp_estat is not None else None
        s_es_cnt = 0

        s_ns_mean = np.zeros(shapes["nb_static"][-1], dtype=np.float64) if fp_nstat is not None else None
        s_ns_m2 = np.zeros(shapes["nb_static"][-1], dtype=np.float64) if fp_nstat is not None else None
        s_ns_cnt = 0

    # 4) Write only kept samples
    cursor = 0
    for p, keep_idx, n_keep in tqdm(valid_file_info, desc="Writing"):
        with np.load(p, allow_pickle=True) as d:
            sel = keep_idx
            end = cursor + n_keep

            fp_x[cursor:end] = d["x_hist"][sel].astype(np.float32)
            fp_y[cursor:end] = d["y_fut"][sel].astype(np.float32)
            fp_nb[cursor:end] = d["nb_hist"][sel].astype(np.float32)
            fp_mask[cursor:end] = d["nb_mask"][sel].astype(bool)

            if "x_last_abs" in d:
                fp_last[cursor:end] = d["x_last_abs"][sel].astype(np.float32)
            else:
                fp_last[cursor:end] = d["x_hist"][sel, -1, 0:2].astype(np.float32)

            # Optional vel/acc
            if fp_yv is not None and "y_fut_vel" in d:
                fp_yv[cursor:end] = d["y_fut_vel"][sel].astype(np.float32)
            if fp_ya is not None and "y_fut_acc" in d:
                fp_ya[cursor:end] = d["y_fut_acc"][sel].astype(np.float32)

            # Optional static
            if fp_estat is not None and "ego_static" in d:
                fp_estat[cursor:end] = d["ego_static"][sel].astype(np.float32)

            if fp_safe is not None and "ego_safety" in d:
                fp_safe[cursor:end] = d["ego_safety"][sel].astype(np.float32)

            ns_val = None
            if fp_nstat is not None and "nb_static" in d:
                ns_val = d["nb_static"][sel].astype(np.float32)
                if ns_val.ndim == 3:
                    # (N, M, C) -> (N, T, M, C)
                    T = d["x_hist"].shape[1]
                    ns_val = np.repeat(np.expand_dims(ns_val, 1), T, axis=1)
                fp_nstat[cursor:end] = ns_val

            # meta
            if "recordingId" in d:
                meta_rec[cursor:end] = d["recordingId"][sel]
            if "trackId" in d:
                meta_track[cursor:end] = d["trackId"][sel]
            if "t0_frame" in d:
                meta_frame[cursor:end] = d["t0_frame"][sel]

            # stats (IMPORTANT: must use only selected samples)
            if args.calc_stats:
                x_sel = d["x_hist"][sel].astype(np.float32)
                s_dyn_cnt, s_dyn_mean, s_dyn_m2 = update_welford(
                    s_dyn_cnt, s_dyn_mean, s_dyn_m2,
                    x_sel.reshape(-1, shapes["x_hist"][-1])
                )

                nb_sel = d["nb_hist"][sel]
                mask_sel = d["nb_mask"][sel].astype(bool)
                valid_nb = nb_sel[mask_sel].astype(np.float32)
                if valid_nb.size > 0:
                    s_nb_cnt, s_nb_mean, s_nb_m2 = update_welford(s_nb_cnt, s_nb_mean, s_nb_m2, valid_nb)

                if s_es_mean is not None and "ego_static" in d:
                    s_es_cnt, s_es_mean, s_es_m2 = update_welford(
                        s_es_cnt, s_es_mean, s_es_m2,
                        d["ego_static"][sel].astype(np.float32)
                    )

                if s_ns_mean is not None and (ns_val is not None):
                    valid_ns = ns_val[mask_sel]
                    if valid_ns.size > 0:
                        s_ns_cnt, s_ns_mean, s_ns_m2 = update_welford(s_ns_cnt, s_ns_mean, s_ns_m2, valid_ns)

            cursor = end

    # 5) Flush + save meta
    fp_x.flush()
    fp_y.flush()
    fp_nb.flush()
    fp_mask.flush()
    fp_last.flush()
    if fp_yv is not None:
        fp_yv.flush()
    if fp_ya is not None:
        fp_ya.flush()
    if fp_estat is not None:
        fp_estat.flush()
    if fp_safe is not None:
        fp_safe.flush()
    if fp_nstat is not None:
        fp_nstat.flush()

    np.save(out_dir / f"meta_recordingId.npy", meta_rec)
    np.save(out_dir / f"meta_trackId.npy", meta_track)
    np.save(out_dir / f"meta_frame.npy", meta_frame)

    # 6) Finalize stats
    if args.calc_stats:
        print("\n[STATS] Finalizing...")
        m_x, s_x = finalize_stats(s_dyn_cnt, s_dyn_mean, s_dyn_m2)
        m_nb, s_nb = finalize_stats(s_nb_cnt, s_nb_mean, s_nb_m2)

        save_dict = {
            "dyn_ego_mean": m_x,
            "dyn_ego_std": s_x,
            "dyn_nb_mean": m_nb,
            "dyn_nb_std": s_nb,
        }

        if s_es_mean is not None:
            m_es, s_es = finalize_stats(s_es_cnt, s_es_mean, s_es_m2)
            save_dict["stat_ego_mean"] = m_es
            save_dict["stat_ego_std"] = s_es

        if s_ns_mean is not None:
            m_ns, s_ns = finalize_stats(s_ns_cnt, s_ns_mean, s_ns_m2)
            save_dict["stat_nb_mean"] = m_ns
            save_dict["stat_nb_std"] = s_ns

        stats_out = out_dir / "stats.npz"
        np.savez(stats_out, **save_dict)
        print(f"[STATS] Saved to {stats_out}")


if __name__ == "__main__":
    main()
