import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from numpy.lib.format import open_memmap

def get_shapes(valid_file_path):
    d = np.load(valid_file_path, allow_pickle=True)
    shapes = {}
    shapes["x_hist"] = d["x_hist"].shape[1:] 
    shapes["y_fut"] = d["y_fut"].shape[1:]
    # [추가] Velocity/Acceleration Shapes
    shapes["y_fut_vel"] = d["y_fut_vel"].shape[1:] if "y_fut_vel" in d else None
    shapes["y_fut_acc"] = d["y_fut_acc"].shape[1:] if "y_fut_acc" in d else None
    
    shapes["nb_hist"] = d["nb_hist"].shape[1:]
    shapes["nb_mask"] = d["nb_mask"].shape[1:]
    
    if "ego_static" in d:
        shapes["ego_static"] = d["ego_static"].shape[1:]
    if "nb_static" in d:
        shp = list(d["nb_static"].shape)
        if len(shp) == 3: 
            T = shapes["x_hist"][0]
            shp.insert(1, T)
        shapes["nb_static"] = tuple(shp[1:])
        
    return shapes

def check_nan_inf(d):
    # [수정] y_fut_vel, y_fut_acc도 검사 대상에 포함
    keys_to_check = ["x_hist", "y_fut", "nb_hist", "ego_static", "nb_static", "y_fut_vel", "y_fut_acc"]
    for k in keys_to_check:
        if k in d:
            data = d[k]
            if np.issubdtype(data.dtype, np.floating):
                if not np.isfinite(data).all():
                    return False, k
    return True, None

def update_welford(count, mean, m2, new_data):
    n = new_data.shape[0]
    if n == 0: return count, mean, m2
    
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--calc_stats", action="store_true")
    args = ap.parse_args()

    npz_dir = Path(args.npz_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag

    files = sorted(list(npz_dir.glob("*.npz")))
    if not files: raise FileNotFoundError(f"No .npz files in {npz_dir}")

    # 1. Scan
    print(f"[INFO] Scanning {len(files)} files...")
    total_samples = 0
    valid_file_info = []
    
    for p in tqdm(files, desc="Scanning"):
        try:
            with np.load(p, allow_pickle=True) as d:
                is_valid, bad_key = check_nan_inf(d)
                if not is_valid:
                    print(f"⚠️ [SKIP] {p.name} contains NaN/Inf in '{bad_key}'")
                    continue
                n = d["x_hist"].shape[0]
                valid_file_info.append((p, n))
                total_samples += n
        except Exception as e:
            print(f"Skipping {p}: {e}")

    print(f"-> Total valid samples: {total_samples}")
    if total_samples == 0: raise RuntimeError("No valid samples found!")

    # 2. Allocate
    shapes = get_shapes(valid_file_info[0][0])
    
    fp_x = open_memmap(out_dir / f"{tag}_x_ego.npy", mode='w+', dtype='float32', shape=(total_samples, *shapes["x_hist"]))
    fp_y = open_memmap(out_dir / f"{tag}_y.npy", mode='w+', dtype='float32', shape=(total_samples, *shapes["y_fut"]))
    fp_nb = open_memmap(out_dir / f"{tag}_x_nb.npy", mode='w+', dtype='float32', shape=(total_samples, *shapes["nb_hist"]))
    fp_mask = open_memmap(out_dir / f"{tag}_nb_mask.npy", mode='w+', dtype='bool', shape=(total_samples, *shapes["nb_mask"]))
    fp_last = open_memmap(out_dir / f"{tag}_x_last_abs.npy", mode='w+', dtype='float32', shape=(total_samples, 2))
    
    # [추가] Velocity / Acceleration Memmaps
    fp_yv = None
    if shapes["y_fut_vel"] is not None:
        fp_yv = open_memmap(out_dir / f"{tag}_y_vel.npy", mode='w+', dtype='float32', shape=(total_samples, *shapes["y_fut_vel"]))
    
    fp_ya = None
    if shapes["y_fut_acc"] is not None:
        fp_ya = open_memmap(out_dir / f"{tag}_y_acc.npy", mode='w+', dtype='float32', shape=(total_samples, *shapes["y_fut_acc"]))

    fp_estat = None
    if "ego_static" in shapes:
        fp_estat = open_memmap(out_dir / f"{tag}_ego_static.npy", mode='w+', dtype='float32', shape=(total_samples, *shapes["ego_static"]))
    
    fp_nstat = None
    if "nb_static" in shapes:
        fp_nstat = open_memmap(out_dir / f"{tag}_nb_static.npy", mode='w+', dtype='float32', shape=(total_samples, *shapes["nb_static"]))

    meta_rec = np.zeros(total_samples, dtype=np.int32)
    meta_track = np.zeros(total_samples, dtype=np.int32)
    meta_frame = np.zeros(total_samples, dtype=np.int32)

    # Stats logic (unchanged)
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

    # 3. Write
    cursor = 0
    for p, n in tqdm(valid_file_info, desc="Writing"):
        with np.load(p, allow_pickle=True) as d:
            end = cursor + n
            
            fp_x[cursor:end] = d["x_hist"].astype(np.float32)
            fp_y[cursor:end] = d["y_fut"].astype(np.float32)
            fp_nb[cursor:end] = d["nb_hist"].astype(np.float32)
            fp_mask[cursor:end] = d["nb_mask"].astype(bool)
            
            if "x_last_abs" in d: fp_last[cursor:end] = d["x_last_abs"].astype(np.float32)
            else: fp_last[cursor:end] = d["x_hist"][:, -1, 0:2].astype(np.float32)

            # [추가] Write Vel/Acc
            if fp_yv is not None and "y_fut_vel" in d:
                fp_yv[cursor:end] = d["y_fut_vel"].astype(np.float32)
            if fp_ya is not None and "y_fut_acc" in d:
                fp_ya[cursor:end] = d["y_fut_acc"].astype(np.float32)

            if fp_estat is not None: fp_estat[cursor:end] = d["ego_static"].astype(np.float32)
            if fp_nstat is not None:
                ns_val = d["nb_static"].astype(np.float32)
                if ns_val.ndim == 3: 
                    T = d["x_hist"].shape[1]
                    ns_val = np.repeat(np.expand_dims(ns_val, 1), T, axis=1)
                fp_nstat[cursor:end] = ns_val

            if "recordingId" in d: meta_rec[cursor:end] = d["recordingId"]
            if "trackId" in d: meta_track[cursor:end] = d["trackId"]
            if "t0_frame" in d: meta_frame[cursor:end] = d["t0_frame"]

            if args.calc_stats:
                s_dyn_cnt, s_dyn_mean, s_dyn_m2 = update_welford(s_dyn_cnt, s_dyn_mean, s_dyn_m2, d["x_hist"].astype(np.float32).reshape(-1, shapes["x_hist"][-1]))
                valid_nb = d["nb_hist"][d["nb_mask"].astype(bool)].astype(np.float32)
                if valid_nb.size > 0:
                    s_nb_cnt, s_nb_mean, s_nb_m2 = update_welford(s_nb_cnt, s_nb_mean, s_nb_m2, valid_nb)
                if s_es_mean is not None:
                    s_es_cnt, s_es_mean, s_es_m2 = update_welford(s_es_cnt, s_es_mean, s_es_m2, d["ego_static"].astype(np.float32))
                if s_ns_mean is not None:
                    valid_ns = ns_val[d["nb_mask"].astype(bool)]
                    if valid_ns.size > 0:
                        s_ns_cnt, s_ns_mean, s_ns_m2 = update_welford(s_ns_cnt, s_ns_mean, s_ns_m2, valid_ns)

            cursor = end

    # Flush
    fp_x.flush(); fp_y.flush(); fp_nb.flush(); fp_mask.flush(); fp_last.flush()
    if fp_yv is not None: fp_yv.flush()
    if fp_ya is not None: fp_ya.flush()
    if fp_estat is not None: fp_estat.flush()
    if fp_nstat is not None: fp_nstat.flush()
    
    np.save(out_dir / f"{tag}_meta_recordingId.npy", meta_rec)
    np.save(out_dir / f"{tag}_meta_trackId.npy", meta_track)
    np.save(out_dir / f"{tag}_meta_frame.npy", meta_frame)

    if args.calc_stats:
        print("\n[STATS] Finalizing...")
        m_x, s_x = finalize_stats(s_dyn_cnt, s_dyn_mean, s_dyn_m2)
        m_nb, s_nb = finalize_stats(s_nb_cnt, s_nb_mean, s_nb_m2)
        save_dict = {
            "dyn_ego_mean": m_x, "dyn_ego_std": s_x,
            "dyn_nb_mean": m_nb, "dyn_nb_std": s_nb,
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