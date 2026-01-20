# scripts/scan_nonfinite.py
from pathlib import Path
import torch
from scripts.train import PtWindowDataset, collate_batch, infer_stats_path_from_data_dir, load_stats_npz_strict
from torch.utils.data import DataLoader

def first_bad_locs(x: torch.Tensor, max_show: int = 20):
    bad = torch.nonzero(~torch.isfinite(x), as_tuple=False)
    return bad[:max_show].tolist()

def main():
    pt_dir = Path("data/exiD/data_pt/exid_T2_Tf5_hz5")
    data_dir = Path("data/exiD/exid_T2_Tf5_hz5")        # 네 data_dir
    split_txt = Path("data/exiD/splits/test.txt")   
    use_ego_static = True
    use_nb_static  = True

    stats_path = infer_stats_path_from_data_dir(data_dir, use_ego_static, use_nb_static)
    stats = load_stats_npz_strict(stats_path)

    ds = PtWindowDataset(
        data_dir=pt_dir,
        split_txt=split_txt,
        stats=stats,
        return_meta=True,
        use_ego_static=use_ego_static,
        use_nb_static=use_nb_static,
    )

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_batch)

    bad = []
    for i, batch in enumerate(loader):
        x_ego = batch["x_ego"]
        if not torch.isfinite(x_ego).all():
            meta = batch.get("meta", None)
            print("[BAD]", i, meta)
            print("x_ego bad locs (b,t,d):", first_bad_locs(x_ego))
            # 어떤 feature(d)가 문제인지 빈도도 찍기
            bad = torch.nonzero(~torch.isfinite(x_ego), as_tuple=False)
            d_idx = bad[:, -1]
            uniq, cnt = torch.unique(d_idx, return_counts=True)
            print("bad feature d indices:", list(zip(uniq.tolist(), cnt.tolist())))
    print("total bad:", len(bad))

if __name__ == "__main__":
    main()