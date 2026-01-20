import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd

# ---------- utils ----------
def smart_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, sep=";", low_memory=False)

# ---------- parse osm ----------
def parse_lanelet2_osm(osm_path: Path, location_id: int) -> pd.DataFrame:
    tree = ET.parse(osm_path)
    root = tree.getroot()

    rows = []
    for rel in root.findall("relation"):
        rel_id = rel.get("id")
        if rel_id is None:
            continue

        tags = {t.get("k"): t.get("v") for t in rel.findall("tag") if t.get("k")}
        if tags.get("type") != "lanelet":
            continue

        if tags.get("onramp") == "yes":
            ramp_type = "onramp"
        elif tags.get("offramp") == "yes":
            ramp_type = "offramp"
        else:
            ramp_type = "none"

        rows.append({
            "locationId": location_id,
            "laneletId": int(rel_id),
            "ramp_type": ramp_type
        })

    return pd.DataFrame(rows)

# ---------- build lookup ----------
def build_ramp_lookup(maps_root: Path, num_locations=7) -> pd.DataFrame:
    all_parts = []

    for loc_id in range(num_locations):
        matches = list(maps_root.glob(f"**/location{loc_id}.osm"))
        if not matches:
            raise FileNotFoundError(f"location{loc_id}.osm not found under {maps_root}")

        osm_path = matches[0]
        df = parse_lanelet2_osm(osm_path, loc_id)
        all_parts.append(df)

    return pd.concat(all_parts, ignore_index=True)

# ---------- main ----------
def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    data_dir = project_root / "raw"
    maps_root = project_root / "maps" / "lanelet2"

    # lanelet → ramp_type lookup (한 번만 생성)
    ramp_lookup = build_ramp_lookup(maps_root)

    # 00 ~ 92
    for i in range(0, 93):
        xx = f"{i:02d}"

        tracks_path = data_dir / f"{xx}_tracks.csv"
        meta_path   = data_dir / f"{xx}_recordingMeta.csv"

        if not tracks_path.exists():
            print(f"[SKIP] {tracks_path.name} not found")
            continue
        if not meta_path.exists():
            print(f"[SKIP] {meta_path.name} not found")
            continue

        print(f"[PROCESS] {xx}")

        tracks = smart_read_csv(tracks_path)
        recmeta = smart_read_csv(meta_path)[["recordingId", "locationId"]]

        # recordingId → locationId
        tracks = tracks.merge(recmeta, on="recordingId", how="left")

        # (locationId, laneletId) → ramp_type
        tracks["laneletId"] = tracks["laneletId"].astype("Int64")
        tracks = tracks.merge(
            ramp_lookup,
            on=["locationId", "laneletId"],
            how="left"
        )

        tracks["ramp_type"] = tracks["ramp_type"].fillna("not_found")

        tracks.to_csv(tracks_path, index=False)

        print(tracks["ramp_type"].value_counts())

    print("DONE.")

if __name__ == "__main__":
    main()