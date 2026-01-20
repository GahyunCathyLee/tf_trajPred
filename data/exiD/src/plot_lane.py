import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer

OSM_PATH = "./maps/location0.osm"

def get_tags(elem):
    return {t.attrib["k"]: t.attrib.get("v") for t in elem.findall("tag")}

tree = ET.parse(OSM_PATH)
root = tree.getroot()

nodes = root.findall("node")
ways  = root.findall("way")
rels  = root.findall("relation")

# node id -> lat, lon
node_latlon = {}
lats, lons = [], []
for n in nodes:
    nid = int(n.attrib["id"])
    lat = float(n.attrib["lat"])
    lon = float(n.attrib["lon"])
    node_latlon[nid] = (lat, lon)
    lats.append(lat); lons.append(lon)

# UTM projection
lat0, lon0 = np.mean(lats), np.mean(lons)
zone = int((lon0 + 180) // 6) + 1
epsg = (32600 + zone) if lat0 >= 0 else (32700 + zone)
tf = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)

node_xy = {
    nid: tf.transform(lon, lat)
    for nid, (lat, lon) in node_latlon.items()
}

# way id -> node ids
ways_dict = {
    int(w.attrib["id"]): [int(nd.attrib["ref"]) for nd in w.findall("nd")]
    for w in ways
}

plt.figure(figsize=(8, 8))

for r in rels:
    tags = get_tags(r)
    if tags.get("type") != "lanelet":
        continue

    lanelet_id = int(r.attrib["id"])
    left = right = None

    for m in r.findall("member"):
        if m.attrib.get("type") != "way":
            continue
        if m.attrib.get("role") == "left":
            left = int(m.attrib["ref"])
        elif m.attrib.get("role") == "right":
            right = int(m.attrib["ref"])

    if left is None or right is None:
        continue

    # draw boundaries
    centers = []
    for wid in (left, right):
        pts = [node_xy[nid] for nid in ways_dict[wid] if nid in node_xy]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, color="black", linewidth=0.8)
        centers.append(np.mean(pts, axis=0))

    # lane center (for text)
    cx, cy = np.mean(centers, axis=0)
    plt.text(cx, cy, str(lanelet_id), fontsize=6, bbox=dict(fc="white", alpha=0.7))

plt.gca().set_aspect("equal", "box")
plt.title("Lanelets with laneletId")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.show()

plt.tight_layout()
plt.savefig("lanelets_debug.png", dpi=200)
print("[OK] saved lanelets_debug.png")