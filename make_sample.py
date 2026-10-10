"""
Pick a diverse subset of your existing images to caption.

Usage:
  python make_sample.py --images_dir "D:\\sentinel_images" --n 4000 --max_per_group 5 --out sample.jsonl

Output (one per line):  {"image": "<abs path>", "group": "<event/location key>"}

EDIT group_of() so that all tiles from the same fire event / location share one key.
That key is used later to split train/val/test without leakage.
"""
import argparse, json, os, random
from collections import defaultdict


def group_of(path):
    # --- EDIT ME to match your filenames ---
    # Example filename: fire_38.512_-120.331_2021-08-14.png  ->  group "38.5_-120.3"
    name = os.path.splitext(os.path.basename(path))[0]
    parts = name.split("_")
    try:
        lat, lon = float(parts[-3]), float(parts[-2])
        return f"{round(lat, 1)}_{round(lon, 1)}"
    except (ValueError, IndexError):
        # fallback: parent folder name, else filename prefix
        parent = os.path.basename(os.path.dirname(path))
        return parent or name.rsplit("_", 1)[0]


ap = argparse.ArgumentParser()
ap.add_argument("--images_dir", required=True)
ap.add_argument("--n", type=int, default=4000)
ap.add_argument("--max_per_group", type=int, default=5)
ap.add_argument("--out", default="sample.jsonl")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

random.seed(a.seed)
by_group = defaultdict(list)
for root, _, files in os.walk(a.images_dir):
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            p = os.path.join(root, f)
            by_group[group_of(p)].append(p)

print(f"found {sum(len(v) for v in by_group.values()):,} images in {len(by_group):,} groups")

# cap per group so one big fire doesn't dominate, then sample across groups
pool = []
for g, paths in by_group.items():
    random.shuffle(paths)
    pool += [(p, g) for p in paths[: a.max_per_group]]
random.shuffle(pool)
pool = pool[: a.n]

with open(a.out, "w", encoding="utf-8") as f:
    for p, g in pool:
        f.write(json.dumps({"image": os.path.abspath(p), "group": g}) + "\n")
print(f"wrote {len(pool):,} rows to {a.out}")