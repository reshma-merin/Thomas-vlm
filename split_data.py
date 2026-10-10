"""
Split captions.jsonl into train / val / test by 1-degree geographic blocks
(so nearby tiles never end up on both sides of the split).

Usage:
  python split_data.py --inp captions.jsonl
Writes: train.jsonl, val.jsonl, test.jsonl
"""
import argparse, json, math, os, random
from collections import Counter, defaultdict


def block_of(group):
    lat, lon = (float(x) for x in group.split("_"))
    return f"{math.floor(lat)}_{math.floor(lon)}"


def smoke(r):
    return "Smoke: none" not in r["caption"]


def assign(blocks, seed, fr=(0.8, 0.1, 0.1)):
    keys = sorted(blocks)
    random.Random(seed).shuffle(keys)
    total = sum(len(blocks[k]) for k in keys)
    cut1, cut2 = fr[0] * total, (fr[0] + fr[1]) * total
    out, seen = {"train": [], "val": [], "test": []}, 0
    for k in keys:
        name = "train" if seen < cut1 else "val" if seen < cut2 else "test"
        out[name] += blocks[k]
        seen += len(blocks[k])
    return out


def score(split, overall):
    # want val/test sizes near 10% and smoke rate near the overall rate
    n = sum(len(v) for v in split.values())
    s = 0.0
    for name in ("val", "test"):
        rows = split[name]
        s += abs(len(rows) / n - 0.10) * 5
        s += abs(sum(smoke(r) for r in rows) / max(1, len(rows)) - overall)
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="captions.jsonl")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("valid")]
    missing = [r["image"] for r in rows if not os.path.exists(r["image"])]
    print(f"{len(rows)} valid captions, {len(missing)} image files missing")
    rows = [r for r in rows if os.path.exists(r["image"])]

    blocks = defaultdict(list)
    for r in rows:
        blocks[block_of(r["group"])].append(r)
    print(f"{len(blocks)} one-degree blocks")

    overall = sum(smoke(r) for r in rows) / len(rows)
    best = min(range(500), key=lambda s: score(assign(blocks, s), overall))
    split = assign(blocks, best)
    print(f"chosen seed: {best} | overall smoke rate: {overall:.1%}\n")

    for name, rs in split.items():
        with open(f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        sm = Counter(r["caption"].split(".")[0] for r in rs)
        burn = Counter(r["caption"].split("Burn scar: ")[1].split(".")[0] for r in rs)
        print(f"{name:5s} n={len(rs):5d} | smoke: {dict(sm)} | burn: {dict(burn)}")

    # safety check: no block appears in two splits
    owner = {}
    for name, rs in split.items():
        for r in rs:
            b = block_of(r["group"])
            assert owner.setdefault(b, name) == name, "block leaked across splits!"
    print("\nno geographic block appears in more than one split")