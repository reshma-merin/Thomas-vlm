"""
Error bars + error analysis for a saved predictions file (no GPU, no API needed).

  python analyze.py --preds preds_ckpt_s2d_test.jsonl --train train.jsonl
Outputs: printed tables + error_sheet.png
"""
import argparse, json, random, re
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PATTERN = re.compile(
    r"^Smoke: (none|thin|dense)\. Cloud cover: (none|partial|heavy)\. "
    r"Land cover: (forest|savanna/grassland|cropland|bare/dry soil|mixed)\. "
    r"Burn scar: (none|possible|clear)\. Water: (visible|not visible)\.$"
)
FIELDS = ["Smoke", "Cloud cover", "Land cover", "Burn scar", "Water"]


def parse(text):
    m = PATTERN.match(text.strip())
    return dict(zip(FIELDS, m.groups())) if m else None


def brighten(path):  # for display only
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    scale = max(float(np.percentile(arr, 99.5)), 0.05)
    arr = np.clip(arr / scale, 0, 1) ** 0.7
    return Image.fromarray((arr * 255).astype(np.uint8))


def prf(ref, pred):
    tp = int(np.sum(ref & pred)); fp = int(np.sum(~ref & pred)); fn = int(np.sum(ref & ~pred))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def ci(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return lo, hi


def main(a):
    rows = [json.loads(l) for l in open(a.preds, encoding="utf-8") if l.strip()]
    train = [parse(json.loads(l)["caption"]) for l in open(a.train, encoding="utf-8") if l.strip()]
    train = [t for t in train if t]
    majority = {f: Counter(t[f] for t in train).most_common(1)[0][0] for f in FIELDS}

    refs = [parse(r["ref"]) for r in rows]
    preds = [parse(r["pred"]) for r in rows]
    n = len(rows)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(a.boot, n))

    print(f"n = {n}, bootstrap resamples = {a.boot}\n")
    print("field".ljust(12), "model acc  [95% CI]".ljust(26), "baseline".ljust(9), "model - baseline  [95% CI]")
    for f in FIELDS:
        model_ok = np.array([bool(p and p[f] == r[f]) for p, r in zip(preds, refs)], dtype=float)
        base_ok = np.array([r[f] == majority[f] for r in refs], dtype=float)
        acc = model_ok[idx].mean(1)
        diff = (model_ok - base_ok)[idx].mean(1)
        lo, hi = ci(acc)
        dlo, dhi = ci(diff)
        flag = "ABOVE baseline" if dlo > 0 else ("BELOW baseline" if dhi < 0 else "not distinguishable")
        print(f"{f.ljust(12)} {model_ok.mean():.3f} [{lo:.3f}, {hi:.3f}]".ljust(40),
              f"{base_ok.mean():.3f}".ljust(9), f"{diff.mean():+.3f} [{dlo:+.3f}, {dhi:+.3f}]  {flag}")

    ref_s = np.array([r["Smoke"] != "none" for r in refs])
    pred_s = np.array([bool(p and p["Smoke"] != "none") for p in preds])
    P, R, F = prf(ref_s, pred_s)
    boots = np.array([prf(ref_s[i], pred_s[i]) for i in idx])
    print(f"\nsmoke present (thin+dense), positives = {int(ref_s.sum())}")
    for name, val, col in (("precision", P, 0), ("recall", R, 1), ("F1", F, 2)):
        lo, hi = ci(boots[:, col])
        print(f"  {name.ljust(9)} {val:.3f}  [{lo:.3f}, {hi:.3f}]")

    levels = ["none", "thin", "dense"]
    print("\nsmoke confusion matrix (rows = reference, columns = prediction)")
    print("          " + "".join(l.rjust(9) for l in levels) + "  invalid")
    for rl in levels:
        counts = [sum(1 for p, r in zip(preds, refs) if p and r["Smoke"] == rl and p["Smoke"] == pl) for pl in levels]
        inv = sum(1 for p, r in zip(preds, refs) if p is None and r["Smoke"] == rl)
        print(f"{rl.ljust(10)}" + "".join(str(c).rjust(9) for c in counts) + f"{inv:9d}")

    # error gallery
    fn = [(r, p) for r, p, rf in zip(rows, preds, refs) if rf["Smoke"] != "none" and not (p and p["Smoke"] != "none")]
    fp = [(r, p) for r, p, rf in zip(rows, preds, refs) if rf["Smoke"] == "none" and p and p["Smoke"] != "none"]
    random.Random(0).shuffle(fn); random.Random(0).shuffle(fp)
    picks = [("MISSED", x) for x in fn[:6]] + [("FALSE ALARM", x) for x in fp[:6]]
    print(f"\nmissed smoke: {len(fn)} | false alarms: {len(fp)} -> showing {len(picks)} of them in {a.sheet}")
    T, LH, cols = 256, 38, 4
    rows_n = (len(picks) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * T, rows_n * (T + LH)), "white")
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()
    for i, (kind, (r, p)) in enumerate(picks):
        x, y = (i % cols) * T, (i // cols) * (T + LH)
        canvas.paste(brighten(r["image"]).resize((T, T)), (x, y))
        ref_sm = parse(r["ref"])["Smoke"]
        pred_sm = p["Smoke"] if p else "invalid"
        d.text((x + 4, y + T + 2), f"{kind}\nref: {ref_sm}   pred: {pred_sm}", fill="black", font=font)
    canvas.save(a.sheet)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--train", default="train.jsonl")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--sheet", default="error_sheet.png")
    main(ap.parse_args())