"""
Evaluate a THOMAS-VLM checkpoint on val.jsonl / test.jsonl.

  python evaluate.py --ckpt ckpt_s1 --data val.jsonl
  python evaluate.py --ckpt ckpt_s2 --data test.jsonl

Reports, per field: accuracy, macro-F1, and the accuracy of a "always predict the most
common training label" baseline. The model must clearly beat that baseline to count as
actually using the image. Also reports smoke-present precision / recall / F1.
Saves predictions to preds_<ckpt>_<data>.jsonl
"""
import argparse, json, os, re
from collections import Counter

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from thomas_vlm import (DEV, LLM_ID, PROMPT, TRANSFORM, load_for_inference)

PATTERN = re.compile(
    r"^Smoke: (none|thin|dense)\. Cloud cover: (none|partial|heavy)\. "
    r"Land cover: (forest|savanna/grassland|cropland|bare/dry soil|mixed)\. "
    r"Burn scar: (none|possible|clear)\. Water: (visible|not visible)\.$"
)
FIELDS = ["Smoke", "Cloud cover", "Land cover", "Burn scar", "Water"]


def parse(text):
    m = PATTERN.match(text.strip())
    return dict(zip(FIELDS, m.groups())) if m else None


class ImgData(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return TRANSFORM(Image.open(self.rows[i]["image"]).convert("RGB"))


def macro_f1(refs, preds):
    labels = sorted(set(refs))
    f1s = []
    for lab in labels:
        tp = sum(r == lab and p == lab for r, p in zip(refs, preds))
        fp = sum(r != lab and p == lab for r, p in zip(refs, preds))
        fn = sum(r == lab and p != lab for r, p in zip(refs, preds))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


@torch.no_grad()
def main(a):
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[: a.limit]
    train_rows = [json.loads(l) for l in open(a.train, encoding="utf-8") if l.strip()]
    base = {f: Counter() for f in FIELDS}
    for r in train_rows:
        p = parse(r["caption"])
        if p:
            for f in FIELDS:
                base[f][p[f]] += 1
    majority = {f: base[f].most_common(1)[0][0] for f in FIELDS}

    tok = AutoTokenizer.from_pretrained(LLM_ID)
    model = load_for_inference(a.ckpt)
    prompt = tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(DEV)

    texts = []
    dl = DataLoader(ImgData(rows), batch_size=a.bs, num_workers=0)
    for bi, px in enumerate(dl):
        with torch.autocast(DEV, dtype=torch.bfloat16):
            out = model.generate(px.to(DEV), ids.expand(px.size(0), -1), max_new_tokens=80)
        texts += [t.strip() for t in tok.batch_decode(out, skip_special_tokens=True)]
        print(f"  {len(texts)}/{len(rows)}", flush=True)

    ckpt_name = os.path.basename(os.path.normpath(a.ckpt))
    data_name = os.path.splitext(os.path.basename(a.data))[0]
    name = "preds_" + ckpt_name + "_" + data_name + ".jsonl"
    with open(name, "w", encoding="utf-8") as f:
        for r, t in zip(rows, texts):
            f.write(json.dumps({"image": r["image"], "ref": r["caption"], "pred": t}) + "\n")

    refs = [parse(r["caption"]) for r in rows]
    preds = [parse(t) for t in texts]
    valid = sum(p is not None for p in preds)
    print(f"\nvalid-format predictions: {valid}/{len(rows)}")

    print("\n" + "field".ljust(12) + " " + "acc".rjust(6) + " " + "macroF1".rjust(8) + " | " + "baseline acc".rjust(12))
    for f in FIELDS:
        rf = [r[f] for r in refs]
        pf = [(p[f] if p else "INVALID") for p in preds]
        acc = sum(x == y for x, y in zip(rf, pf)) / len(rf)
        bacc = sum(x == majority[f] for x in rf) / len(rf)
        print(f"{f:12s} {acc:6.3f} {macro_f1(rf, pf):8.3f} | {bacc:12.3f}   (majority = {majority[f]})")

    ref_s = [r["Smoke"] != "none" for r in refs]
    pred_s = [(p["Smoke"] != "none") if p else False for p in preds]
    tp = sum(r and p for r, p in zip(ref_s, pred_s))
    fp = sum((not r) and p for r, p in zip(ref_s, pred_s))
    fn = sum(r and (not p) for r, p in zip(ref_s, pred_s))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"\nsmoke present (thin+dense): n_pos={sum(ref_s)}  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}")
    print("saved", name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="val.jsonl")
    ap.add_argument("--train", default="train.jsonl")  # for the majority baseline
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--limit", type=int)
    main(ap.parse_args())