"""
Caption satellite tiles with Claude using a fixed template.

Trial run (20 images):
  python caption.py run --n 20 --out trial.jsonl
  python caption.py sheet --inp trial.jsonl --out trial_sheet.png

Needs ANTHROPIC_API_KEY set in the same window.
Safe to re-run: images already in --out are skipped.
"""
import argparse, base64, io, json, os, random, re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import anthropic

MODEL = "claude-sonnet-5-5"

PATTERN = re.compile(
    r"^Smoke: (none|thin|dense)\. Cloud cover: (none|partial|heavy)\. "
    r"Land cover: (forest|savanna/grassland|cropland|bare/dry soil|mixed)\. "
    r"Burn scar: (none|possible|clear)\. Water: (visible|not visible)\.$"
)

INSTRUCTIONS = """You are labeling Sentinel-2 true-color satellite tiles (512x512 pixels) for wildfire monitoring.
The image has been brightened for visibility. Describe ONLY what is visible in the image.

Fill in these fields:
- Smoke: none = no smoke visible; thin = a faint or small plume; dense = a large, opaque plume.
  Only call smoke when you can see a plume with a visible source, or a streak that clearly traces back to a point on the ground.
  Large bright white billowing masses, uniform haze, and fog with no visible source are CLOUDS, not smoke: use Smoke: none and describe them under Cloud cover.
- Cloud cover: none = ground fully visible; partial = some clouds or haze, but more than half of the ground is visible; heavy = more than about half of the tile is hidden by cloud or haze.
  Large white masses count as cloud: if they cover about half the tile or more, choose heavy.
- Land cover: the dominant type: forest, savanna/grassland, cropland (regular field patterns), bare/dry soil, or mixed.
- Burn scar: none = no burned-looking ground; possible = a diffuse darker, pink, or reddish-brown area with no clear edge; clear = any distinct black or dark-charcoal patch with sharp edges, even if small.
  If a distinct black patch is visible, choose "clear", not "possible". If the ground just looks dry, brown, or dark overall, choose "none".
  Cleared land, roads, settlements, airstrips, plowed fields, and green vegetation are NOT burn scars.
- Water: visible if a river, lake, or pond is visible, otherwise not visible.

Reply with exactly ONE line in this format and nothing else:
Smoke: <none|thin|dense>. Cloud cover: <none|partial|heavy>. Land cover: <forest|savanna/grassland|cropland|bare/dry soil|mixed>. Burn scar: <none|possible|clear>. Water: <visible|not visible>."""


def brighten(path):
    """Brighten for the captioner / review sheet ONLY. Never used for THOMAS training."""
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    scale = max(float(np.percentile(arr, 99.5)), 0.05)  # shared across channels -> keeps color balance
    arr = np.clip(arr / scale, 0, 1) ** 0.7
    return Image.fromarray((arr * 255).astype(np.uint8))


def to_b64_jpeg(img):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def caption_one(client, model, row):
    b64 = to_b64_jpeg(brighten(row["image"]))
    in_tok = out_tok = 0
    text = ""
    for _ in range(3):  # retry if the format is wrong
        msg = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": INSTRUCTIONS},
            ]}],
        )
        in_tok += msg.usage.input_tokens
        out_tok += msg.usage.output_tokens
        blocks = [b.text for b in msg.content if getattr(b, "type", "") == "text"]
        text = blocks[0].strip().splitlines()[0].strip() if blocks and blocks[0].strip() else ""
        if PATTERN.match(text):
            break
    out = dict(row)
    out["caption"] = text
    out["valid"] = bool(PATTERN.match(text))
    return out, in_tok, out_tok


def run(a):
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()]
    random.Random(0).shuffle(rows)  # fixed order: the trial is the first N of the full run
    done = set()
    if os.path.exists(a.out):
        old = [json.loads(l) for l in open(a.out, encoding="utf-8") if l.strip()]
        good = [r for r in old if r.get("valid")]
        if len(good) != len(old):  # remove failed/empty rows so they are retried
            with open(a.out, "w", encoding="utf-8") as f:
                for r in good:
                    f.write(json.dumps(r) + "\n")
            print(f"removed {len(old) - len(good)} invalid rows from {a.out} (will retry them)")
        done = {r["image"] for r in good}
    todo = [r for r in rows if r["image"] not in done][: a.n]
    print(f"{len(done)} already done, captioning {len(todo)} now with {a.model}")

    client = anthropic.Anthropic(max_retries=5)
    tot_in = tot_out = 0
    results = []
    with open(a.out, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(caption_one, client, a.model, r): r for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                out, ti, to = fut.result()
            except Exception as e:  # not written -> will be retried on the next run
                print("FAILED:", futs[fut]["image"], "|", repr(e)[:150])
                continue
            tot_in += ti
            tot_out += to
            results.append(out)
            f.write(json.dumps(out) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")

    if not results:
        return
    ok = [r for r in results if r["valid"]]
    print(f"\nvalid format: {len(ok)}/{len(results)}")
    print("Smoke values:", dict(Counter(re.search(r"Smoke: (\w+)", r["caption"]).group(1) for r in ok)))
    print(f"tokens used this run: input {tot_in:,}  output {tot_out:,}  (~{tot_in // len(results)} in / {tot_out // len(results)} out per image)")


def short(caption):
    m = PATTERN.match(caption)
    if not m:
        return ["INVALID FORMAT"]
    s, c, l, b, w = m.groups()
    return [f"Smoke:{s}  Cloud:{c}", f"Land:{l}", f"Burn:{b}  Water:{w}"]


def sheet(a):
    allrows = [json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip() and json.loads(l).get("valid")]
    random.Random(1).shuffle(allrows)
    hard = [r for r in allrows if "Smoke: none" not in r["caption"] or "Burn scar: clear" in r["caption"]]
    rest = [r for r in allrows if r not in hard]
    rows = (hard[: a.max // 2] + rest)[: a.max]  # up to half the sheet = smoke / clear-burn cases
    cols, T, LH = 4, 256, 84
    n_rows = (len(rows) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * T, n_rows * (T + LH)), "white")
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()
    for i, r in enumerate(rows):
        x, y = (i % cols) * T, (i // cols) * (T + LH)
        canvas.paste(brighten(r["image"]).resize((T, T)), (x, y))
        lines = [f"#{i + 1}"] + short(r["caption"])
        d.text((x + 4, y + T + 2), "\n".join(lines), fill="black", font=font)
    canvas.save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--inp", default="sample.jsonl")
    r.add_argument("--out", required=True)
    r.add_argument("--n", type=int, default=20)
    r.add_argument("--model", default=MODEL)
    r.add_argument("--workers", type=int, default=4)
    s = sub.add_parser("sheet")
    s.add_argument("--inp", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--max", type=int, default=20)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else sheet(a)