"""
THOMAS-VLM: your pretrained MAE encoder (model_mae.py + mae_vit_small_best.pt)
            -> MLP projector -> small LLM (Qwen2.5-0.5B-Instruct) with LoRA.

Needs the thomas_model\\ folder (model_mae.py + mae_vit_small_best.pt) next to this file.
Data: train.jsonl / val.jsonl with {"image": ..., "caption": ...}

Stage 1 (projector only; encoder + LLM frozen):
  python thomas_vlm.py train --stage 1 --train train.jsonl --val val.jsonl --out ckpt_s1 --lr 1e-3 --epochs 2

Stage 2 (LoRA on LLM + projector keeps training):
  python thomas_vlm.py train --stage 2 --init_proj ckpt_s1\\proj.pt --train train.jsonl --val val.jsonl --out ckpt_s2 --lr 1e-4 --epochs 3

Quick smoke test (64 images):
  python thomas_vlm.py train --stage 1 --train train.jsonl --limit 64 --out ckpt_tiny --lr 1e-3 --epochs 5 --bs 4 --accum 1 --print_every 5

Inference:
  python thomas_vlm.py infer --ckpt ckpt_s2 --image "C:\\path\\to\\tile.png"
"""
import argparse, json, math, os, sys
import torch, torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

sys.path.insert(0, "thomas_model")
from model_mae import vit_small_mae

MAE_CKPT = "thomas_model/mae_vit_small_best.pt"
LLM_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Describe this satellite image for wildfire monitoring."
POOL = 2  # 32x32 patch grid -> 16x16 = 256 visual tokens for the LLM
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# SAME preprocessing as pretraining (model_dataset.py): resize to 512, scale to [0,1], no mean/std. No flip.
TRANSFORM = T.Compose([T.Resize((512, 512)), T.ToTensor()])


class ThomasVLM(nn.Module):
    def __init__(self, unfreeze_blocks=0):
        super().__init__()
        self.unfreeze = unfreeze_blocks
        mae = vit_small_mae()
        ck = torch.load(MAE_CKPT, map_location="cpu", weights_only=False)
        sd = {k.replace("module.", "", 1).replace("_orig_mod.", "", 1): v for k, v in ck["model_state_dict"].items()}
        mae.load_state_dict(sd, strict=True)
        self.vision = mae.requires_grad_(False).eval()
        if unfreeze_blocks:
            for blk in self.vision.encoder_blocks[-unfreeze_blocks:]:
                blk.requires_grad_(True)
            self.vision.encoder_norm.requires_grad_(True)
        self.llm = AutoModelForCausalLM.from_pretrained(LLM_ID, torch_dtype=torch.bfloat16)
        d_v, d_l = 384, self.llm.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(d_v, d_l), nn.GELU(), nn.Linear(d_l, d_l))

    def encoder_ft_state(self):
        n = len(self.vision.encoder_blocks)
        keep = tuple(f"encoder_blocks.{i}." for i in range(n - self.unfreeze, n)) + ("encoder_norm.",)
        return {k: v.cpu() for k, v in self.vision.state_dict().items() if k.startswith(keep)}

    def add_lora(self):
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGETS, task_type="CAUSAL_LM")
        self.llm = get_peft_model(self.llm, cfg)

    def encode(self, imgs):
        """Encoder on the FULL image: no masking, no shuffling (MAE's random_masking would scramble patch order)."""
        m = self.vision
        k = len(m.encoder_blocks) - self.unfreeze  # blocks [0, k) stay frozen
        with torch.no_grad():
            x = m.patch_embed(imgs) + m.pos_embed[:, 1:, :]
            cls = (m.cls_token + m.pos_embed[:, :1, :]).expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)
            for blk in m.encoder_blocks[:k]:
                x = blk(x)
        for blk in m.encoder_blocks[k:]:  # empty when nothing is unfrozen
            x = blk(x)
        x = m.encoder_norm(x)[:, 1:]  # (B, 1024, 384), drop CLS
        B, N, D = x.shape
        g = int(N ** 0.5)
        x = x.transpose(1, 2).reshape(B, D, g, g)
        x = nn.functional.avg_pool2d(x, POOL).flatten(2).transpose(1, 2)  # (B, 256, 384)
        return self.proj(x)

    def embed(self, imgs, input_ids):
        vis = self.encode(imgs)
        txt = self.llm.get_input_embeddings()(input_ids)
        return torch.cat([vis.to(txt.dtype), txt], dim=1), vis.size(1)

    def forward(self, imgs, input_ids, attention_mask, labels):
        emb, n_img = self.embed(imgs, input_ids)
        ones = torch.ones(emb.size(0), n_img, dtype=attention_mask.dtype, device=emb.device)
        pad = torch.full((labels.size(0), n_img), -100, dtype=labels.dtype, device=labels.device)
        out = self.llm(inputs_embeds=emb, attention_mask=torch.cat([ones, attention_mask], 1), labels=torch.cat([pad, labels], 1))
        return out.loss

    @torch.no_grad()
    def generate(self, imgs, prompt_ids, max_new_tokens=80):
        emb, _ = self.embed(imgs, prompt_ids)
        mask = torch.ones(emb.shape[:2], dtype=torch.long, device=emb.device)
        return self.llm.generate(inputs_embeds=emb, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False)


class CaptionData(Dataset):
    def __init__(self, path, tok, limit=None, smoke_repeat=1):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        if limit:
            self.rows = self.rows[:limit]
        if smoke_repeat > 1:  # show smoke tiles more often (training set only)
            extra = [r for r in self.rows if "Smoke: none" not in r["caption"]]
            self.rows = self.rows + extra * (smoke_repeat - 1)
        self.tok = tok
        self.prompt = tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        px = TRANSFORM(Image.open(r["image"]).convert("RGB"))
        p = self.tok(self.prompt, add_special_tokens=False)["input_ids"]
        c = self.tok(r["caption"] + self.tok.eos_token, add_special_tokens=False)["input_ids"]
        return px, p, c


def collate(batch, pad_id):
    px = torch.stack([b[0] for b in batch])
    n = max(len(b[1]) + len(b[2]) for b in batch)
    ids, att, lab = [], [], []
    for _, p, c in batch:
        seq, k = p + c, n - len(p) - len(c)
        ids.append(seq + [pad_id] * k)
        att.append([1] * len(seq) + [0] * k)
        lab.append([-100] * len(p) + c + [-100] * k)  # loss on the caption only
    return px, torch.tensor(ids), torch.tensor(att), torch.tensor(lab)


def train(a):
    tok = AutoTokenizer.from_pretrained(LLM_ID)
    model = ThomasVLM(unfreeze_blocks=a.unfreeze_blocks)
    if a.stage == 2:
        model.proj.load_state_dict(torch.load(a.init_proj, map_location="cpu"))
        model.add_lora()
    else:
        model.llm.requires_grad_(False)
    model.to(DEV)
    model.vision.eval()

    col = lambda b: collate(b, tok.pad_token_id)
    tl = DataLoader(CaptionData(a.train, tok, a.limit, a.smoke_repeat), batch_size=a.bs, shuffle=True, collate_fn=col, num_workers=a.workers)
    vl = DataLoader(CaptionData(a.val, tok), batch_size=a.bs, collate_fn=col, num_workers=a.workers) if a.val else None

    enc_params = [p for p in model.vision.parameters() if p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("vision.")]
    params = other_params + enc_params
    print(f"trainable params: {sum(p.numel() for p in params):,} (encoder: {sum(p.numel() for p in enc_params):,}) | train batches/epoch: {len(tl)}")
    groups = [{"params": other_params, "lr": a.lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": a.enc_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    total = max(1, a.epochs * math.ceil(len(tl) / a.accum))
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))
    )

    step = 0
    for ep in range(a.epochs):
        model.train(); model.vision.eval()
        for i, (px, ids, att, lab) in enumerate(tl):
            with torch.autocast(DEV, dtype=torch.bfloat16):
                loss = model(px.to(DEV), ids.to(DEV), att.to(DEV), lab.to(DEV)) / a.accum
            loss.backward()
            if (i + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step == 1 or step % a.print_every == 0:
                    print(f"ep {ep} step {step}/{total} loss {loss.item() * a.accum:.4f}", flush=True)
        if vl:
            model.eval(); tot = 0.0
            with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16):
                for px, ids, att, lab in vl:
                    tot += model(px.to(DEV), ids.to(DEV), att.to(DEV), lab.to(DEV)).item()
            print(f"== epoch {ep} val loss {tot / len(vl):.4f}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    torch.save(model.proj.state_dict(), os.path.join(a.out, "proj.pt"))
    if a.stage == 2:
        model.llm.save_pretrained(a.out)  # LoRA adapter only
    if a.unfreeze_blocks:
        torch.save({"unfreeze": a.unfreeze_blocks, "state": model.encoder_ft_state()}, os.path.join(a.out, "encoder_ft.pt"))
    print("saved to", a.out)


def load_for_inference(ckpt):
    model = ThomasVLM()
    model.proj.load_state_dict(torch.load(os.path.join(ckpt, "proj.pt"), map_location="cpu"))
    enc_path = os.path.join(ckpt, "encoder_ft.pt")
    if os.path.exists(enc_path):
        d = torch.load(enc_path, map_location="cpu")
        res = model.vision.load_state_dict(d["state"], strict=False)
        assert not res.unexpected_keys, res.unexpected_keys
        model.unfreeze = 0  # inference: everything runs under no_grad anyway
    if os.path.exists(os.path.join(ckpt, "adapter_config.json")):
        model.llm = PeftModel.from_pretrained(model.llm, ckpt)
    return model.to(DEV).eval()


def infer(a):
    tok = AutoTokenizer.from_pretrained(LLM_ID)
    model = load_for_inference(a.ckpt)
    px = TRANSFORM(Image.open(a.image).convert("RGB")).unsqueeze(0).to(DEV)
    prompt = tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(DEV)
    with torch.autocast(DEV, dtype=torch.bfloat16):
        out = model.generate(px, ids)
    print(tok.decode(out[0], skip_special_tokens=True).strip())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--stage", type=int, choices=[1, 2], required=True)
    t.add_argument("--train", required=True)
    t.add_argument("--val")
    t.add_argument("--out", required=True)
    t.add_argument("--init_proj")
    t.add_argument("--limit", type=int)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--bs", type=int, default=8)
    t.add_argument("--accum", type=int, default=2)
    t.add_argument("--print_every", type=int, default=10)
    t.add_argument("--unfreeze_blocks", type=int, default=0)  # fine-tune the last N encoder blocks
    t.add_argument("--enc_lr", type=float, default=2e-5)  # learning rate for those blocks
    t.add_argument("--smoke_repeat", type=int, default=1)  # show each smoke tile K times per epoch
    t.add_argument("--workers", type=int, default=0)  # keep 0 on Windows unless you know multiprocessing works
    i = sub.add_parser("infer")
    i.add_argument("--ckpt", required=True)
    i.add_argument("--image", required=True)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else infer(a)