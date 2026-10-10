"""
Sanity-check the THOMAS MAE encoder (custom model_mae.py + mae_vit_small_best.pt).

Run from the thomas-vlm folder (needs the thomas_model\\ folder next to this file):
  python check_thomas.py "C:\\path\\to\\one_image.png"
"""
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "thomas_model")
from model_mae import vit_small_mae

CKPT = "thomas_model/mae_vit_small_best.pt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_thomas():
    model = vit_small_mae()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    sd = {k.replace("module.", "", 1).replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    print("load_state_dict:", model.load_state_dict(sd, strict=True))
    print("checkpoint epoch:", ck.get("epoch"), "| saved loss:", ck.get("loss"))
    return model.eval(), ck


@torch.no_grad()
def encode_full(mae, imgs):
    """Encoder on the FULL image, no masking, no shuffling. Returns (B, 1024, 384) patch features."""
    x = mae.patch_embed(imgs) + mae.pos_embed[:, 1:, :]
    cls = (mae.cls_token + mae.pos_embed[:, :1, :]).expand(x.size(0), -1, -1)
    x = torch.cat([cls, x], dim=1)
    for blk in mae.encoder_blocks:
        x = blk(x)
    x = mae.encoder_norm(x)
    return x[:, 1:]  # drop CLS


def load_image(path):
    img = Image.open(path).convert("RGB")
    print("raw image size:", img.size)
    if img.size != (512, 512):
        img = img.resize((512, 512))
    # NOTE: scaling here is just [0,1]. Must match model_dataset.py -- we will confirm next.
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return x.unsqueeze(0)


if __name__ == "__main__":
    model, ck = load_thomas()
    model.to(DEV)
    n = sum(p.numel() for p in model.parameters())
    print(f"params: {n:,}  device: {DEV}")

    x = load_image(sys.argv[1]).to(DEV)
    print("input:", tuple(x.shape), "min/max/mean:", x.min().item(), x.max().item(), x.mean().item())

    a = encode_full(model, x)
    b = encode_full(model, x)
    print("features:", tuple(a.shape), "(expected (1, 1024, 384))")
    print("deterministic:", torch.allclose(a, b), "| has NaN:", torch.isnan(a).any().item())
    print("feature mean/std:", a.mean().item(), a.std().item())

    # Are the trained weights really loaded? Compare reconstruction loss vs. a random-init model.
    fresh = vit_small_mae().to(DEV).eval()
    with torch.no_grad():
        torch.manual_seed(0)
        loss_trained, _, _ = model(x, mask_ratio=0.75)
        torch.manual_seed(0)
        loss_random, _, _ = fresh(x, mask_ratio=0.75)
    print(f"recon loss  trained: {loss_trained.item():.4f}   random init: {loss_random.item():.4f}")
    print("-> trained should be clearly LOWER than random init")