"""
Masked Autoencoder (MAE) pretraining model built on a ViT encoder.

Reference design: He et al., "Masked Autoencoders Are Scalable Vision Learners" (2021).
Adapted here for 512x512 3-channel Sentinel-2 RGB PNGs.
"""

import torch
import torch.nn as nn
import numpy as np
  # should print your L40

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Standard 2D sin/cos positional embedding (no gradient, fixed)."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # note: w goes first
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])

    def sincos_1d(embed_dim, pos):
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.0
        omega = 1.0 / (10000 ** omega)
        pos = pos.reshape(-1)
        out = np.einsum("m,d->md", pos, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    emb_h = sincos_1d(embed_dim // 2, grid[0])
    emb_w = sincos_1d(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)  # (grid_size*grid_size, embed_dim)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=512, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                  # (B, embed_dim, grid, grid)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x):
        y = self.norm1(x)
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MaskedAutoencoderViT(nn.Module):
    def __init__(
        self,
        img_size=512,
        patch_size=16,
        in_chans=3,
        embed_dim=384,          # ViT-Small
        encoder_depth=12,
        encoder_heads=6,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_heads=6,
        mask_ratio=0.75,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.in_chans = in_chans

        # --- Encoder ---
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        pos_embed = get_2d_sincos_pos_embed(embed_dim, self.patch_embed.grid_size)
        self.register_buffer(
            "pos_embed",
            torch.cat([torch.zeros(1, embed_dim), torch.from_numpy(pos_embed).float()], dim=0).unsqueeze(0),
            persistent=False,
        )
        self.encoder_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, encoder_heads) for _ in range(encoder_depth)]
        )
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # --- Decoder ---
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        decoder_pos_embed = get_2d_sincos_pos_embed(decoder_embed_dim, self.patch_embed.grid_size)
        self.register_buffer(
            "decoder_pos_embed",
            torch.cat(
                [torch.zeros(1, decoder_embed_dim), torch.from_numpy(decoder_pos_embed).float()], dim=0
            ).unsqueeze(0),
            persistent=False,
        )
        self.decoder_blocks = nn.ModuleList(
            [TransformerBlock(decoder_embed_dim, decoder_heads) for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * patch_size * in_chans)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def patchify(self, imgs):
        p = self.patch_size
        B, C, H, W = imgs.shape
        h = w = H // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = torch.einsum("bchpwq->bhwpqc", x)
        x = x.reshape(B, h * w, p * p * C)
        return x

    def random_masking(self, x, mask_ratio):
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, imgs, mask_ratio):
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x[:, 1:, :]  # drop cls token

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # per-patch mean loss
        loss = (loss * mask).sum() / mask.sum()  # only masked patches
        return loss

    def forward(self, imgs, mask_ratio=None):
        mask_ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def vit_small_mae(**kwargs):
    return MaskedAutoencoderViT(
        embed_dim=384, encoder_depth=12, encoder_heads=6,
        decoder_embed_dim=192, decoder_depth=4, decoder_heads=6, **kwargs
    )


def vit_base_mae(**kwargs):
    return MaskedAutoencoderViT(
        embed_dim=768, encoder_depth=12, encoder_heads=12,
        decoder_embed_dim=384, decoder_depth=6, decoder_heads=6, **kwargs
    )