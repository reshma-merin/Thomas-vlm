"""
MAE pretraining loop for the ViT-Small encoder on your 400k Sentinel-2 RGB images.

Adjust ROOT_DIRS, IMG_SIZE, BATCH_SIZE below to match your setup.
"""

import os
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from model_mae import vit_small_mae
from model_dataset import PretrainImageDataset

# ---- Config ----
ROOT_DIRS = [r"C:\Users\reshmamerinthomas\Wildfire\pretraining_dataset"]    
IMG_SIZE = 512
PATCH_SIZE = 16
BATCH_SIZE = 128                   
EPOCHS = 200
BASE_LR = 1.5e-4                     # standard MAE base LR (scaled by batch size below)
WARMUP_EPOCHS = 20
MASK_RATIO = 0.75
NUM_WORKERS = 8
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_EVERY = 10                # epochs
LOG_EVERY = 50                       # steps

# Early stopping: stop if epoch-avg loss doesn't improve by at least MIN_DELTA
# for PATIENCE consecutive epochs. Set PATIENCE = None to disable.
PATIENCE = 15
MIN_DELTA = 1e-4

# Set this to a checkpoint path to resume from it (e.g. after a Ctrl+C).
# Leave as None to start fresh.
RESUME_FROM = r"C:\Users\reshmamerinthomas\Wildfire\pretraining\checkpoints\mae_vit_small_epoch30.pt"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_scheduler(optimizer, steps_per_epoch, epochs, warmup_epochs, last_epoch=-1):
    warmup_steps = steps_per_epoch * warmup_epochs
    total_steps = steps_per_epoch * epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item())

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def main():
    dataset = PretrainImageDataset(ROOT_DIRS, img_size=IMG_SIZE)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        persistent_workers=True,
    )
    print(f"Dataset size: {len(dataset)} images, {len(loader)} steps/epoch")

    model = vit_small_mae(img_size=IMG_SIZE, patch_size=PATCH_SIZE, mask_ratio=MASK_RATIO)
    model = model.to(device)

    # Linear LR scaling rule (standard MAE convention): lr = base_lr * batch_size / 256
    lr = BASE_LR * BATCH_SIZE / 256
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.95))

    start_epoch = 0
    best_loss = float("inf")
    epochs_without_improvement = 0

    if RESUME_FROM is not None:
        print(f"Resuming from checkpoint: {RESUME_FROM}")
        ckpt = torch.load(RESUME_FROM, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]  # this many epochs already completed
        best_loss = ckpt.get("loss", float("inf"))
        epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
        print(f"Resumed at epoch {start_epoch}, best_loss={best_loss}, "
              f"epochs_without_improvement={epochs_without_improvement}")

    # Fast-forward the scheduler to the correct step, whether resuming or not.
    # last_epoch here is a step index (we call scheduler.step() every batch, not every epoch).
    resumed_step = start_epoch * len(loader) - 1
    scheduler = build_scheduler(optimizer, len(loader), EPOCHS, WARMUP_EPOCHS, last_epoch=resumed_step)

    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        epoch_loss_sum = 0.0

        for step, imgs in enumerate(loader, 1):
            imgs = imgs.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                loss, _, _ = model(imgs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            epoch_loss_sum += loss.item()

            if step % LOG_EVERY == 0:
                avg_loss = running_loss / LOG_EVERY
                print(f"Epoch {epoch+1}/{EPOCHS} | Step {step}/{len(loader)} | "
                      f"Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
                running_loss = 0.0

        epoch_time = time.time() - epoch_start
        epoch_avg_loss = epoch_loss_sum / len(loader)
        print(f"Epoch {epoch+1} done in {epoch_time/60:.1f} min | "
              f"Epoch avg loss: {epoch_avg_loss:.4f}")

        if (epoch + 1) % CHECKPOINT_EVERY == 0 or (epoch + 1) == EPOCHS:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"mae_vit_small_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_avg_loss,
                "epochs_without_improvement": epochs_without_improvement,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # --- Early stopping check ---
        if PATIENCE is not None:
            if epoch_avg_loss < best_loss - MIN_DELTA:
                best_loss = epoch_avg_loss
                epochs_without_improvement = 0
                best_path = os.path.join(CHECKPOINT_DIR, "mae_vit_small_best.pt")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "epochs_without_improvement": 0,
                }, best_path)
                print(f"New best loss: {best_loss:.4f} -> saved {best_path}")
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement}/{PATIENCE} epochs "
                      f"(best: {best_loss:.4f})")

                if epochs_without_improvement >= PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch+1}. "
                          f"Best loss {best_loss:.4f} was at epoch {epoch+1-epochs_without_improvement}.")
                    break


if __name__ == "__main__":
    main()