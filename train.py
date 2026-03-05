"""
YOLOSaphire Training Script
========================
Trains YOLOSaphire on a YOLO-format dataset.

Usage:
    python train.py --data dataset/data.yaml --model medium --epochs 100

Dataset structure expected:
    dataset/
    ├── images/train/    ← .jpg / .png files
    ├── images/val/
    ├── labels/train/    ← .txt files (YOLO format)
    ├── labels/val/
    └── data.yaml        ← nc, names, train/val paths
"""

import argparse
import os
import time
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path

from model import yolosaphire_nano, yolosaphire_small, yolosaphire_medium, yolosaphire_large


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class YOLODataset(Dataset):
    """Minimal YOLO-format dataset loader."""

    def __init__(self, img_dir: str, label_dir: str, img_size: int = 640):
        self.img_size = img_size
        self.img_paths = sorted(Path(img_dir).glob("*.jpg")) + \
                         sorted(Path(img_dir).glob("*.png"))
        self.label_dir = Path(label_dir)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        label_path = self.label_dir / (img_path.stem + ".txt")
        labels = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f.readlines():
                    vals = list(map(float, line.strip().split()))
                    if len(vals) == 5:
                        labels.append(vals)  # [cls, x, y, w, h]

        labels = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5))
        return img, labels


def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    return imgs, list(labels)


# ─────────────────────────────────────────────
# LOSS (Simplified for structure — extend for full training)
# ─────────────────────────────────────────────

class YOLOSaphireLoss(nn.Module):
    """
    Simplified YOLOSaphire loss combining:
      - CIoU loss for box regression
      - BCE loss for objectness
      - BCE loss for classification

    Note: For production training, replace with a full
    Task-Aligned Label Assignment (TALA) strategy.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        # preds: list of 3 tensors [(B, 5+nc, H, W), ...]
        # targets: list of label tensors per image
        # Placeholder — returns zero loss for structure demo
        total_loss = sum(p.sum() * 0 for p in preds)
        return total_loss


import torch.nn as nn  # needed for loss class above


# ─────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────

def train(args):
    # Load config
    with open(args.data) as f:
        cfg = yaml.safe_load(f)

    num_classes = cfg["nc"]
    base_path   = Path(args.data).parent

    print(f"\n{'='*55}")
    print(f"  YOLOSaphire Training")
    print(f"{'='*55}")
    print(f"  Classes   : {num_classes} ({cfg.get('names', [])})")
    print(f"  Model     : YOLOSaphire-{args.model.upper()}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Img size  : {args.imgsz}")
    print(f"  Batch     : {args.batch}")
    print(f"{'='*55}\n")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Model
    build = {
        "nano":   yolosaphire_nano,
        "small":  yolosaphire_small,
        "medium": yolosaphire_medium,
        "large":  yolosaphire_large,
    }[args.model]
    model = build(num_classes).to(device)
    print(f"Parameters: {model.count_params():,}\n")

    # Dataset
    train_ds = YOLODataset(
        str(base_path / cfg["train"] / "images") if "images" not in cfg["train"] else str(base_path / cfg["train"]),
        str(base_path / cfg["train"]).replace("images", "labels"),
        args.imgsz,
    )
    val_ds = YOLODataset(
        str(base_path / cfg["val"] / "images") if "images" not in cfg["val"] else str(base_path / cfg["val"]),
        str(base_path / cfg["val"]).replace("images", "labels"),
        args.imgsz,
    )

    train_loader = DataLoader(train_ds, args.batch, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds, args.batch, shuffle=False,
                              num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    # Optimizer — Cosine LR + Warmup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    loss_fn   = YOLOSaphireLoss(num_classes).to(device)

    # Output dir
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"  Epoch [{epoch}/{args.epochs}] "
                      f"Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - t0

        print(f"\nEpoch {epoch}/{args.epochs} — "
              f"Loss: {avg_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Time: {elapsed:.1f}s\n")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": best_loss,
                "num_classes": num_classes,
                "model_variant": args.model,
            }, save_dir / "best.pt")
            print(f"  ✓ Saved best model → {save_dir/'best.pt'}\n")

        # Save last
        torch.save(model.state_dict(), save_dir / "last.pt")

    print(f"\n{'='*55}")
    print(f"  Training complete! Best loss: {best_loss:.4f}")
    print(f"  Weights saved to: {save_dir}")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOSaphire")
    parser.add_argument("--data",     type=str, default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model",    type=str, default="medium",            help="nano|small|medium|large")
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--imgsz",    type=int, default=640)
    parser.add_argument("--batch",    type=int, default=16)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--workers",  type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="runs/train")
    args = parser.parse_args()
    train(args)
