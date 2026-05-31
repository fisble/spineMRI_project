"""
train_v2.py  —  MODEL 2: EfficientNet-B0, 3 classes, 60 epochs
Changes from Model 1 (train.py):
  1. EfficientNet-B0 instead of ResNet-50
  2. Grades merged: 0=Normal, 1=Stenosis (old 1+2), 2=Severe (old 3)
  3. 60 epochs default
  4. Saves to best_model_v2.pth

Usage:
    python train_v2.py --data "D:/spine/lab data/Foramina_Detection" --epochs 60
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import classification_report
import numpy as np

from dataset import ForaminaDataset, GRADE_NAMES
from split import split_records

# ── Model 2 uses 3 classes instead of 4 ──────────────────────────────────────
NUM_CLASSES  = 3
CLASS_NAMES  = ["Normal (Grade 0)", "Stenosis (Grade 1-2)", "Severe (Grade 3)"]

def remap_grade(grade: int) -> int:
    """Collapse 4-class labels into 3-class labels."""
    if grade == 0:
        return 0          # Normal
    elif grade in (1, 2):
        return 1          # Mild + Moderate → Stenosis
    else:
        return 2          # Severe


class RemappedDataset(torch.utils.data.Dataset):
    """Wraps ForaminaDataset and remaps grade labels to 3 classes."""
    def __init__(self, records, split):
        self.base = ForaminaDataset(records, split)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        item["grade"] = torch.tensor(
            remap_grade(item["grade"].item()), dtype=torch.long
        )
        return item


def get_class_weights(records):
    from collections import Counter
    remapped = [remap_grade(r["grade"]) for r in records]
    counts = Counter(remapped)
    total = sum(counts.values())
    weights = [total / (NUM_CLASSES * counts[i]) for i in range(NUM_CLASSES)]
    return torch.tensor(weights, dtype=torch.float)


def build_sampler(records):
    weights = get_class_weights(records)
    sample_w = [weights[remap_grade(r["grade"])].item() for r in records]
    return WeightedRandomSampler(sample_w, num_samples=len(sample_w),
                                 replacement=True)


def build_model():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, NUM_CLASSES
    )
    return model


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            out   = model(batch["image"].to(device))
            preds = out.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["grade"].numpy())
    return np.array(all_labels), np.array(all_preds)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("MODEL 2 — EfficientNet-B0 | 3 classes | {args.epochs} epochs")
    print("Classes:", CLASS_NAMES)

    train_rec, val_rec, test_rec = split_records(args.data, seed=args.seed)

    train_ds = RemappedDataset(train_rec, split="train")
    val_ds   = RemappedDataset(val_rec,   split="val")
    test_ds  = RemappedDataset(test_rec,  split="test")

    sampler      = build_sampler(train_rec)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=4)

    model     = build_model().to(device)
    class_w   = get_class_weights(train_rec).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_val_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            imgs   = batch["image"].to(device)
            labels = batch["grade"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        y_true, y_pred = evaluate(model, val_loader, device)
        report   = classification_report(y_true, y_pred,
                                         target_names=CLASS_NAMES,
                                         output_dict=True, zero_division=0)
        macro_f1 = report["macro avg"]["f1-score"]

        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"loss={total_loss/len(train_loader):.4f}  "
              f"val_macro_F1={macro_f1:.4f}")

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> Saved best model (F1={macro_f1:.4f})")

    print("\n=== Test set results ===")
    model.load_state_dict(torch.load(args.save_path, map_location=device))
    y_true, y_pred = evaluate(model, test_loader, device)
    print(classification_report(y_true, y_pred,
                                target_names=CLASS_NAMES, zero_division=0))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",       required=True,
                   help="Path to Foramina_Detection folder")
    p.add_argument("--epochs",     type=int,   default=60)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--save_path",  default="D:/spine/decideing/models/best_model_v2.pth")
    train(p.parse_args())
