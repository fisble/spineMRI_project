"""
train_v5.py — MODEL 5: MaxViT Tiny
RTX 3050 Laptop (4GB) optimised:
  - AMP (mixed precision)
  - batch_size=8 — MaxViT is the heaviest, 8 is the safe limit for 4GB
  - num_workers=0 — Windows fix
  - gradient accumulation steps=2 — simulates batch=16 without OOM
20 epochs for quick test
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
from torchvision.models import maxvit_t, MaxVit_T_Weights
import torchvision.transforms as T
from sklearn.metrics import classification_report
import numpy as np

from dataset import ForaminaDataset, GRADE_NAMES
from split import split_records

NUM_CLASSES   = 4
ACCUM_STEPS   = 2   # gradient accumulation: effective batch = batch_size * 2


def get_class_weights(records):
    from collections import Counter
    counts = Counter(r["grade"] for r in records)
    total  = sum(counts.values())
    return torch.tensor(
        [total / (NUM_CLASSES * counts[i]) for i in range(NUM_CLASSES)],
        dtype=torch.float
    )


def build_sampler(records):
    weights  = get_class_weights(records)
    sample_w = [weights[r["grade"]].item() for r in records]
    return WeightedRandomSampler(sample_w, len(sample_w), replacement=True)


def build_model():
    model = maxvit_t(weights=MaxVit_T_Weights.IMAGENET1K_V1)
    model.classifier[5] = nn.Linear(
        model.classifier[5].in_features, NUM_CLASSES
    )
    return model


MAXVIT_TRAIN = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(5),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
MAXVIT_VAL = T.Compose([
    T.Resize((224, 224)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            with autocast("cuda"):
                out = model(batch["image"].to(device))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(batch["grade"].numpy())
    return np.array(all_labels), np.array(all_preds)


def train(args):
    device = torch.device("cuda")
    print(f"\nMODEL 5 — MaxViT Tiny | 4-class | {args.epochs} epochs")
    print(f"GPU: {torch.cuda.get_device_name(0)} | AMP + grad accumulation x{ACCUM_STEPS}\n")

    train_rec, val_rec, test_rec = split_records(args.data, seed=42)

    train_ds = ForaminaDataset(train_rec, "train")
    val_ds   = ForaminaDataset(val_rec,   "val")
    test_ds  = ForaminaDataset(test_rec,  "test")

    train_ds.transform = MAXVIT_TRAIN
    val_ds.transform   = MAXVIT_VAL
    test_ds.transform  = MAXVIT_VAL

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=build_sampler(train_rec),
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    model     = build_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=get_class_weights(train_rec).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = GradScaler("cuda")

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            imgs   = batch["image"].to(device)
            labels = batch["grade"].to(device)

            with autocast("cuda"):
                loss = criterion(model(imgs), labels)
                loss = loss / ACCUM_STEPS          # scale loss for accumulation

            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * ACCUM_STEPS

        scheduler.step()

        y_true, y_pred = evaluate(model, val_loader, device)
        report   = classification_report(y_true, y_pred,
                                         target_names=GRADE_NAMES,
                                         output_dict=True, zero_division=0)
        macro_f1 = report["macro avg"]["f1-score"]
        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"loss={total_loss/len(train_loader):.4f}  "
              f"val_macro_F1={macro_f1:.4f}")
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> Saved (F1={macro_f1:.4f})")

    print("\n=== MODEL 5 Test Results ===")
    model.load_state_dict(torch.load(args.save_path, map_location=device))
    y_true, y_pred = evaluate(model, test_loader, device)
    print(classification_report(y_true, y_pred,
                                target_names=GRADE_NAMES, zero_division=0))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",       required=True)
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=5e-5)
    p.add_argument("--save_path",  default=r"D:\spine\decideing\models\best_model_v5.pth")
    train(p.parse_args())
