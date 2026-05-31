"""
train.py  —  Train ResNet-50 for 4-class foramina stenosis grading.

Usage:
    python train.py --data "D:/spine/lab data/Foramina_Detection" --epochs 30
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
from sklearn.metrics import classification_report
import numpy as np

from dataset import ForaminaDataset, GRADE_NAMES
from split import split_records


def get_class_weights(records):
    from collections import Counter
    counts = Counter(r["grade"] for r in records)
    total = sum(counts.values())
    weights = [total / (4 * counts[i]) for i in range(4)]
    return torch.tensor(weights, dtype=torch.float)


def build_sampler(records):
    weights = get_class_weights(records)
    sample_w = [weights[r["grade"]].item() for r in records]
    return WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)


def build_model(num_classes=4):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["image"].to(device))
            preds = out.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["grade"].numpy())
    return np.array(all_labels), np.array(all_preds)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_rec, val_rec, test_rec = split_records(args.data, seed=args.seed)

    train_ds = ForaminaDataset(train_rec, split="train")
    val_ds   = ForaminaDataset(val_rec,   split="val")
    test_ds  = ForaminaDataset(test_rec,  split="test")

    sampler = build_sampler(train_rec)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=4)

    model = build_model().to(device)

    class_weights = get_class_weights(train_rec).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
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

        # --- Validate ---
        y_true, y_pred = evaluate(model, val_loader, device)
        report = classification_report(y_true, y_pred,
                                       target_names=GRADE_NAMES,
                                       output_dict=True, zero_division=0)
        macro_f1 = report["macro avg"]["f1-score"]

        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"loss={total_loss/len(train_loader):.4f}  "
              f"val_macro_F1={macro_f1:.4f}")

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> Saved best model (F1={macro_f1:.4f})")

    # --- Test ---
    print("\n=== Test set results ===")
    model.load_state_dict(torch.load(args.save_path, map_location=device))
    y_true, y_pred = evaluate(model, test_loader, device)
    print(classification_report(y_true, y_pred,
                                target_names=GRADE_NAMES, zero_division=0))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",       required=True,
                   help="Path to Foramina_Detection folder")
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--save_path",  default="best_model.pth")
    train(p.parse_args())
