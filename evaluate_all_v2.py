"""
evaluate_all.py — Final version: all 5 models evaluated on same 4-class basis

Usage:
    python evaluate_all.py --data "D:\spine\lab data\Foramina_Detection"
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.metrics import classification_report, f1_score, accuracy_score
from torchvision.models import (
    resnet50,        ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    convnext_tiny,   ConvNeXt_Tiny_Weights,
    swin_v2_t,       Swin_V2_T_Weights,
    maxvit_t,        MaxVit_T_Weights,
)
import torchvision.transforms as T

from dataset import ForaminaDataset, GRADE_NAMES
from split import split_records

# ── Model builders (all 4-class) ──────────────────────────────────────────────

def build_m1():
    m = resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 4)
    return m

def build_m2():
    m = efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 4)
    return m

def build_m3():
    m = convnext_tiny(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, 4)
    return m

def build_m4():
    m = swin_v2_t(weights=None)
    m.head = nn.Linear(m.head.in_features, 4)
    return m

def build_m5():
    m = maxvit_t(weights=None)
    m.classifier[5] = nn.Linear(m.classifier[5].in_features, 4)
    return m

# ── Shared transform ──────────────────────────────────────────────────────────

STD_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            with autocast("cuda"):
                out = model(batch["image"].to(device))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(batch["grade"].numpy())
    return np.array(all_labels), np.array(all_preds)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    _, _, test_rec = split_records(args.data, seed=42)

    test_ds = ForaminaDataset(test_rec, "test")
    test_ds.transform = STD_TRANSFORM
    loader  = DataLoader(test_ds, batch_size=32,
                         shuffle=False, num_workers=0)

    models_cfg = [
        {"name": "Model 1 — ResNet-50",         "arch": "ResNet-50",
         "epochs": "30", "path": args.m1, "builder": build_m1},
        {"name": "Model 2 — EfficientNet-B0",   "arch": "EfficientNet-B0",
         "epochs": "60", "path": args.m2, "builder": build_m2},
        {"name": "Model 3 — ConvNeXt V2 Tiny",  "arch": "ConvNeXt V2 Tiny",
         "epochs": "50", "path": args.m3, "builder": build_m3},
        {"name": "Model 4 — Swin Transformer V2","arch": "Swin V2 Tiny",
         "epochs": "60", "path": args.m4, "builder": build_m4},
        {"name": "Model 5 — MaxViT Tiny",        "arch": "MaxViT Tiny",
         "epochs": "50", "path": args.m5, "builder": build_m5},
    ]

    results = []

    for cfg in models_cfg:
        print(f"Evaluating {cfg['name']} ...")
        model = cfg["builder"]().to(device)
        try:
            model.load_state_dict(
                torch.load(cfg["path"], map_location=device)
            )
        except FileNotFoundError:
            print(f"  !! Not found: {cfg['path']} — skipping\n")
            continue

        y_true, y_pred = evaluate_model(model, loader, device)

        macro_f1  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
        weighted  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        acc       = accuracy_score(y_true, y_pred)
        per_class = f1_score(y_true, y_pred, average=None,       zero_division=0)
        grade3_f1 = per_class[3]

        results.append({
            "name":      cfg["name"],
            "arch":      cfg["arch"],
            "epochs":    cfg["epochs"],
            "macro_f1":  macro_f1,
            "weighted":  weighted,
            "accuracy":  acc,
            "grade3_f1": grade3_f1,
            "per_class": per_class,
        })

        print(f"  Macro F1={macro_f1:.4f}  "
              f"Accuracy={acc:.4f}  "
              f"Grade3 F1={grade3_f1:.4f}\n")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*78)
    print("  FINAL COMPARISON — ALL 5 MODELS — 4-CLASS — SAME TEST SET")
    print("="*78)
    print(f"  {'Model':<30} {'Ep':<5} {'Macro F1':<12} "
          f"{'Accuracy':<12} {'Severe F1'}")
    print(f"  {'-'*73}")

    best_f1  = max(r["macro_f1"]  for r in results)
    best_acc = max(r["accuracy"]  for r in results)
    best_g3  = max(r["grade3_f1"] for r in results)

    for r in results:
        f1_m  = " ★" if r["macro_f1"]  == best_f1  else "  "
        ac_m  = " ★" if r["accuracy"]  == best_acc else "  "
        g3_m  = " ★" if r["grade3_f1"] == best_g3  else "  "
        print(f"  {r['name']:<30} {r['epochs']:<5} "
              f"{r['macro_f1']:.4f}{f1_m}     "
              f"{r['accuracy']:.4f}{ac_m}     "
              f"{r['grade3_f1']:.4f}{g3_m}")

    print("="*78)
    print("  ★ = best in column\n")

    # ── Per-class breakdown ───────────────────────────────────────────────────
    print("\n" + "="*78)
    print("  PER-CLASS F1 BREAKDOWN (G0=Normal G1=Mild G2=Moderate G3=Severe)")
    print("="*78)
    print(f"  {'Model':<30} {'G0':>8} {'G1':>8} {'G2':>8} {'G3':>8}  {'Macro':>8}")
    print(f"  {'-'*73}")
    for r in results:
        pc = r["per_class"]
        print(f"  {r['name']:<30} "
              f"{pc[0]:>8.4f} {pc[1]:>8.4f} "
              f"{pc[2]:>8.4f} {pc[3]:>8.4f}  "
              f"{r['macro_f1']:>8.4f}")

    print("="*78)
    print("  Evaluation complete.")
    print("="*78 + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--m1", default=r"D:\spine\decideing\models\best_model.pth")
    p.add_argument("--m2", default=r"D:\spine\decideing\models\best_model_v2_4class.pth")
    p.add_argument("--m3", default=r"D:\spine\decideing\models\best_model_v3.pth")
    p.add_argument("--m4", default=r"D:\spine\decideing\models\best_model_v4.pth")
    p.add_argument("--m5", default=r"D:\spine\decideing\models\best_model_v5.pth")
    run(p.parse_args())
