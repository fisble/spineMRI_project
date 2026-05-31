"""
evaluate_all.py — Run all 5 models on the test set and print a clean comparison table

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
    resnet50,         ResNet50_Weights,
    efficientnet_b0,  EfficientNet_B0_Weights,
    convnext_tiny,    ConvNeXt_Tiny_Weights,
    swin_v2_t,        Swin_V2_T_Weights,
    maxvit_t,         MaxVit_T_Weights,
)
import torchvision.transforms as T

from dataset import ForaminaDataset, GRADE_NAMES
from split import split_records
from train_v2 import RemappedDataset, remap_grade

# ── Model builders ────────────────────────────────────────────────────────────

def build_m1():
    m = resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 4)
    return m

def build_m2():
    m = efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 3)
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

# ── Transforms ────────────────────────────────────────────────────────────────

STD_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Evaluate ─────────────────────────────────────────────────────────────────

def evaluate_model(model, loader, device, remap=False):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            with autocast("cuda"):
                out = model(batch["image"].to(device))
            preds = out.argmax(1).cpu().numpy()
            labels = batch["grade"].numpy()
            if remap:
                labels = np.array([remap_grade(l) for l in labels])
            all_preds.extend(preds)
            all_labels.extend(labels)
    return np.array(all_labels), np.array(all_preds)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    _, _, test_rec = split_records(args.data, seed=42)

    # Standard 4-class test set
    test_ds_4 = ForaminaDataset(test_rec, "test")
    test_ds_4.transform = STD_TRANSFORM

    # 3-class test set (Model 2)
    test_ds_3 = RemappedDataset(test_rec, "test")
    test_ds_3.base.transform = STD_TRANSFORM

    loader_4 = DataLoader(test_ds_4, batch_size=32, shuffle=False, num_workers=0)
    loader_3 = DataLoader(test_ds_3, batch_size=32, shuffle=False, num_workers=0)

    # ── Model configs ─────────────────────────────────────────────────────────
    models_cfg = [
        {
            "name":    "Model 1 — ResNet-50",
            "arch":    "ResNet-50",
            "classes": 4,
            "epochs":  "30",
            "path":    args.m1,
            "builder": build_m1,
            "loader":  loader_4,
            "remap":   False,
            "labels":  GRADE_NAMES,
        },
        {
            "name":    "Model 2 — EfficientNet-B0",
            "arch":    "EfficientNet-B0",
            "classes": 3,
            "epochs":  "60",
            "path":    args.m2,
            "builder": build_m2,
            "loader":  loader_3,
            "remap":   True,
            "labels":  ["Normal (G0)", "Stenosis (G1-2)", "Severe (G3)"],
        },
        {
            "name":    "Model 3 — ConvNeXt V2 Tiny",
            "arch":    "ConvNeXt V2",
            "classes": 4,
            "epochs":  "50",
            "path":    args.m3,
            "builder": build_m3,
            "loader":  loader_4,
            "remap":   False,
            "labels":  GRADE_NAMES,
        },
        {
            "name":    "Model 4 — Swin Transformer V2",
            "arch":    "Swin V2 Tiny",
            "classes": 4,
            "epochs":  "60",
            "path":    args.m4,
            "builder": build_m4,
            "loader":  loader_4,
            "remap":   False,
            "labels":  GRADE_NAMES,
        },
        {
            "name":    "Model 5 — MaxViT Tiny",
            "arch":    "MaxViT Tiny",
            "classes": 4,
            "epochs":  "50",
            "path":    args.m5,
            "builder": build_m5,
            "loader":  loader_4,
            "remap":   False,
            "labels":  GRADE_NAMES,
        },
    ]

    results = []

    for cfg in models_cfg:
        print(f"Evaluating {cfg['name']} ...")
        model = cfg["builder"]().to(device)
        try:
            model.load_state_dict(torch.load(cfg["path"], map_location=device))
        except FileNotFoundError:
            print(f"  !! Model file not found: {cfg['path']} — skipping\n")
            continue

        y_true, y_pred = evaluate_model(
            model, cfg["loader"], device, remap=cfg["remap"]
        )

        macro_f1  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
        weighted  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        acc       = accuracy_score(y_true, y_pred)

        # Per-class F1
        per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        # Grade 3 F1 — last class for 4-class models, last for 3-class too
        grade3_f1 = per_class[-1]

        results.append({
            "name":      cfg["name"],
            "arch":      cfg["arch"],
            "classes":   cfg["classes"],
            "epochs":    cfg["epochs"],
            "macro_f1":  macro_f1,
            "weighted":  weighted,
            "accuracy":  acc,
            "grade3_f1": grade3_f1,
            "per_class": per_class,
            "labels":    cfg["labels"],
            "y_true":    y_true,
            "y_pred":    y_pred,
        })

        print(f"  Macro F1={macro_f1:.4f}  Accuracy={acc:.4f}  "
              f"Grade3/Severe F1={grade3_f1:.4f}\n")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*75)
    print("  FINAL COMPARISON TABLE")
    print("="*75)
    print(f"  {'Model':<28} {'Classes':<8} {'Epochs':<8} "
          f"{'Macro F1':<10} {'Accuracy':<10} {'Severe F1'}")
    print(f"  {'-'*70}")

    best_f1  = max(r["macro_f1"]  for r in results)
    best_acc = max(r["accuracy"]  for r in results)
    best_g3  = max(r["grade3_f1"] for r in results)

    for r in results:
        f1_mark  = " ★" if r["macro_f1"]  == best_f1  else ""
        acc_mark = " ★" if r["accuracy"]  == best_acc else ""
        g3_mark  = " ★" if r["grade3_f1"] == best_g3  else ""
        print(f"  {r['name']:<28} {r['classes']:<8} {r['epochs']:<8} "
              f"{r['macro_f1']:.4f}{f1_mark:<4}  "
              f"{r['accuracy']:.4f}{acc_mark:<4}  "
              f"{r['grade3_f1']:.4f}{g3_mark}")

    print("="*75)
    print("  ★ = best in column\n")

    # ── Per-class breakdown ───────────────────────────────────────────────────
    print("\n" + "="*75)
    print("  PER-CLASS F1 BREAKDOWN")
    print("="*75)
    for r in results:
        print(f"\n  {r['name']} ({r['classes']}-class):")
        for label, score in zip(r["labels"], r["per_class"]):
            bar = "█" * int(score * 20)
            print(f"    {label:<25} {score:.4f}  {bar}")

    print("\n" + "="*75)
    print("  Evaluation complete.")
    print("="*75 + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True,
                   help="Path to Foramina_Detection folder")
    p.add_argument("--m1", default=r"D:\spine\decideing\models\best_model.pth")
    p.add_argument("--m2", default=r"D:\spine\decideing\models\best_model_v2.pth")
    p.add_argument("--m3", default=r"D:\spine\decideing\models\best_model_v3.pth")
    p.add_argument("--m4", default=r"D:\spine\decideing\models\best_model_v4.pth")
    p.add_argument("--m5", default=r"D:\spine\decideing\models\best_model_v5.pth")
    run(p.parse_args())
