"""
visualize_results.py — Complete evaluation visualizations for research paper
Generates:
  1.  Confusion matrices (all 5 models)
  2.  Per-class F1 bar chart
  3.  Macro F1 + Accuracy comparison chart
  4.  ROC curves (one-vs-rest, all models)
  5.  Precision-Recall curves
  6.  Training loss curves (if history CSVs exist)
  7.  Grade distribution pie chart
  8.  Confidence distribution box plots
  9.  Grad-CAM sample grid
  10. Model complexity comparison (params vs F1)

All figures saved to D:\spine\decideing\paper_figures\
High DPI (300) PNG + PDF for paper submission

Usage:
    python visualize_results.py --data "D:\spine\lab data\Foramina_Detection"
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix, f1_score, accuracy_score,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, classification_report
)
from sklearn.preprocessing import label_binarize

from torchvision.models import (
    resnet50, efficientnet_b0, convnext_tiny, swin_v2_t, maxvit_t
)
import torchvision.transforms as T

from dataset import ForaminaDataset, GRADE_NAMES
from split import split_records

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

COLORS   = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
GRADE_SHORT = ["G0\nNormal", "G1\nMild", "G2\nModerate", "G3\nSevere"]
MODEL_NAMES = [
    "ResNet-50",
    "EfficientNet-B0",
    "ConvNeXt V2",
    "Swin V2",
    "MaxViT"
]

# ── Model builders ─────────────────────────────────────────────────────────────

def build_models():
    def m1():
        m = resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 4)
        return m
    def m2():
        m = efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 4)
        return m
    def m3():
        m = convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 4)
        return m
    def m4():
        m = swin_v2_t(weights=None)
        m.head = nn.Linear(m.head.in_features, 4)
        return m
    def m5():
        m = maxvit_t(weights=None)
        m.classifier[5] = nn.Linear(m.classifier[5].in_features, 4)
        return m
    return [m1, m2, m3, m4, m5]

# ── Evaluate ───────────────────────────────────────────────────────────────────

STD = T.Compose([
    T.Resize((224, 224)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            with autocast("cuda"):
                out = model(batch["image"].to(device))
            probs  = F.softmax(out, dim=1).cpu().numpy()
            preds  = probs.argmax(1)
            labels = batch["grade"].numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)
    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs))


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Confusion matrices (2x3 grid, all 5 models)
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(all_results, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Confusion Matrices — All 5 Models", fontsize=15, fontweight="bold", y=1.01)

    for idx, (name, y_true, y_pred, _, _) in enumerate(all_results):
        ax  = axes[idx // 3][idx % 3]
        cm  = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(cm_norm, annot=False, fmt=".2f", cmap="Blues",
                    xticklabels=["G0","G1","G2","G3"],
                    yticklabels=["G0","G1","G2","G3"],
                    linewidths=0.5, linecolor="white",
                    cbar_kws={"shrink": 0.8}, ax=ax)

        # Annotate cells with count + percentage
        for i in range(4):
            for j in range(4):
                ax.text(j + 0.5, i + 0.5,
                        f"{cm[i,j]}\n({cm_norm[i,j]:.0%})",
                        ha="center", va="center", fontsize=8,
                        color="white" if cm_norm[i,j] > 0.5 else "black")

        macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        ax.set_title(f"{name}\nMacro F1 = {macro:.4f}", fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    # Hide unused subplot (6th cell)
    axes[1][2].set_visible(False)

    plt.tight_layout()
    path = out_dir / "fig1_confusion_matrices.png"
    plt.savefig(path)
    plt.savefig(out_dir / "fig1_confusion_matrices.pdf")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Per-class F1 grouped bar chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_perclass_f1(all_results, out_dir):
    fig, ax = plt.subplots(figsize=(13, 6))

    n_models = len(all_results)
    n_classes = 4
    x = np.arange(n_classes)
    width = 0.15
    offsets = np.linspace(-(n_models-1)*width/2,
                           (n_models-1)*width/2, n_models)

    for i, (name, y_true, y_pred, _, _) in enumerate(all_results):
        pc = f1_score(y_true, y_pred, average=None, zero_division=0)
        bars = ax.bar(x + offsets[i], pc, width,
                      label=name, color=COLORS[i],
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, pc):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=7.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(GRADE_SHORT)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1-Score")
    ax.set_title("Per-Class F1 Score Comparison — All Models",
                 fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(3.7, 0.51, "0.5 threshold", fontsize=8, color="gray")

    plt.tight_layout()
    path = out_dir / "fig2_perclass_f1.png"
    plt.savefig(path)
    plt.savefig(out_dir / "fig2_perclass_f1.pdf")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Macro F1 + Accuracy + Weighted F1 summary bar
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary_metrics(all_results, out_dir):
    names     = [r[0] for r in all_results]
    macro_f1  = [f1_score(r[1],r[2],average="macro",    zero_division=0) for r in all_results]
    weighted  = [f1_score(r[1],r[2],average="weighted", zero_division=0) for r in all_results]
    accuracy  = [accuracy_score(r[1],r[2]) for r in all_results]

    x     = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13, 6))

    b1 = ax.bar(x - width, macro_f1, width, label="Macro F1",
                color="#2196F3", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x,          weighted, width, label="Weighted F1",
                color="#4CAF50", alpha=0.85, edgecolor="white")
    b3 = ax.bar(x + width,  accuracy, width, label="Accuracy",
                color="#FF9800", alpha=0.85, edgecolor="white")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    # Star on best macro F1
    best_idx = int(np.argmax(macro_f1))
    ax.text(x[best_idx] - width, macro_f1[best_idx] + 0.04,
            "★ Best", ha="center", color="#2196F3",
            fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Macro F1 / Weighted F1 / Accuracy",
                 fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.axhline(0.7, color="red", linestyle="--",
               linewidth=0.8, alpha=0.5)
    ax.text(len(names)-0.5, 0.71, "0.7 target", fontsize=8, color="red")

    plt.tight_layout()
    path = out_dir / "fig3_summary_metrics.png"
    plt.savefig(path)
    plt.savefig(out_dir / "fig3_summary_metrics.pdf")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — ROC curves (best model, one-vs-rest per class)
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(all_results, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("ROC Curves — One-vs-Rest per Class",
                 fontsize=15, fontweight="bold", y=1.01)

    classes = 4
    y_bin_template = None

    for idx, (name, y_true, _, y_probs, _) in enumerate(all_results):
        ax = axes[idx // 3][idx % 3]
        y_bin = label_binarize(y_true, classes=list(range(classes)))

        for c, color, grade in zip(range(classes),
                                    ["#2196F3","#4CAF50","#FF9800","#F44336"],
                                    ["G0 Normal","G1 Mild",
                                     "G2 Moderate","G3 Severe"]):
            fpr, tpr, _ = roc_curve(y_bin[:, c], y_probs[:, c])
            roc_auc     = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=1.8,
                    label=f"{grade} (AUC={roc_auc:.3f})")

        ax.plot([0,1],[0,1], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{name}", fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.fill_between([0,1],[0,1], alpha=0.05, color="gray")

    axes[1][2].set_visible(False)
    plt.tight_layout()
    path = out_dir / "fig4_roc_curves.png"
    plt.savefig(path)
    plt.savefig(out_dir / "fig4_roc_curves.pdf")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Precision-Recall curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_pr_curves(all_results, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Precision-Recall Curves — One-vs-Rest per Class",
                 fontsize=15, fontweight="bold", y=1.01)

    for idx, (name, y_true, _, y_probs, _) in enumerate(all_results):
        ax    = axes[idx // 3][idx % 3]
        y_bin = label_binarize(y_true, classes=list(range(4)))

        for c, color, grade in zip(range(4),
                                    ["#2196F3","#4CAF50","#FF9800","#F44336"],
                                    ["G0 Normal","G1 Mild",
                                     "G2 Moderate","G3 Severe"]):
            prec, rec, _ = precision_recall_curve(y_bin[:,c], y_probs[:,c])
            ap           = average_precision_score(y_bin[:,c], y_probs[:,c])
            ax.plot(rec, prec, color=color, lw=1.8,
                    label=f"{grade} (AP={ap:.3f})")
            ax.fill_between(rec, prec, alpha=0.05, color=color)

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{name}", fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)

    axes[1][2].set_visible(False)
    plt.tight_layout()
    path = out_dir / "fig5_pr_curves.png"
    plt.savefig(path)
    plt.savefig(out_dir / "fig5_pr_curves.pdf")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Grade distribution in dataset
# ══════════════════════════════════════════════════════════════════════════════

def plot_grade_distribution(test_rec, out_dir):
    from collections import Counter

    grades = [r["grade"] for r in test_rec]
    counts = Counter(grades)
    labels = ["G0 Normal", "G1 Mild", "G2 Moderate", "G3 Severe"]
    values = [counts.get(i, 0) for i in range(4)]
    colors = ["#2196F3","#4CAF50","#FF9800","#F44336"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Test Set Grade Distribution", fontsize=14,
                 fontweight="bold")

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax1.set_title("Class Distribution (Pie)")

    # Bar chart with counts
    bars = ax2.bar(labels, values, color=colors,
                   edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1,
                 str(val), ha="center", va="bottom",
                 fontweight="bold", fontsize=11)
    ax2.set_ylabel("Number of Samples")
    ax2.set_title("Class Distribution (Count)")
    ax2.set_ylim(0, max(values) * 1.15)

    plt.tight_layout()
    path = out_dir / "fig6_grade_distribution.png"
    plt.savefig(path)
    plt.savefig(out_dir / "fig6_grade_distribution.pdf")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Model complexity vs performance (params vs macro F1)
# ══════════════════════════════════════════════════════════════════════════════

def plot_complexity_vs_performance(all_results, model_params, out_dir):
    names    = [r[0] for r in all_results]
    macro_f1 = [f1_score(r[1],r[2],average="macro",zero_division=0)
                for r in all_results]
    params   = model_params

    fig, ax = plt.subplots(figsize=(9, 6))

    for i, (name, f1, param, color) in enumerate(
            zip(names, macro_f1, params, COLORS)):
        ax.scatter(param, f1, s=200, color=color,
                   zorder=5, edgecolors="white", linewidths=1.5)
        ax.annotate(name,
                    xy=(param, f1),
                    xytext=(8, 5), textcoords="offset points",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_xlabel("Model Parameters (Millions)", fontsize=12)
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_title("Model Complexity vs Performance\n"
                 "(Bottom-right = efficient; Top-left = accurate but heavy)",
                 fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(0, 1.0)

    # Ideal quadrant annotation
    ax.axhline(0.6, color="green", linestyle=":", alpha=0.4)
    ax.text(max(params)*0.02, 0.61,
            "F1 > 0.6 target", fontsize=8, color="green", alpha=0.7)

    plt.tight_layout()
    path = out_dir / "fig7_complexity_vs_performance.png"
    plt.savefig(path)
    plt.savefig(out_dir / "fig7_complexity_vs_performance.pdf")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Confidence distribution per class (best model)
# ══════════════════════════════════════════════════════════════════════════════

def plot_confidence_distribution(all_results, out_dir):
    # Use best model (highest macro F1)
    best_idx = int(np.argmax([
        f1_score(r[1],r[2],average="macro",zero_division=0)
        for r in all_results
    ]))
    name, y_true, y_pred, y_probs, _ = all_results[best_idx]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(f"Confidence Distribution per Class — {name} (Best Model)",
                 fontsize=13, fontweight="bold")

    grade_names_short = ["Normal (G0)","Mild (G1)","Moderate (G2)","Severe (G3)"]
    colors = ["#2196F3","#4CAF50","#FF9800","#F44336"]

    for c in range(4):
        ax = axes[c]
        # Correct vs incorrect predictions
        mask_true  = (y_true == c) & (y_pred == c)
        mask_false = (y_true == c) & (y_pred != c)

        conf_correct   = y_probs[mask_true,  c]
        conf_incorrect = y_probs[mask_false, c]

        ax.hist(conf_correct,   bins=15, alpha=0.7,
                color=colors[c], label=f"Correct (n={mask_true.sum()})",
                edgecolor="white")
        ax.hist(conf_incorrect, bins=15, alpha=0.5,
                color="gray",    label=f"Wrong   (n={mask_false.sum()})",
                edgecolor="white")

        ax.set_title(grade_names_short[c], fontweight="bold")
        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

    plt.tight_layout()
    path = out_dir / "fig8_confidence_distribution.png"
    plt.savefig(path)
    plt.savefig(out_dir / "fig8_confidence_distribution.pdf")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Heat map: F1 scores across models and classes
# ══════════════════════════════════════════════════════════════════════════════

def plot_f1_heatmap(all_results, out_dir):
    names    = [r[0] for r in all_results]
    f1_matrix = np.zeros((len(all_results), 4))

    for i, (_, y_true, y_pred, _, _) in enumerate(all_results):
        pc = f1_score(y_true, y_pred, average=None, zero_division=0)
        f1_matrix[i] = pc

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(f1_matrix, cmap="RdYlGn",
                   vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(4))
    ax.set_xticklabels(["G0\nNormal","G1\nMild",
                         "G2\nModerate","G3\nSevere"])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)

    for i in range(len(names)):
        for j in range(4):
            val = f1_matrix[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="black" if 0.3 < val < 0.8 else "white")

    plt.colorbar(im, ax=ax, label="F1-Score", shrink=0.8)
    ax.set_title("F1-Score Heatmap — Models vs Grade Classes",
                 fontweight="bold", pad=15)
    ax.set_xlabel("Stenosis Grade")

    plt.tight_layout()
    path = out_dir / "fig9_f1_heatmap.png"
    plt.savefig(path)
    plt.savefig(out_dir / "fig9_f1_heatmap.pdf")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Radar / Spider chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_radar_chart(all_results, out_dir):
    categories = ["G0 F1","G1 F1","G2 F1","G3 F1",
                  "Macro F1","Accuracy"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9),
                            subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=8)

    for i, (name, y_true, y_pred, _, _) in enumerate(all_results):
        pc  = f1_score(y_true, y_pred, average=None, zero_division=0)
        mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        values = list(pc) + [mac, acc]
        values += values[:1]
        ax.plot(angles, values, linewidth=2,
                linestyle="solid", color=COLORS[i], label=name)
        ax.fill(angles, values, color=COLORS[i], alpha=0.1)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
              framealpha=0.9)
    ax.set_title("Model Performance Radar Chart\n"
                 "(Larger area = better overall performance)",
                 fontweight="bold", pad=20)

    plt.tight_layout()
    path = out_dir / "fig10_radar_chart.png"
    plt.savefig(path)
    plt.savefig(out_dir / "fig10_radar_chart.pdf")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output: {out_dir}\n")

    _, _, test_rec = split_records(args.data, seed=42)

    test_ds = ForaminaDataset(test_rec, "test")
    test_ds.transform = STD
    loader  = DataLoader(test_ds, batch_size=32,
                         shuffle=False, num_workers=0)

    model_paths = [args.m1, args.m2, args.m3, args.m4, args.m5]
    builders    = build_models()

    all_results  = []
    model_params = []

    for name, builder, path in zip(MODEL_NAMES, builders, model_paths):
        print(f"Loading {name} from {path} ...")
        model = builder().to(device)
        model_params.append(count_params(model))
        try:
            model.load_state_dict(torch.load(path, map_location=device))
        except FileNotFoundError:
            print(f"  !! Not found — skipping {name}")
            continue
        y_true, y_pred, y_probs = get_predictions(model, loader, device)
        all_results.append((name, y_true, y_pred, y_probs, path))
        macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print(f"  Macro F1 = {macro:.4f}\n")

    if not all_results:
        print("No models loaded — check paths.")
        return

    print("\nGenerating figures ...")
    plot_confusion_matrices(all_results, out_dir)
    plot_perclass_f1(all_results, out_dir)
    plot_summary_metrics(all_results, out_dir)
    plot_roc_curves(all_results, out_dir)
    plot_pr_curves(all_results, out_dir)
    plot_grade_distribution(test_rec, out_dir)
    plot_complexity_vs_performance(all_results, model_params, out_dir)
    plot_confidence_distribution(all_results, out_dir)
    plot_f1_heatmap(all_results, out_dir)
    plot_radar_chart(all_results, out_dir)

    print(f"\n All 10 figures saved to: {out_dir}")
    print("  PNG (screen) + PDF (paper submission) for each figure")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",       required=True)
    p.add_argument("--m1",  default=r"D:\spine\decideing\models\best_model.pth")
    p.add_argument("--m2",  default=r"D:\spine\decideing\models\best_model_v2_4class.pth")
    p.add_argument("--m3",  default=r"D:\spine\decideing\models\best_model_v3.pth")
    p.add_argument("--m4",  default=r"D:\spine\decideing\models\best_model_v4.pth")
    p.add_argument("--m5",  default=r"D:\spine\decideing\models\best_model_v5.pth")
    p.add_argument("--output_dir",
                   default=r"D:\spine\decideing\paper_figures")
    main(p.parse_args())
