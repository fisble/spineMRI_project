"""
gradcam_all.py — Grad-CAM for all 5 models, saved in separate subfolders

Output structure:
  D:\spine\decideing\gradcam_outputs\
    Model_1_ResNet50\
      0001_IM000002_RFS_L4L5.png
      ...
    Model_2_EfficientNet\
    Model_3_ConvNeXtV2\
    Model_4_SwinV2\
    Model_5_MaxViT\

Usage:
    python gradcam_all.py --data "D:\spine\lab data\Foramina_Detection"
                          --patient 0001
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from torchvision.models import (
    resnet50,        resnet50,
    efficientnet_b0, EfficientNet_B0_Weights,
    convnext_tiny,   ConvNeXt_Tiny_Weights,
    swin_v2_t,       Swin_V2_T_Weights,
    maxvit_t,        MaxVit_T_Weights,
)

from dataset import parse_xml, GRADE_NAMES
from split import split_records

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_NAMES   = [
    "Model_1_ResNet50",
    "Model_2_EfficientNet",
    "Model_3_ConvNeXtV2",
    "Model_4_SwinV2",
    "Model_5_MaxViT",
]

GRADE_LABELS = {
    0: "Normal",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
}

STD = T.Compose([
    T.Resize((224, 224)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

# ── Model builders ─────────────────────────────────────────────────────────────

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

BUILDERS = [build_m1, build_m2, build_m3, build_m4, build_m5]


# ── Grad-CAM ───────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, model_name):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._register(model_name)

    def _register(self, name):
        layer = self._target_layer(name)
        self._fwd_h = layer.register_forward_hook(self._fwd)
        self._bwd_h = layer.register_full_backward_hook(self._bwd)

    def _target_layer(self, name):
        m = self.model
        if   name == "Model_1_ResNet50":      return m.layer4[-1]
        elif name == "Model_2_EfficientNet":  return m.features[-1]
        elif name == "Model_3_ConvNeXtV2":    return m.features[-1]
        elif name == "Model_4_SwinV2":        return m.features[-1]
        elif name == "Model_5_MaxViT":        return m.blocks[-1]
        else: raise ValueError(f"Unknown: {name}")

    def _fwd(self, m, i, o): self.activations = o.detach()
    def _bwd(self, m, gi, go): self.gradients  = go[0].detach()

    def generate(self, tensor, class_idx):
        self.model.zero_grad()
        out = self.model(tensor)
        out[0, class_idx].backward()

        grad = self.gradients
        act  = self.activations

        # Handle transformer outputs [B, tokens, C] → [B, C, H, W]
        if grad.dim() == 3:
            B, T, C = grad.shape
            H = W   = int(T ** 0.5)
            if H * W == T:
                grad = grad.permute(0,2,1).reshape(B,C,H,W)
                act  = act.permute(0,2,1).reshape(B,C,H,W)
            else:
                grad = grad.mean(1, keepdim=True).unsqueeze(-1)
                act  = act.mean(1, keepdim=True).unsqueeze(-1)

        weights = grad.mean(dim=[2,3], keepdim=True)
        cam     = F.relu((weights * act).sum(1, keepdim=True))
        cam     = F.interpolate(cam, (224,224),
                                mode="bilinear", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def remove(self):
        self._fwd_h.remove()
        self._bwd_h.remove()


# ── Overlay ────────────────────────────────────────────────────────────────────

def make_overlay(original_pil, cam, bbox, alpha=0.45):
    """
    Returns a PIL Image with heatmap overlay and green bounding box.
    Fixes the OpenCV non-contiguous array issue.
    """
    # Convert to contiguous RGB numpy array
    orig_rgb = np.ascontiguousarray(
        np.array(original_pil.convert("RGB"))
    )
    h, w = orig_rgb.shape[:2]

    # Build heatmap in BGR then convert
    heat_bgr = cv2.applyColorMap(
        (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heat_bgr = cv2.resize(heat_bgr, (w, h))

    # Blend — work in BGR throughout, convert once at end
    orig_bgr = np.ascontiguousarray(orig_rgb[:, :, ::-1])
    blend    = cv2.addWeighted(orig_bgr, 1 - alpha, heat_bgr, alpha, 0)

    # Draw bounding box on contiguous array
    xmin, ymin, xmax, ymax = bbox
    cv2.rectangle(blend, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Add text label
    label_text = f"GT:{GRADE_LABELS.get(0,'')} Pred:..."
    cv2.putText(blend, label_text, (xmin, max(ymin-8, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

    # Convert BGR → RGB → PIL
    result_rgb = np.ascontiguousarray(blend[:, :, ::-1])
    return Image.fromarray(result_rgb)


def make_overlay_with_label(original_pil, cam, bbox,
                             gt_grade, pred_grade, conf,
                             level, side, alpha=0.45):
    """Full overlay with grade label burned into image."""
    orig_rgb = np.ascontiguousarray(
        np.array(original_pil.convert("RGB"))
    )
    h, w = orig_rgb.shape[:2]

    heat_bgr = cv2.applyColorMap(
        (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heat_bgr = cv2.resize(heat_bgr, (w, h))
    orig_bgr = np.ascontiguousarray(orig_rgb[:, :, ::-1])
    blend    = cv2.addWeighted(orig_bgr, 1-alpha, heat_bgr, alpha, 0)

    # Bounding box — green if correct, red if wrong
    box_color = (0,255,0) if pred_grade==gt_grade else (0,0,255)
    xmin, ymin, xmax, ymax = bbox
    cv2.rectangle(blend, (xmin,ymin), (xmax,ymax), box_color, 2)

    # Label bar at top
    side_str = "Right" if side=="RFS" else "Left"
    label    = (f"{level} {side_str} | "
                f"GT:{GRADE_LABELS[gt_grade]} "
                f"Pred:{GRADE_LABELS[pred_grade]} ({conf:.0f}%)")
    bar_h = 22
    cv2.rectangle(blend, (0,0), (w, bar_h), (0,0,0), -1)
    cv2.putText(blend, label, (4, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)

    result_rgb = np.ascontiguousarray(blend[:,:,::-1])
    return Image.fromarray(result_rgb)


# ── Core: generate Grad-CAM for one model, one patient ─────────────────────────

def run_gradcam_for_model(model, model_name, patient_dir,
                           device, out_dir):
    """
    Loops all XML annotations for a patient,
    generates Grad-CAM overlay for each,
    saves to out_dir/model_name/
    """
    save_dir = out_dir / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    gcam    = GradCAM(model, model_name)
    model.eval()

    # Deduplicate: worst grade per (slice, level, side)
    all_recs = {}
    for xml_path in sorted(patient_dir.glob("*.xml")):
        for rec in parse_xml(xml_path):
            png_p = patient_dir / f"{rec['slice_name']}.png"
            if not png_p.exists():
                continue
            key = (rec["slice_name"], rec["level"], rec["side"])
            if key not in all_recs or rec["grade"] > all_recs[key]["grade"]:
                all_recs[key] = rec
                all_recs[key]["png_path"] = png_p

    count = 0
    for rec in all_recs.values():
        original = Image.open(rec["png_path"]).convert("L")
        w, h     = original.size
        pad      = 5
        xmin, ymin, xmax, ymax = rec["bbox"]
        crop = original.crop((
            max(0, xmin-pad), max(0, ymin-pad),
            min(w, xmax+pad), min(h, ymax+pad)
        ))

        tensor = STD(crop).unsqueeze(0).to(device)

        with torch.enable_grad():
            out = model(tensor)
        probs      = F.softmax(out, dim=1)[0].detach().cpu().numpy()
        pred_grade = int(probs.argmax())
        conf       = probs[pred_grade] * 100
        cam        = gcam.generate(tensor, pred_grade)

        overlay = make_overlay_with_label(
            original, cam, rec["bbox"],
            gt_grade=rec["grade"],
            pred_grade=pred_grade,
            conf=conf,
            level=rec["level"],
            side=rec["side"],
        )

        side_str = rec["side"]
        lvl_str  = rec["level"].replace("-", "")
        fname    = (f"{rec['slice_name']}"
                    f"_{side_str}_{lvl_str}"
                    f"_GT{rec['grade']}_Pred{pred_grade}.png")
        overlay.save(save_dir / fname)
        count += 1

    gcam.remove()
    print(f"  {model_name}: {count} images saved → {save_dir}")
    return count


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Patient: {args.patient}")
    print(f"Output: {out_dir}\n")

    model_paths = [args.m1, args.m2, args.m3, args.m4, args.m5]
    patient_dir = Path(args.data) / args.patient

    if not patient_dir.exists():
        print(f"Patient folder not found: {patient_dir}")
        return

    total = 0
    for name, builder, path in zip(MODEL_NAMES, BUILDERS, model_paths):
        print(f"Loading {name} ...")
        model = builder().to(device)
        try:
            model.load_state_dict(
                torch.load(path, map_location=device)
            )
        except FileNotFoundError:
            print(f"  !! Model file not found: {path} — skipping\n")
            continue

        n = run_gradcam_for_model(
            model, name, patient_dir, device, out_dir
        )
        total += n

    print(f"\nDone — {total} total Grad-CAM images")
    print(f"Saved in subfolders under: {out_dir}")
    print("\nFolder structure:")
    for name in MODEL_NAMES:
        folder = out_dir / name
        if folder.exists():
            files = list(folder.glob("*.png"))
            print(f"  {name}/ — {len(files)} images")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",     required=True)
    p.add_argument("--patient",  default="0001")
    p.add_argument("--m1", default=r"D:\spine\decideing\models\best_model.pth")
    p.add_argument("--m2", default=r"D:\spine\decideing\models\best_model_v2_4class.pth")
    p.add_argument("--m3", default=r"D:\spine\decideing\models\best_model_v3.pth")
    p.add_argument("--m4", default=r"D:\spine\decideing\models\best_model_v4.pth")
    p.add_argument("--m5", default=r"D:\spine\decideing\models\best_model_v5.pth")
    p.add_argument("--output_dir",
                   default=r"D:\spine\decideing\gradcam_outputs")
    main(p.parse_args())
