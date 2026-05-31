"""
predict.py  —  Run inference on one patient folder.
Outputs a clinical-style report + Grad-CAM images.

Usage:
    python predict.py --data "D:/spine/lab data/Foramina_Detection"
                      --patient 0001
                      --model best_model.pth
                      --output_dir reports/0001
"""

import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

from dataset import parse_xml, get_transform, GRADE_NAMES, LEVEL_NAMES
from train import build_model


# ---------- Grad-CAM ----------

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        # Hook onto the last ResNet layer
        target = model.layer4[-1]
        target.register_forward_hook(self._save_activation)
        target.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, tensor, class_idx):
        self.model.zero_grad()
        output = self.model(tensor)
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_cam(original_pil: Image.Image, cam: np.ndarray,
                bbox: tuple, alpha: float = 0.5) -> Image.Image:
    """Draw Grad-CAM heatmap over the original slice, with bbox rectangle."""
    orig = np.array(original_pil.convert("RGB"))
    h, w = orig.shape[:2]

    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (w, h))
    overlay = cv2.addWeighted(orig[:, :, ::-1], 1 - alpha,
                               heatmap, alpha, 0)

    xmin, ymin, xmax, ymax = bbox
    cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    return Image.fromarray(overlay[:, :, ::-1])


# ---------- Inference ----------

def predict_patient(patient_dir: Path, model, device, transform,
                    output_dir: Path):
    gradcam = GradCAM(model)
    model.eval()

    findings = defaultdict(list)     # level -> list of finding dicts

    for xml_path in sorted(patient_dir.glob("*.xml")):
        records = parse_xml(xml_path)
        if not records:
            continue

        png_path = patient_dir / f"{records[0]['slice_name']}.png"
        if not png_path.exists():
            continue
        original = Image.open(png_path).convert("L")
        w, h = original.size

        for rec in records:
            xmin, ymin, xmax, ymax = rec["bbox"]
            pad = 5
            crop = original.crop((max(0, xmin-pad), max(0, ymin-pad),
                                  min(w, xmax+pad), min(h, ymax+pad)))

            tensor = transform(crop).unsqueeze(0).to(device)

            with torch.enable_grad():
                output = model(tensor)
            probs = F.softmax(output, dim=1)[0].detach().cpu().numpy()
            pred_grade = int(probs.argmax())

            cam = gradcam.generate(tensor, pred_grade)
            vis = overlay_cam(original, cam, rec["bbox"])

            out_name = (f"{rec['slice_name']}_{rec['side']}"
                        f"_{rec['level'].replace('-','')}.png")
            vis.save(output_dir / out_name)

            findings[rec["level"]].append({
                "side":       rec["side"],
                "pred_grade": pred_grade,
                "confidence": float(probs[pred_grade]),
                "probs":      probs.tolist(),
                "cam_file":   out_name,
            })

    return findings


def print_report(patient_id: str, findings: dict):
    print(f"\n{'='*55}")
    print(f"  Patient {patient_id} — Foraminal Stenosis Report")
    print(f"{'='*55}")

    severity_order = {"L5-S1": 0, "L4-L5": 1, "L3-L4": 2,
                      "L2-L3": 3, "L1-L2": 4}
    for level in sorted(findings, key=lambda x: severity_order.get(x, 9)):
        print(f"\n  {level}:")
        for f in findings[level]:
            grade = f["pred_grade"]
            side_full = "Right" if f["side"] == "RFS" else "Left"
            conf = f["confidence"] * 100
            status = "NORMAL" if grade == 0 else f"*** {GRADE_NAMES[grade].upper()} ***"
            print(f"    {side_full:5s}  {status}  ({conf:.0f}% confidence)")

    print(f"\n{'='*55}\n")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    transform = get_transform("test")

    patient_dir = Path(args.data) / args.patient
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    findings = predict_patient(patient_dir, model, device,
                               transform, output_dir)
    print_report(args.patient, findings)
    print(f"Grad-CAM images saved to: {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",       required=True)
    p.add_argument("--patient",    required=True, help="e.g. 0001")
    p.add_argument("--model",      required=True, help="path to best_model.pth")
    p.add_argument("--output_dir", default="reports")
    main(p.parse_args())
