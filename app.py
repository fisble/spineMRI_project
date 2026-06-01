"""
app.py — SpineMRI Foraminal Stenosis Grader
Mobile-friendly | Dropdown model selector | PDF report download | Base64 images
Run: streamlit run app.py
"""

import io
import datetime
import base64
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import (
    resnet50, efficientnet_b0, convnext_tiny, swin_v2_t, maxvit_t,
)
from PIL import Image
import numpy as np
import cv2

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ── SAFE IMAGE LOADER (won't crash if file missing) ───────────────────────────
def _b64(path: Path) -> str:
    """Return base64 data-URI for an image, or '' if file not found."""
    try:
        with open(path, "rb") as f:
            ext = path.suffix.lstrip(".").lower()
            mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
            return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"
    except FileNotFoundError:
        return ""

ASSETS = Path("assets")
HERO_SRC   = _b64(ASSETS / "hero.png")   or _b64(ASSETS / "hero.jpg")
GRADE0_SRC = _b64(ASSETS / "grade0.png") or _b64(ASSETS / "grade0.jpg")
GRADE1_SRC = _b64(ASSETS / "grade1.png") or _b64(ASSETS / "grade1.jpg")
GRADE2_SRC = _b64(ASSETS / "grade2.png") or _b64(ASSETS / "grade2.jpg")
GRADE3_SRC = _b64(ASSETS / "grade3.png") or _b64(ASSETS / "grade3.jpg")

# SVG placeholder shown when an image is missing
_PH_SVG = (
    '<svg width="28" height="28" viewBox="0 0 28 28" fill="none" '
    'xmlns="http://www.w3.org/2000/svg" opacity="0.35">'
    '<rect x="2" y="2" width="24" height="24" rx="3" stroke="currentColor" stroke-width="1.5"/>'
    '<circle cx="9" cy="9" r="2.5" stroke="currentColor" stroke-width="1.5"/>'
    '<path d="M2 18 L9 11 L14 16 L19 11 L26 18" stroke="currentColor" '
    'stroke-width="1.5" stroke-linejoin="round"/>'
    '</svg>'
)

def _img_slot(src: str, label: str, border_color: str = "rgba(0,0,0,0.1)") -> str:
    """Return HTML for an image slot: real image if src provided, SVG placeholder otherwise."""
    wrap_style = (
        "width:100%;aspect-ratio:16/9;border-radius:8px;margin-bottom:14px;"
        "overflow:hidden;border:1.5px dashed " + border_color + ";"
    )
    if src:
        return (
            f'<div style="{wrap_style}background:rgba(0,0,0,0.05);">'
            f'<img src="{src}" style="width:100%;height:100%;object-fit:cover;'
            f'display:block;border-radius:6px;" alt="{label}"/>'
            f'</div>'
        )
    return (
        f'<div style="{wrap_style}display:flex;flex-direction:column;'
        f'align-items:center;justify-content:center;gap:6px;'
        f'background:rgba(0,0,0,0.03);">'
        f'{_PH_SVG}'
        f'<span style="font-size:11px;color:rgba(0,0,0,0.3);font-weight:300;">'
        f'{label}</span>'
        f'</div>'
    )

def _hero_img_slot(src: str) -> str:
    if src:
        return (
            '<div class="hero-img-slot" style="padding:0;">'
            f'<img src="{src}" style="width:100%;height:100%;'
            f'object-fit:cover;border-radius:14px;" alt="Sagittal MRI"/>'
            '</div>'
        )
    return (
        '<div class="hero-img-slot">'
        '<svg width="48" height="48" viewBox="0 0 48 48" fill="none" '
        'xmlns="http://www.w3.org/2000/svg" opacity="0.15">'
        '<rect x="6" y="6" width="36" height="36" rx="4" stroke="white" stroke-width="2"/>'
        '<circle cx="18" cy="18" r="4" stroke="white" stroke-width="2"/>'
        '<path d="M6 30 L16 20 L24 28 L32 20 L42 30" stroke="white" stroke-width="2" '
        'stroke-linejoin="round"/>'
        '</svg>'
        '<div style="font-size:12px;color:rgba(255,255,255,0.2);margin-top:6px;text-align:center;">'
        'Place sagittal MRI image<br>in assets/hero.png</div>'
        '</div>'
    )

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpineMRI — Stenosis Grader",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
# NOTE: CSS is a plain string (NOT an f-string) so {} don't need escaping.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f5f4f0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── HERO ── */
.hero {
    background: #0b1628;
    padding: 72px 6vw 60px 6vw;
    display: flex; gap: 40px;
    align-items: center; flex-wrap: wrap;
}
.hero-text { flex: 1; min-width: 280px; }
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(28px, 4vw, 50px);
    font-weight: 800; color: #fff;
    margin: 0 0 16px 0; line-height: 1.08;
}
.hero h1 span { color: #7eb3ff; }
.hero-pills { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 32px; }
.pill {
    font-family: 'Syne', sans-serif;
    font-size: 12px; font-weight: 500;
    padding: 5px 14px; border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.16);
    color: rgba(255,255,255,0.7);
    background: rgba(255,255,255,0.05);
}
.hero-img-slot {
    flex: 0 0 auto; width: clamp(180px, 26vw, 340px);
    aspect-ratio: 3/4; background: #0d1e38;
    border-radius: 14px; overflow: hidden;
    border: 1.5px dashed rgba(255,255,255,0.12);
    display: flex; align-items: center; justify-content: center;
    flex-direction: column; gap: 10px;
    color: rgba(255,255,255,0.2); font-size: 13px;
    font-weight: 300; text-align: center;
}

/* ── GRADE CARD ── */
.grade-card { border-radius: 12px; padding: 22px 18px; }
/* Force text colours inside grade cards — overrides Streamlit global resets */
.grade-card div, .grade-card p, .grade-card h3, .grade-card span:not(.grade-tag) {
    color: inherit !important;
}
.grade-tag {
    font-family: 'Syne', sans-serif;
    font-size: 11px; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    padding: 3px 10px; border-radius: 999px;
    display: inline-block; margin-bottom: 12px; color: #fff;
}
.grade-card h3 {
    font-family: 'Syne', sans-serif;
    font-size: 20px; font-weight: 800; margin-bottom: 8px;
}
.grade-card p { font-size: 13px; line-height: 1.6; font-weight: 300; margin: 0; }

/* ── INFERENCE DARK SECTION ── */
.infer-header { background: #0b1628; padding: 52px 6vw 28px 6vw; }

/* Target the container that follows .infer-header */
div[data-testid="stVerticalBlock"]:has(.infer-header) {
    background: #0b1628;
}
div[data-testid="stVerticalBlock"]:has(.infer-header)
    > div[data-testid="stHorizontalBlock"] {
    padding: 0 6vw 52px 6vw;
    background: #0b1628;
}
div[data-testid="stVerticalBlock"]:has(.infer-header)
    div[data-testid="stVerticalBlock"] {
    background: #0b1628;
}
div[data-testid="stVerticalBlock"]:has(.infer-header)
    [data-baseweb="select"] > div:first-child {
    background: rgba(255,255,255,0.07) !important;
    border-color: rgba(255,255,255,0.15) !important;
    color: #fff !important;
}
div[data-testid="stVerticalBlock"]:has(.infer-header) input {
    background: rgba(255,255,255,0.07) !important;
    border-color: rgba(255,255,255,0.15) !important;
    color: #fff !important;
}
div[data-testid="stVerticalBlock"]:has(.infer-header)
    [data-testid="stFileUploadDropzone"] {
    background: rgba(255,255,255,0.03) !important;
    border-color: rgba(255,255,255,0.2) !important;
}
div[data-testid="stVerticalBlock"]:has(.infer-header) label,
div[data-testid="stVerticalBlock"]:has(.infer-header) p,
div[data-testid="stVerticalBlock"]:has(.infer-header) span {
    color: rgba(255,255,255,0.75) !important;
}
div[data-testid="stVerticalBlock"]:has(.infer-header) [data-baseweb="select"] svg {
    fill: rgba(255,255,255,0.5) !important;
}
div[data-testid="stVerticalBlock"]:has(.infer-header)
    [data-testid="stHorizontalBlock"] > div {
    background: #0b1628 !important;
}

/* ── MODEL STATS BOX ── */
.model-stats-box {
    background: rgba(255,255,255,0.04);
    border-radius: 8px; padding: 14px 16px;
    display: flex; gap: 24px; flex-wrap: wrap; margin-top: 12px;
}
.sel-stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 20px; font-weight: 700; color: #7eb3ff;
}
.sel-stat-lbl {
    font-size: 11px; color: rgba(255,255,255,0.35);
    text-transform: uppercase; letter-spacing: 0.08em;
}
.best-pill {
    display: inline-block; font-family: 'Syne', sans-serif;
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; background: #185fa5; color: #fff;
    padding: 2px 10px; border-radius: 999px; margin-left: 6px;
}
.dark-label {
    font-family: 'Syne', sans-serif; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: rgba(255,255,255,0.4); margin: 16px 0 6px 0;
}
.guide-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 16px 18px; margin-top: 14px;
}
.guide-item { font-size: 13px; color: rgba(255,255,255,0.4); padding: 3px 0; font-weight: 300; }

/* ── RESULT CARD ── */
.result-card {
    background: #fff; border-radius: 14px;
    overflow: hidden; border: 1px solid #e4e2dc;
}
.result-head { padding: 20px 24px 16px; }
.result-body { padding: 24px; }
.verdict-grade {
    font-family: 'Syne', sans-serif;
    font-size: clamp(24px, 2.5vw, 34px); font-weight: 800;
    line-height: 1; margin-bottom: 4px;
}
.verdict-conf { font-size: 13px; color: #888; margin-bottom: 14px; }
.prob-row { margin-bottom: 12px; }
.prob-header { display: flex; justify-content: space-between; margin-bottom: 5px; }
.prob-name { font-size: 13px; color: #444; }
.prob-pct { font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 700; }
.prob-track { height: 6px; background: #f0ede8; border-radius: 999px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 999px; }
.clin-note {
    border-left: 3px solid transparent; border-radius: 0 8px 8px 0;
    padding: 13px 16px; font-size: 13px; line-height: 1.6;
    font-weight: 300; margin-top: 20px;
}
.disclaimer {
    font-size: 11px; color: #aaa; margin-top: 14px;
    padding-top: 12px; border-top: 1px solid #f0ede8; line-height: 1.5;
}
.info-note {
    background: #e6f1fb; border-left: 3px solid #185fa5;
    border-radius: 0 8px 8px 0; padding: 12px 16px;
    font-size: 13px; color: #0c447c; font-weight: 300;
    margin-top: 16px; line-height: 1.65;
}
.err-banner {
    background: #fcebeb; border: 1px solid #f7c1c1;
    border-radius: 10px; padding: 14px 18px;
    font-size: 13px; color: #791f1f; margin-bottom: 20px;
}

/* ── DOWNLOAD BUTTON ── */
div[data-testid="stDownloadButton"] > button {
    width: 100% !important; background: #0d0d0d !important;
    color: #fff !important; border: none !important;
    border-radius: 8px !important; padding: 14px 24px !important;
    font-family: 'Syne', sans-serif !important; font-size: 13px !important;
    font-weight: 700 !important; letter-spacing: 0.06em !important;
    text-transform: uppercase !important; margin-top: 16px !important;
}
div[data-testid="stDownloadButton"] > button:hover { background: #222 !important; }

/* ── FOOTER ── */
.site-footer {
    background: #0b1628; padding: 32px 6vw;
    display: flex; justify-content: space-between;
    align-items: center; flex-wrap: wrap; gap: 16px;
    border-top: 1px solid rgba(255,255,255,0.06);
}
.footer-l { font-family: 'Syne', sans-serif; font-size: 14px; font-weight: 600; color: #fff; }
.footer-r { font-size: 12px; color: rgba(255,255,255,0.3); font-weight: 300; text-align: right; line-height: 1.7; }

/* ── MOBILE ── */
@media (max-width: 640px) {
    .hero { padding: 48px 5vw 40px 5vw; flex-direction: column; }
    .hero-img-slot { width: 100%; aspect-ratio: 16/9; }
    .infer-header { padding: 36px 5vw 20px 5vw; }
    .site-footer { flex-direction: column; align-items: flex-start; }
    .footer-r { text-align: left; }
}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
GRADE_NAMES  = ["Normal", "Mild", "Moderate", "Severe"]
GRADE_FULL   = ["Normal (Grade 0)", "Mild (Grade 1)", "Moderate (Grade 2)", "Severe (Grade 3)"]
GRADE_COLORS = ["#2d7a3a", "#92600a", "#993c1d", "#a32d2d"]
GRADE_BG     = ["#eaf3de", "#faeeda", "#faece7", "#fcebeb"]
GRADE_BORDER = ["#b8dcc0", "#f0c97b", "#f0c4b0", "#f7c1c1"]
GRADE_TEXT   = ["#1e5228", "#633806", "#712b13", "#791f1f"]
GRADE_NOTES  = [
    "No significant foraminal narrowing detected. Perineural fat is preserved and the nerve root appears uncompressed.",
    "Mild narrowing of the foramen. Perineural fat is partially reduced. Clinical correlation and monitoring recommended.",
    "Moderate stenosis with complete fat obliteration. Nerve contact is present. Specialist evaluation is advised.",
    "Severe stenosis with nerve root deformation. Urgent clinical review — surgical decompression may be indicated.",
]

MODEL_CONFIG = {
    "ConvNeXt V2 Tiny  —  Macro F1: 0.638  ★ Best": {
        "key": "convnext", "f1": 0.638, "acc": 0.730,
        "params": "28M", "best": True,
        "default_path": "models/best_model_v3.pth",
        "display": "ConvNeXt V2 Tiny", "epochs": 50,
    },
    "EfficientNet-B0  —  Macro F1: 0.607": {
        "key": "efficientnet", "f1": 0.607, "acc": 0.723,
        "params": "4M", "best": False,
        "default_path": "models/best_model_v2_4class.pth",
        "display": "EfficientNet-B0", "epochs": 60,
    },
    "ResNet-50  —  Macro F1: 0.575": {
        "key": "resnet", "f1": 0.575, "acc": 0.723,
        "params": "24M", "best": False,
        "default_path": "models/best_model.pth",
        "display": "ResNet-50", "epochs": 30,
    },
    "MaxViT Tiny  —  Macro F1: 0.587": {
        "key": "maxvit", "f1": 0.587, "acc": 0.652,
        "params": "30M", "best": False,
        "default_path": "models/best_model_v5.pth",
        "display": "MaxViT Tiny", "epochs": 50,
    },
    "Swin Transformer V2  —  Macro F1: 0.155  (poor convergence)": {
        "key": "swin", "f1": 0.155, "acc": 0.128,
        "params": "28M", "best": False,
        "default_path": "models/best_model_v4.pth",
        "display": "Swin Transformer V2", "epochs": 60,
    },
}

TRANSFORM = T.Compose([
    T.Resize((224, 224)), T.Grayscale(num_output_channels=3),
    T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── MODEL LOADING ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(key: str, path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if key == "resnet":
        m = resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, 4)
    elif key == "efficientnet":
        m = efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 4)
    elif key == "convnext":
        m = convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 4)
    elif key == "swin":
        m = swin_v2_t(weights=None); m.head = nn.Linear(m.head.in_features, 4)
    elif key == "maxvit":
        m = maxvit_t(weights=None)
        m.classifier[5] = nn.Linear(m.classifier[5].in_features, 4)
    else:
        raise ValueError(f"Unknown model key: {key}")
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device).eval()
    return m, device

# ── GRAD-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, key):
        self.model = model; self.grad = None; self.act = None
        layers = {
            "resnet": model.layer4[-1], "efficientnet": model.features[-1],
            "convnext": model.features[-1], "swin": model.features[-1],
            "maxvit": model.blocks[-1],
        }
        layer = layers[key]
        self._fh = layer.register_forward_hook(lambda m,i,o: setattr(self,"act",o.detach()))
        self._bh = layer.register_full_backward_hook(lambda m,gi,go: setattr(self,"grad",go[0].detach()))

    def generate(self, tensor, cls):
        self.model.zero_grad()
        out = self.model(tensor); out[0, cls].backward()
        g, a = self.grad, self.act
        if g.dim() == 3:
            B, T, C = g.shape; H = W = int(T**0.5)
            if H*W == T:
                g = g.permute(0,2,1).reshape(B,C,H,W)
                a = a.permute(0,2,1).reshape(B,C,H,W)
            else:
                g = g.mean(1,keepdim=True).unsqueeze(-1)
                a = a.mean(1,keepdim=True).unsqueeze(-1)
        cam = F.relu((g.mean([2,3],keepdim=True)*a).sum(1,keepdim=True))
        cam = F.interpolate(cam,(224,224),mode="bilinear",align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam-cam.min())/(cam.max()-cam.min())
        return cam

    def remove(self): self._fh.remove(); self._bh.remove()


def gradcam_overlay(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    img = np.array(pil_img.convert("RGB").resize((224,224)))
    heat = cv2.applyColorMap((cam*255).astype(np.uint8), cv2.COLORMAP_JET)
    blend = cv2.addWeighted(img[:,:,::-1], 1-alpha, heat, alpha, 0)
    return Image.fromarray(blend[:,:,::-1])


def run_inference(model, device, pil_img: Image.Image, key: str):
    tensor = TRANSFORM(pil_img).unsqueeze(0).to(device)
    gcam = GradCAM(model, key)
    with torch.enable_grad():
        out = model(tensor)
    probs = F.softmax(out,dim=1)[0].detach().cpu().numpy()
    pred  = int(probs.argmax())
    cam   = gcam.generate(tensor, pred)
    overlay = gradcam_overlay(pil_img, cam)
    gcam.remove()
    return pred, probs, overlay

# ── PDF REPORT ────────────────────────────────────────────────────────────────
def build_pdf_report(pred, probs, model_name, model_f1, model_acc,
                     patient_id, pil_img, overlay_img):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)

    grade_hex   = ["#2d7a3a","#92600a","#993c1d","#a32d2d"]
    grade_color = colors.HexColor(grade_hex[pred])

    t_sty = ParagraphStyle("T", fontName="Helvetica-Bold", fontSize=18,
        textColor=colors.HexColor("#0b1628"), spaceAfter=4)
    s_sty = ParagraphStyle("S", fontName="Helvetica", fontSize=10,
        textColor=colors.HexColor("#888888"), spaceAfter=16)
    h_sty = ParagraphStyle("H", fontName="Helvetica-Bold", fontSize=12,
        textColor=colors.HexColor("#0b1628"), spaceBefore=14, spaceAfter=6)
    d_sty = ParagraphStyle("D", fontName="Helvetica-Oblique", fontSize=9,
        textColor=colors.HexColor("#999999"), leading=14)

    elems = []
    now   = datetime.datetime.now().strftime("%d %B %Y, %H:%M")

    elems.append(Paragraph("SpineMRI Foraminal Stenosis Grader", t_sty))
    elems.append(Paragraph(f"Clinical AI Report  ·  Generated {now}", s_sty))
    elems.append(HRFlowable(width="100%", thickness=1,
        color=colors.HexColor("#e4e2dc"), spaceAfter=14))

    info = [
        ["Patient ID", patient_id or "Not specified", "Model", model_name],
        ["Date",       now.split(",")[0],              "Macro F1", f"{model_f1:.3f}"],
        ["Dataset",    "282 test samples",             "Accuracy", f"{model_acc:.1%}"],
    ]
    info_t = Table(info, colWidths=["25%","25%","25%","25%"])
    info_t.setStyle(TableStyle([
        ("FONTNAME",(0,0),(-1,-1),"Helvetica"),
        ("FONTSIZE",(0,0),(-1,-1),9),
        ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
        ("FONTNAME",(2,0),(2,-1),"Helvetica-Bold"),
        ("TEXTCOLOR",(0,0),(0,-1),colors.HexColor("#0b1628")),
        ("TEXTCOLOR",(2,0),(2,-1),colors.HexColor("#0b1628")),
        ("TEXTCOLOR",(1,0),(1,-1),colors.HexColor("#444")),
        ("TEXTCOLOR",(3,0),(3,-1),colors.HexColor("#444")),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#f7f6f3"),colors.white]),
        ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#e4e2dc")),
        ("TOPPADDING",(0,0),(-1,-1),6),
        ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LEFTPADDING",(0,0),(-1,-1),8),
    ]))
    elems.append(info_t); elems.append(Spacer(1,16))

    elems.append(Paragraph("Predicted Grade", h_sty))
    v_sty  = ParagraphStyle("v",  fontName="Helvetica-Bold", fontSize=16, leading=20)
    vn_sty = ParagraphStyle("vn", fontName="Helvetica", fontSize=9, leading=14)
    verdict_t = Table([[
        Paragraph(f'<font color="{grade_hex[pred]}" size="16"><b>{GRADE_NAMES[pred]} — Grade {pred}</b></font>', v_sty),
        Paragraph(f'<font size="9" color="#666">Confidence: <b>{float(probs[pred])*100:.1f}%</b><br/>{GRADE_NOTES[pred]}</font>', vn_sty),
    ]], colWidths=["35%","65%"])
    verdict_t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor(["#eaf3de","#faeeda","#faece7","#fcebeb"][pred])),
        ("BOX",(0,0),(-1,-1),1,colors.HexColor(["#b8dcc0","#f0c97b","#f0c4b0","#f7c1c1"][pred])),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),12),
        ("BOTTOMPADDING",(0,0),(-1,-1),12),
        ("LEFTPADDING",(0,0),(-1,-1),14),
    ]))
    elems.append(verdict_t); elems.append(Spacer(1,16))

    elems.append(Paragraph("Class Probability Distribution", h_sty))
    prob_rows = [["Grade","Class","Probability","Confidence"]]
    for i,(gname,prob) in enumerate(zip(GRADE_FULL,probs)):
        pct = float(prob)*100
        bar = "█"*int(pct/5)+"░"*(20-int(pct/5))
        prob_rows.append([f"Grade {i}", gname, f"{pct:.1f}%", bar[:20]])
    prob_t = Table(prob_rows, colWidths=["12%","30%","15%","43%"])
    ts = [
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),9),
        ("FONTNAME",(0,1),(-1,-1),"Helvetica"),
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0b1628")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#e4e2dc")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f7f6f3")]),
        ("TOPPADDING",(0,0),(-1,-1),5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),8),
        ("FONTNAME",(0,pred+1),(-1,pred+1),"Helvetica-Bold"),
        ("TEXTCOLOR",(2,pred+1),(2,pred+1),grade_color),
    ]
    prob_t.setStyle(TableStyle(ts))
    elems.append(prob_t); elems.append(Spacer(1,16))

    elems.append(Paragraph("MRI Images", h_sty))
    def pil_to_rl(img, sz=(6*cm,6*cm)):
        tmp = io.BytesIO()
        img.convert("RGB").save(tmp, format="PNG"); tmp.seek(0)
        from reportlab.platypus import Image as RLImage
        return RLImage(tmp, width=sz[0], height=sz[1])

    img_t = Table([[pil_to_rl(pil_img.resize((224,224))), pil_to_rl(overlay_img)]],
        colWidths=["50%","50%"])
    img_t.setStyle(TableStyle([
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),8),
        ("BOTTOMPADDING",(0,0),(-1,-1),8),
    ]))
    elems.append(img_t)
    cap_t = Table([["Original MRI crop (224×224)","Grad-CAM — warmer = higher activation"]],
        colWidths=["50%","50%"])
    cap_t.setStyle(TableStyle([
        ("FONTNAME",(0,0),(-1,-1),"Helvetica-Oblique"),
        ("FONTSIZE",(0,0),(-1,-1),8),
        ("TEXTCOLOR",(0,0),(-1,-1),colors.HexColor("#999")),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("TOPPADDING",(0,0),(-1,-1),2),
    ]))
    elems.append(cap_t); elems.append(Spacer(1,20))

    elems.append(Paragraph("Grade Scale Reference", h_sty))
    descs = [
        "No foraminal narrowing. Perineural fat intact. No nerve root compression.",
        "Mild narrowing. Partial fat obliteration. Nerve root not yet deformed.",
        "Moderate compression. Complete fat obliteration. Nerve root contact present.",
        "Severe compression. Nerve root deformation. Surgical consult may be needed.",
    ]
    scale_rows = [["Grade","Severity","Clinical Description"]]
    for i,(g,d) in enumerate(zip(GRADE_NAMES,descs)):
        scale_rows.append([f"Grade {i}", g, d])
    scale_t = Table(scale_rows, colWidths=["15%","15%","70%"])
    scale_t.setStyle(TableStyle([
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),9),
        ("FONTNAME",(0,1),(-1,-1),"Helvetica"),
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0b1628")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#e4e2dc")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f7f6f3")]),
        ("TOPPADDING",(0,0),(-1,-1),5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),8),
    ]))
    elems.append(scale_t); elems.append(Spacer(1,24))

    elems.append(HRFlowable(width="100%", thickness=0.5,
        color=colors.HexColor("#e4e2dc"), spaceAfter=10))
    elems.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI research tool for educational and "
        "research purposes only. It does not constitute a medical diagnosis or substitute "
        "for evaluation by a qualified radiologist or physician.", d_sty))

    doc.build(elems)
    buf.seek(0)
    return buf.read()


# =============================================================================
# ── UI ────────────────────────────────────────────────────────────────────────
# =============================================================================

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero">'
    '<div class="hero-text">'
    '<div style="font-family:Syne,sans-serif;font-size:11px;font-weight:600;'
    'letter-spacing:0.18em;text-transform:uppercase;color:#7eb3ff;margin-bottom:18px;">'
    'Deep Learning in Radiology — Research Project</div>'
    '<h1>Lumbar <span>Foraminal</span><br>Stenosis AI</h1>'
    '<p style="font-size:clamp(14px,1.5vw,16px);color:rgba(255,255,255,0.75);'
    'font-weight:300;max-width:480px;margin:0;line-height:1.75;">'
    'Automated severity grading of lumbar nerve compression from sagittal MRI '
    'using five deep learning architectures with Grad-CAM explainability.</p>'
    '<div class="hero-pills">'
    '<span class="pill">5 Models Compared</span>'
    '<span class="pill">Grad-CAM Explainability</span>'
    '<span class="pill">4-Class Grading</span>'
    '<span class="pill">282 Test Samples</span>'
    '<span class="pill">Downloadable Report</span>'
    '</div>'
    '</div>'
    + _hero_img_slot(HERO_SRC) +
    '</div>',
    unsafe_allow_html=True,
)

# ── BACKGROUND ────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="background:#fff;border-bottom:1px solid #e4e2dc;padding:52px 6vw;">'
    '<div style="font-family:Syne,sans-serif;font-size:11px;font-weight:600;'
    'letter-spacing:0.18em;text-transform:uppercase;color:#185fa5;margin-bottom:10px;">Background</div>'
    '<div style="font-family:Syne,sans-serif;font-size:clamp(22px,3vw,32px);'
    'font-weight:700;color:#0d0d0d;margin:0 0 8px 0;line-height:1.15;">Understanding the condition</div>'
    '<div style="font-size:15px;color:#444;font-weight:300;max-width:560px;margin:0 0 36px 0;line-height:1.7;">'
    'Lumbar foraminal stenosis is a narrowing of the nerve exit channels in the lower spine. '
    'AI-assisted grading accelerates radiologist workflows and reduces inter-rater variability.</div>'
    '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px;">'

    '<div style="background:#fff;border:1px solid #e4e2dc;border-radius:12px;padding:20px;display:flex;gap:14px;align-items:flex-start;">'
    '<div style="width:40px;height:40px;border-radius:8px;background:#e6f1fb;display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0;">🦴</div>'
    '<div><div style="font-family:Syne,sans-serif;font-size:14px;font-weight:600;color:#0d0d0d;margin:0 0 5px 0;">The neural foramen</div>'
    '<div style="font-size:13px;color:#444;line-height:1.65;font-weight:300;">A small opening on either side of each vertebra through which spinal nerve roots exit. '
    'In the lumbar spine five disc levels (L1–S1) can be affected bilaterally — ten foramina in total.</div></div></div>'

    '<div style="background:#fff;border:1px solid #e4e2dc;border-radius:12px;padding:20px;display:flex;gap:14px;align-items:flex-start;">'
    '<div style="width:40px;height:40px;border-radius:8px;background:#faece7;display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0;">⚠️</div>'
    '<div><div style="font-family:Syne,sans-serif;font-size:14px;font-weight:600;color:#0d0d0d;margin:0 0 5px 0;">The stenosis problem</div>'
    '<div style="font-size:13px;color:#444;line-height:1.65;font-weight:300;">Disc degeneration, osteophytes, or ligament thickening progressively narrow the foramen, '
    'compressing the nerve root and causing radiating pain, numbness, or weakness (radiculopathy).</div></div></div>'

    '<div style="background:#fff;border:1px solid #e4e2dc;border-radius:12px;padding:20px;display:flex;gap:14px;align-items:flex-start;">'
    '<div style="width:40px;height:40px;border-radius:8px;background:#e1f5ee;display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0;">🔬</div>'
    '<div><div style="font-family:Syne,sans-serif;font-size:14px;font-weight:600;color:#0d0d0d;margin:0 0 5px 0;">MRI grading</div>'
    '<div style="font-size:13px;color:#444;line-height:1.65;font-weight:300;">Sagittal T1/T2 MRI is the gold standard. A radiologist assesses perineural fat '
    'obliteration and nerve contact per level, assigning Grade 0–3 based on severity of compression.</div></div></div>'

    '<div style="background:#fff;border:1px solid #e4e2dc;border-radius:12px;padding:20px;display:flex;gap:14px;align-items:flex-start;">'
    '<div style="width:40px;height:40px;border-radius:8px;background:#faeeda;display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0;">🤖</div>'
    '<div><div style="font-family:Syne,sans-serif;font-size:14px;font-weight:600;color:#0d0d0d;margin:0 0 5px 0;">How this model works</div>'
    '<div style="font-size:13px;color:#444;line-height:1.65;font-weight:300;">Bounding box crops from XML annotations isolate each foramen. A CNN or transformer classifies '
    'each crop. Grad-CAM highlights which region drove the prediction for clinical interpretability.</div></div></div>'

    '</div></div>',
    unsafe_allow_html=True,
)

# ── GRADE SCALE ───────────────────────────────────────────────────────────────
grade_cfgs = [
    (GRADE0_SRC, "rgba(0,0,0,0.08)", "#2d7a3a", "#1e5228", "#2d5e38",
     "#d4edda", "#8ec49e", "Grade 0", "Normal",
     "Full perineural fat preserved. No contact between disc or osteophyte and the nerve root. No intervention required."),
    (GRADE1_SRC, "rgba(0,0,0,0.08)", "#92600a", "#633806", "#7a5010",
     "#fde8b0", "#e8b84b", "Grade 1", "Mild",
     "Partial obliteration of perineural fat. Nerve root not yet deformed. Conservative management typically sufficient."),
    (GRADE2_SRC, "rgba(0,0,0,0.08)", "#993c1d", "#712b13", "#7a3518",
     "#fdd5c0", "#e8926a", "Grade 2", "Moderate",
     "Complete fat obliteration with nerve contact but no deformation. Interventional pain management may be warranted."),
    (GRADE3_SRC, "rgba(0,0,0,0.08)", "#a32d2d", "#791f1f", "#7a2525",
     "#fcc5c5", "#e88080", "Grade 3", "Severe",
     "Nerve root is compressed and deformed. Surgical decompression may be indicated depending on clinical symptoms."),
]

grade_cfgs_dark = [
    # h3_dark, p_dark (strong readable colors)
    ("#1a4a24", "#2a5a30"),  # G0 green card
    ("#4a2e00", "#5a3a00"),  # G1 amber card
    ("#4a1800", "#5a2000"),  # G2 orange card
    ("#4a0f0f", "#5a1515"),  # G3 red card
]
grade_cards_html = ""
for (src, bdr, tag_bg, h3c, pc, bg, card_bdr, tag, name, desc), (h3_dark, p_dark) in zip(grade_cfgs, grade_cfgs_dark):
    grade_cards_html += (
        f'<div style="background:{bg};border:1px solid {card_bdr};border-radius:12px;padding:22px 18px;">'
        + _img_slot(src, f"{tag} MRI example", bdr)
        + f'<span style="font-family:Syne,sans-serif;font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;padding:3px 10px;border-radius:999px;display:inline-block;margin-bottom:14px;background:{tag_bg};color:#ffffff;">{tag}</span>'
        + f'<div style="font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:{h3_dark};margin-bottom:10px;line-height:1.15;">{name}</div>'
        + f'<div style="font-size:13px;color:{p_dark};line-height:1.65;font-weight:400;">{desc}</div>'
        + '</div>'
    )

st.markdown(
    '<div style="background:#f5f4f0;border-bottom:1px solid #e4e2dc;padding:52px 6vw;">'
    '<div style="font-family:Syne,sans-serif;font-size:11px;font-weight:600;'
    'letter-spacing:0.18em;text-transform:uppercase;color:#185fa5;margin-bottom:10px;">Grading System</div>'
    '<div style="font-family:Syne,sans-serif;font-size:clamp(22px,3vw,32px);'
    'font-weight:700;color:#0d0d0d;margin:0 0 8px 0;line-height:1.15;">Four-class severity scale</div>'
    '<div style="font-size:15px;color:#444;font-weight:300;max-width:560px;margin:0 0 36px 0;line-height:1.7;">'
    'Each foramen at each disc level on each side is assigned one of four grades '
    'based on degree of nerve root compression visible on MRI.</div>'
    '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;">'
    + grade_cards_html +
    '</div></div>',
    unsafe_allow_html=True,
)

# ── INFERENCE ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="infer-header">'
    '<div style="font-family:Syne,sans-serif;font-size:11px;font-weight:600;'
    'letter-spacing:0.18em;text-transform:uppercase;color:#7eb3ff;margin-bottom:10px;">Inference</div>'
    '<div style="font-family:Syne,sans-serif;font-size:clamp(22px,3vw,32px);'
    'font-weight:700;color:#fff;margin:0 0 8px 0;line-height:1.15;">Run a prediction</div>'
    '<div style="font-size:15px;color:rgba(255,255,255,0.55);font-weight:300;'
    'max-width:560px;margin:0;line-height:1.7;">'
    'Choose a model, set the weights path, upload an MRI crop and get a grade prediction '
    'with Grad-CAM explainability and a downloadable clinical report.</div>'
    '</div>',
    unsafe_allow_html=True,
)

with st.container():
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<div class="dark-label">Select architecture</div>', unsafe_allow_html=True)
        selected_key = st.selectbox(
            "Architecture", options=list(MODEL_CONFIG.keys()),
            index=0, label_visibility="collapsed",
        )
        cfg = MODEL_CONFIG[selected_key]

        best_html = '<span class="best-pill">Best Model</span>' if cfg["best"] else ""
        st.markdown(
            f'<div class="model-stats-box">'
            f'<div><div class="sel-stat-val">{cfg["f1"]:.3f}</div><div class="sel-stat-lbl">Macro F1</div></div>'
            f'<div><div class="sel-stat-val">{cfg["acc"]:.1%}</div><div class="sel-stat-lbl">Accuracy</div></div>'
            f'<div><div class="sel-stat-val">{cfg["params"]}</div><div class="sel-stat-lbl">Parameters</div></div>'
            f'<div><div class="sel-stat-val">{cfg["epochs"]}</div><div class="sel-stat-lbl">Epochs</div></div>'
            f'</div>{best_html}',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="dark-label">Weights path (.pth file)</div>', unsafe_allow_html=True)
        model_path = st.text_input(
            "Weights path", value=cfg["default_path"],
            placeholder=f"e.g. {cfg['default_path']}",
            label_visibility="collapsed",
        )

        st.markdown('<div class="dark-label">Patient ID (optional)</div>', unsafe_allow_html=True)
        patient_id = st.text_input(
            "Patient ID", placeholder="e.g. 0001",
            label_visibility="collapsed",
        )

        st.markdown(
            '<div class="guide-box">'
            '<div class="guide-item">PNG or JPG format</div>'
            '<div class="guide-item">Sagittal MRI foraminal crop</div>'
            '<div class="guide-item">Grayscale or RGB — both supported</div>'
            '<div class="guide-item">Any resolution — auto-resized to 224×224</div>'
            '<div class="guide-item">5px padding around the foramen recommended</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col_r:
        st.markdown('<div class="dark-label" style="margin-top:0;">Upload MRI crop</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload MRI crop — PNG or JPG",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
        if not uploaded:
            st.markdown(
                '<div style="margin-top:20px;">'
                '<div class="dark-label">Grad-CAM preview</div>'
                '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px;">'
                '<div style="background:rgba(255,255,255,0.04);border:1.5px dashed rgba(255,255,255,0.12);border-radius:10px;aspect-ratio:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;padding:16px;text-align:center;">'
                '<svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg" opacity="0.2"><rect x="3" y="3" width="30" height="30" rx="4" stroke="white" stroke-width="1.5"/><circle cx="12" cy="12" r="3" stroke="white" stroke-width="1.5"/><path d="M3 24 L11 16 L17 22 L23 16 L33 24" stroke="white" stroke-width="1.5" stroke-linejoin="round"/></svg>'
                '<span style="font-size:11px;color:rgba(255,255,255,0.25);font-weight:300;">Original MRI</span>'
                '</div>'
                '<div style="background:rgba(255,255,255,0.04);border:1.5px dashed rgba(255,255,255,0.12);border-radius:10px;aspect-ratio:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;padding:16px;text-align:center;position:relative;overflow:hidden;">'
                '<div style="position:absolute;inset:0;background:radial-gradient(circle at 60% 45%,rgba(255,60,0,0.18) 0%,rgba(255,160,0,0.12) 30%,rgba(0,80,255,0.06) 60%,transparent 75%);border-radius:10px;"></div>'
                '<svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg" opacity="0.22" style="position:relative;z-index:1;"><circle cx="18" cy="18" r="14" stroke="rgba(255,100,0,0.8)" stroke-width="1.5" stroke-dasharray="4 2"/><circle cx="18" cy="18" r="7" stroke="rgba(255,180,0,0.6)" stroke-width="1.5"/><circle cx="18" cy="18" r="2" fill="rgba(255,80,0,0.8)"/></svg>'
                '<span style="font-size:11px;color:rgba(255,255,255,0.25);font-weight:300;position:relative;z-index:1;">Grad-CAM heatmap</span>'
                '</div>'
                '</div>'
                '<div style="margin-top:10px;font-size:12px;color:rgba(255,255,255,0.2);font-weight:300;line-height:1.6;">Upload an MRI crop above to generate the original image and its Grad-CAM activation heatmap.</div>'
                '</div>',
                unsafe_allow_html=True,
            )

# ── RESULTS ───────────────────────────────────────────────────────────────────
if uploaded:
    pil_img = Image.open(uploaded).convert("L")
    path_ok = model_path and Path(model_path).exists()

    if not path_ok:
        st.markdown(
            f'<div style="padding:24px 6vw 0 6vw;">'
            f'<div class="err-banner">Model weights not found at '
            f'<code>{model_path}</code> — please update the path above.</div></div>',
            unsafe_allow_html=True,
        )
        st.image(pil_img, caption="Uploaded image", width=300)
    else:
        with st.spinner(f"Running {cfg['display']} — this may take a moment..."):
            try:
                model, device = load_model(cfg["key"], model_path)
                pred, probs, overlay = run_inference(model, device, pil_img, cfg["key"])
            except Exception as e:
                st.error(f"Inference error: {e}")
                st.stop()

        st.markdown(
            '<div style="background:#f5f4f0;padding:48px 6vw 12px 6vw;">'
            '<div style="font-family:Syne,sans-serif;font-size:11px;font-weight:600;'
            'letter-spacing:0.18em;text-transform:uppercase;color:#185fa5;margin-bottom:10px;">Output</div>'
            '<div style="font-family:Syne,sans-serif;font-size:clamp(22px,3vw,32px);'
            'font-weight:700;color:#0d0d0d;margin:0 0 8px 0;">Prediction results</div>'
            '<div style="font-size:15px;color:#888;font-weight:300;max-width:560px;margin:0;line-height:1.7;">'
            'Explainability heatmap, grade verdict, and confidence breakdown. '
            'Download the full clinical report below.</div></div>',
            unsafe_allow_html=True,
        )

        g      = pred
        col_h  = GRADE_COLORS[g]; bg  = GRADE_BG[g]
        bdr    = GRADE_BORDER[g]; tc  = GRADE_TEXT[g]
        gname  = GRADE_NAMES[g];  conf = float(probs[g])*100

        st.markdown('<div style="background:#f5f4f0;padding:0 6vw 52px 6vw;">', unsafe_allow_html=True)
        res_l, res_r = st.columns([1, 1], gap="large")

        with res_l:
            st.markdown(
                '<div style="font-family:Syne,sans-serif;font-size:12px;font-weight:600;'
                'text-transform:uppercase;letter-spacing:0.1em;color:#888;margin-bottom:12px;">'
                'Image viewer</div>',
                unsafe_allow_html=True,
            )
            tab1, tab2 = st.tabs(["Original MRI", "Grad-CAM heatmap"])
            with tab1:
                st.image(pil_img, use_container_width=True, caption="Input foraminal crop")
            with tab2:
                st.image(overlay, use_container_width=True,
                         caption="Grad-CAM — warmer colours = higher model activation")
            st.markdown(
                '<div class="info-note">Grad-CAM highlights the regions that most influenced '
                'the prediction. Warmer colours (red/yellow) indicate higher activation — '
                'typically the narrowest part of the foramen.</div>',
                unsafe_allow_html=True,
            )

        with res_r:
            st.markdown(
                '<div style="font-family:Syne,sans-serif;font-size:12px;font-weight:600;'
                'text-transform:uppercase;letter-spacing:0.1em;color:#888;margin-bottom:12px;">'
                'Grade verdict</div>',
                unsafe_allow_html=True,
            )

            prob_bars = ""
            for i,(gn,prob) in enumerate(zip(GRADE_FULL,probs)):
                pct = float(prob)*100
                c   = GRADE_COLORS[i]
                bold = "font-weight:600;color:#0d0d0d;" if i==g else "color:#666;"
                mark = " ◀" if i==g else ""
                prob_bars += (
                    f'<div class="prob-row">'
                    f'<div class="prob-header">'
                    f'<span class="prob-name" style="{bold}">{gn}{mark}</span>'
                    f'<span class="prob-pct" style="color:{c};">{pct:.1f}%</span>'
                    f'</div>'
                    f'<div class="prob-track">'
                    f'<div class="prob-fill" style="width:{pct:.1f}%;background:{c};"></div>'
                    f'</div></div>'
                )

            st.markdown(
                f'<div class="result-card">'
                f'<div class="result-head" style="background:{bg};border-bottom:1px solid {bdr};">'
                f'<div style="font-family:Syne,sans-serif;font-size:11px;font-weight:600;'
                f'text-transform:uppercase;letter-spacing:0.1em;color:{tc};opacity:0.65;margin-bottom:6px;">'
                f'{cfg["display"]} — Predicted grade</div>'
                f'<div class="verdict-grade" style="color:{col_h};">{gname} — Grade {g}</div>'
                f'<div class="verdict-conf">Confidence: <strong style="color:{col_h};">{conf:.1f}%</strong></div>'
                f'<div style="height:7px;background:rgba(255,255,255,0.5);border-radius:999px;overflow:hidden;">'
                f'<div style="width:{conf:.1f}%;height:100%;background:{col_h};border-radius:999px;"></div>'
                f'</div></div>'
                f'<div class="result-body">'
                f'<div style="font-family:Syne,sans-serif;font-size:12px;font-weight:600;'
                f'text-transform:uppercase;letter-spacing:0.08em;color:#888;margin-bottom:14px;">Class probabilities</div>'
                + prob_bars +
                f'<div class="clin-note" style="background:{bg};border-left-color:{col_h};color:{tc};">'
                f'{GRADE_NOTES[g]}</div>'
                f'<div class="disclaimer">For research and educational use only. '
                f'Not intended for clinical diagnosis or treatment decisions. '
                f'Always consult a qualified radiologist.</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            with st.spinner("Building PDF..."):
                pdf_bytes = build_pdf_report(
                    pred=pred, probs=probs,
                    model_name=cfg["display"],
                    model_f1=cfg["f1"], model_acc=cfg["acc"],
                    patient_id=patient_id,
                    pil_img=pil_img, overlay_img=overlay,
                )
            fname = (
                f"stenosis_report_{patient_id or 'unknown'}_"
                f"{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            )
            st.download_button(
                label="Download clinical report (.pdf)",
                data=pdf_bytes, file_name=fname,
                mime="application/pdf",
                use_container_width=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="site-footer">'
    '<div><div class="footer-l">SpineMRI Foraminal Stenosis AI</div>'
    '<div style="font-size:12px;color:rgba(255,255,255,0.3);margin-top:4px;font-weight:300;">'
    'Deep Learning Research Project — Radiology × Computer Vision</div></div>'
    '<div class="footer-r">PyTorch · torchvision · Streamlit · ReportLab<br>'
    'ConvNeXt V2 · EfficientNet · ResNet · MaxViT · Swin</div>'
    '</div>',
    unsafe_allow_html=True,
)