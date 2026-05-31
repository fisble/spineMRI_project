"""
dataset.py  —  Parses XML annotations, crops bounding boxes from PNGs,
returns (image_tensor, grade, disc_level, side, patient_id, slice_name)
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# Grade label mapping: name prefix -> int
GRADE_MAP = {"0": 0, "1": 1, "2": 2, "3": 3}

# Disc level -> int  (for per-level reporting)
LEVEL_MAP = {"L1-L2": 0, "L2-L3": 1, "L3-L4": 2, "L4-L5": 3, "L5-S1": 4}
LEVEL_NAMES = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
GRADE_NAMES = ["Normal (Grade 0)", "Mild (Grade 1)",
               "Moderate (Grade 2)", "Severe (Grade 3)"]


def parse_xml(xml_path: Path) -> list[dict]:
    """
    Returns a list of annotation dicts from one XML file.
    Each dict: {patient_id, slice_name, side, grade, level, bbox: (xmin,ymin,xmax,ymax)}
    """
    records = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return records

    slice_name = root.findtext("filename", "").replace(".png", "")
    patient_id = xml_path.parent.name          # e.g. "0001"

    for obj in root.findall("object"):
        name = obj.findtext("name", "")        # e.g. "RFS2"
        level = obj.findtext("level", "")      # e.g. "L4-L5"
        if len(name) < 4:
            continue

        side = name[:3]                        # "RFS" or "LFS"
        grade_char = name[3]                   # "0","1","2","3"
        if grade_char not in GRADE_MAP or level not in LEVEL_MAP:
            continue

        bb = obj.find("bndbox")
        if bb is None:
            continue

        try:
            bbox = (
                int(bb.findtext("xmin")),
                int(bb.findtext("ymin")),
                int(bb.findtext("xmax")),
                int(bb.findtext("ymax")),
            )
        except (TypeError, ValueError):
            continue

        records.append({
            "patient_id": patient_id,
            "slice_name":  slice_name,
            "side":        side,
            "grade":       GRADE_MAP[grade_char],
            "level":       level,
            "level_idx":   LEVEL_MAP[level],
            "bbox":        bbox,
        })
    return records


def build_records(foramina_dir: str) -> list[dict]:
    """Walk Foramina_Detection/, parse all XMLs, return flat record list."""
    base = Path(foramina_dir)
    all_records = []
    for xml_path in sorted(base.rglob("*.xml")):
        for rec in parse_xml(xml_path):
            # Resolve PNG path
            png_path = xml_path.parent / f"{rec['slice_name']}.png"
            if png_path.exists():
                rec["png_path"] = str(png_path)
                all_records.append(rec)
    return all_records


# ---------- Transforms ----------

def get_transform(split: str, crop_size: int = 64):
    """
    split: 'train' | 'val' | 'test'
    Crops are small (foramina ~40-60px), so we resize to 64 then to 224.
    """
    if split == "train":
        return T.Compose([
            T.Resize((crop_size, crop_size)),
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(5),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.Grayscale(num_output_channels=3),   # PNG is 1-ch; ResNet needs 3
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])


class ForaminaDataset(Dataset):
    def __init__(self, records: list[dict], split: str = "train"):
        self.records = records
        self.transform = get_transform(split)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["png_path"]).convert("L")   # grayscale
        xmin, ymin, xmax, ymax = rec["bbox"]

        # Add 5px padding around bounding box, clamp to image size
        w, h = img.size
        pad = 5
        crop = img.crop((
            max(0, xmin - pad),
            max(0, ymin - pad),
            min(w, xmax + pad),
            min(h, ymax + pad),
        ))

        tensor = self.transform(crop)
        return {
            "image":      tensor,
            "grade":      torch.tensor(rec["grade"], dtype=torch.long),
            "level_idx":  torch.tensor(rec["level_idx"], dtype=torch.long),
            "side":       rec["side"],
            "patient_id": rec["patient_id"],
            "slice_name": rec["slice_name"],
            "level":      rec["level"],
        }
