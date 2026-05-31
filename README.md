# Spine MRI Foraminal Stenosis Detection

Automated detection and grading of lumbar foraminal stenosis from sagittal MRI using deep learning.

## Models Compared
| Model | Architecture | Macro F1 | Accuracy |
|---|---|---|---|
| M1 | ResNet-50 | 0.5746 | 0.7234 |
| M2 | EfficientNet-B0 | 0.61 | 0.72 |
| M3 | ConvNeXt V2 Tiny ★ | **0.6377** | **0.7305** |
| M4 | Swin Transformer V2 | 0.1552 | 0.1277 |
| M5 | MaxViT Tiny | 0.5865 | 0.6525 |

## Setup
pip install -r requirements.txt

## Train
python train_v3.py --data "path/to/Foramina_Detection"

## Evaluate  
python evaluate_all_final.py --data "path/to/Foramina_Detection"