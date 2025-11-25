
# Deep Learning for Flood Damage Detection Using Satellite and Real-World Imagery

## Project Overview
This project implements machine learning and deep learning models to detect and classify flood conditions from satellite or sensor imagery.

## Features
- Data preprocessing and augmentation
- Multiple ML/DL model implementations
- Model evaluation and comparison
- Visualization of results

## Project Structure
```
MLDL_FloodDetection/
├── dataset/
├── update1/
    ├── code.ipynb
    ├── loss_curve.png
├── update2/
│   ├── code.ipynb
├── requirements.txt
└── readme.md
```


## Dataset
- **Dataset:** FloodNet
- **Images:** RGB aerial imagery (`.jpg`)
- **Masks:** RGB segmentation masks (`.png`)
- **Classes:** 6 semantic classes encoded via RGB values

| Class ID | RGB |
|---------|-------------|
| 0 | (61, 61, 245) |
| 1 | (250, 50, 83) |
| 2 | (255, 96, 55) |
| 3 | (51, 221, 255) |
| 4 | (102, 255, 102) |
| 5 | (92, 179, 162) |

## Deep Learning Model
| Model | Description |
|-------|-------------|
| Majority-Class Baseline | Always predicts most-frequent class |
| **UNet (Final Model)** | Encoder-decoder CNN with skip connections |

###  Hyperparameters
| Parameter | Value |
|----------|-------|
| Epochs | 10 |
| Batch Size | 4 |
| Learning Rate | 1e-4 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |

##  Results Summary
| Metric | Best Value |
|--------|------------|
| **Pixel Accuracy** | **≈ 92–93%** |
| **mIoU** | **≈ 0.73** |
| **Macro Precision** | **≈ 0.89** |
| **Macro Recall** | **≈ 0.90** |

### Baseline vs UNet Comparison
| Model | Pixel Acc | mIoU |
|--------|----------|------|
| Majority-Class Baseline | 20–30% | 0.10–0.18 |
| **UNet (ours)** | **87–93%** | **0.61–0.73** |

##  Visual Results
Each set includes:
| Image | Ground Truth | Predicted Mask |
Stored inside: `results/predictions/`

## Installation
```bash
pip install -r requirements.txt
```

##  Final Deliverables (In Progress)
| Deliverable | Status |
|-------------|--------|
| Code Checkpoint | ✔ |
| Intermediate Update 2 | ✔ |
| Final Research Paper | pending |
| Final Presentation |  pending|

##  Contributors
- **Visesh Bentula**
- **Teja Reddy Mandadi**
- **Umapathi Konduri**

