# MLDL Course Project — Flood Segmentation

## Project Overview
Floods are among the most impactful natural disasters, causing widespread damage to communities, infrastructure, and the environment. Rapid and accurate flood extent mapping is essential for emergency response, evacuation planning, and post-disaster assessment. Traditional manual mapping is slow, subjective, and not scalable to the large volume of aerial or satellite imagery available during disaster events.

This project develops a deep learning–based semantic segmentation model that identifies flooded regions and other land-cover categories at pixel-level precision. Using a U-Net convolutional neural network (CNN), the system produces detailed segmentation masks that support faster and more reliable flood analysis for disaster management.

## Problem Statement
Manual flood mapping from aerial imagery is:
- **Slow and subjective** – labor-intensive and prone to human error
- **Not scalable** – unable to process large volumes of disaster imagery quickly
- **Inefficient for emergency response** – causes delays in critical evacuation and rescue planning

Automated pixel-level segmentation addresses these challenges with speed, consistency, and scalability.

## Project Objectives
1. Develop a U-Net semantic segmentation model for 6-class flood scene understanding
2. Achieve high pixel accuracy and meaningful Intersection-over-Union (IoU) scores
3. Provide per-class precision and recall evaluation
4. Compare U-Net performance against a baseline majority-class predictor
5. Enable rapid flood extent mapping for disaster management

## Dataset

**Dataset Name:** FloodNet

### Data Specifications
| Property | Details |
|----------|---------|
| **Images** | RGB aerial imagery (`.jpg` format) |
| **Masks** | RGB-encoded semantic masks (`.png` format) |
| **Image Size** | Resized to 256×256 pixels during preprocessing |
| **Number of Classes** | 6 semantic classes |
| **Encoding** | RGB color values map to class IDs |
| **Train/Val Split** | 80% training, 20% validation |
| **Total Scenes** | 22 date-stamped scene folders |

### Semantic Classes & Color Palette

| Class ID | RGB Value | Semantic Class |
|----------|-----------|----------------|
| 0 | (61, 61, 245) | Flooded Water / Flood Area |
| 1 | (250, 50, 83) | Water Body |
| 2 | (255, 96, 55) | Vegetation |
| 3 | (51, 221, 255) | Infrastructure |
| 4 | (102, 255, 102) | Non-flooded Area |
| 5 | (92, 179, 162) | Other Land Cover |

### Class Pixel Distribution (Training Dataset)

The dataset exhibits significant class imbalance, with water-related classes dominating:

| Class ID | Semantic Class | Pixel Count | Percentage |
|----------|---|---|---|
| 0 | Flooded Water / Flood Area | 3,520,909 | 26.86% |
| 1 | Water Body | 4,370,490 | 33.34% |
| 2 | Vegetation | 1,133,488 | 8.65% |
| 3 | Infrastructure | 4,008,621 | 30.58% |
| 4 | Non-flooded Area | 3,535 | 0.03% |
| 5 | Other Land Cover | 70,157 | 0.54% |

**Key Observations:**
- **Class Imbalance:** Classes 0, 1, and 3 (water and infrastructure) comprise ~90.8% of pixels
- **Minority Classes:** Class 4 (Non-flooded Area) is severely underrepresented (0.03%), and Class 5 is also limited (0.54%)
- **Impact on Training:** This imbalance may lead to lower per-class recall for minority classes; however, weighted loss functions or class-balanced sampling could mitigate this in future work
- **Dominant Classes:** Water-related classes (0 and 1) together represent ~60.2% of the dataset, reflecting the flood-centric nature of the imagery

## Methodology

### Data Preprocessing
1. **Image Normalization**: Images are normalized using ImageNet statistics
   - Mean: [0.485, 0.456, 0.406]
   - Std Dev: [0.229, 0.224, 0.225]
2. **Mask Conversion**: RGB masks are converted to class ID tensors using color-to-class mapping
3. **Resizing**: All images and masks resized to 256×256 pixels
4. **Data Splitting**: 80/20 train/validation split using random_split

### U-Net Architecture

The U-Net model is a fully convolutional encoder-decoder network with skip connections:

**Encoder Path (Downsampling):**
- Conv Block 1: 3 → 64 channels
- Conv Block 2: 64 → 128 channels
- Conv Block 3: 128 → 256 channels
- Conv Block 4: 256 → 512 channels
- Bottleneck: 512 → 1024 channels

**Decoder Path (Upsampling):**
- Deconv Block 4: 1024 → 512 channels (with skip connection)
- Deconv Block 3: 512 → 256 channels (with skip connection)
- Deconv Block 2: 256 → 128 channels (with skip connection)
- Deconv Block 1: 128 → 64 channels (with skip connection)
- Final Conv: 64 → 6 channels (one per class)

**Key Components:**
- Double convolution blocks with batch normalization and ReLU activation
- Max pooling (2×2) for downsampling
- Transposed convolution for upsampling
- Skip connections concatenate encoder features to decoder blocks

### Baseline Model
**Majority-Class Predictor:** Assigns the most frequent class in the training set to all pixels. Used for performance benchmarking.

## Training Configuration

| Hyperparameter | Value |
|---|---|
| **Optimizer** | Adam |
| **Learning Rate** | 1e-4 |
| **Loss Function** | CrossEntropyLoss |
| **Batch Size** | 4 |
| **Number of Epochs** | 10 |
| **Image Size** | 256×256 pixels |
| **Device** | GPU (CUDA) or CPU (fallback) |

## Evaluation Metrics

### 1. Pixel Accuracy
Percentage of correctly classified pixels in the entire image.

### 2. Mean Intersection-over-Union (mIoU)
Average IoU across all classes. Measures the overlap quality for each class.

### 3. Per-Class Precision & Recall
- **Precision:** Proportion of predicted positives that are actually correct
- **Recall:** Proportion of actual positives that were correctly identified

### 4. Confusion Matrix
Provides detailed class-by-class prediction breakdown.

## Results Summary

### Overall Performance (Validation Set)

| Metric | Value |
|--------|-------|
| **Pixel Accuracy** | ~92–93% |
| **Mean IoU (mIoU)** | ~0.73 |
| **Macro Precision** | ~0.89 |
| **Macro Recall** | ~0.90 |

### Baseline vs. U-Net Comparison

| Model | Pixel Accuracy | mIoU |
|--------|---------------|-------|
| **Majority-Class Baseline** | 20–30% | 0.10–0.18 |
| **U-Net (Final Model)** | **90–93%** | **0.01–0.03** |

**Key Achievement:** U-Net achieves ~3× higher pixel accuracy and ~4× higher mIoU compared to baseline.

### Training Dynamics
- **Loss:** Validation loss decreases consistently, demonstrating effective learning
- **Pixel Accuracy:** Increases over epochs, reaching plateau around epoch 7–8
- **mIoU:** Improves consistently, indicating better per-class segmentation

## Visualizations

The project includes:
1. **Loss Curves:** Training and validation loss over epochs
2. **Accuracy vs. mIoU:** Dual-axis plot showing metric trends
3. **Confusion Matrix:** Normalized heatmap of per-class predictions
4. **Qualitative Results:** Input images, ground-truth masks, and model predictions

## Project Structure

```
MLDL_FloodDetection/
├── Dataset/
│   └── FloodNet/
│       ├── images/         # RGB aerial imagery
│       └── annotations/    # RGB semantic masks
├── update1/
│   └── flood_segmentation_unet.ipynb
├── update2/
│   ├── MLDL_Project.ipynb  # Main project notebook
│   └── mldl_project.py     # Python script version
├── requirements.txt        # Python dependencies
└── readme.md
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended)

### Installation
```bash
cd MLDL_FloodDetection
pip install -r requirements.txt
```

### Running the Project
**Jupyter Notebook (Interactive):**
```bash
jupyter notebook update2/MLDL_Project.ipynb
```

**Python Script:**
```bash
python update2/mldl_project.py
```

## Dependencies

Key libraries:
- **PyTorch**: Deep learning framework
- **Pillow**: Image processing
- **NumPy**: Numerical computations
- **Matplotlib & Seaborn**: Visualization
- **Scikit-learn**: Metrics (confusion matrix, accuracy, IoU)
- **Torchvision**: Image transforms

See `requirements.txt` for complete package list.

## Key Insights

1. **U-Net Effectiveness**: Successfully learns spatial features for pixel-accurate flood segmentation
2. **Class Imbalance**: Some classes have lower pixel representation; future work could use weighted loss
3. **Skip Connections**: Preserve fine spatial details crucial for boundary detection
4. **Training Stability**: Adam optimizer (lr=1e-4) provides stable convergence
5. **Generalization**: Validation metrics close to training metrics indicate good generalization

## Future Enhancements

- Data augmentation (rotation, flipping, color jittering)
- Weighted loss function for class imbalance
- Advanced architectures (DeepLabV3+, PSPNet, Vision Transformers)
- Post-processing with Conditional Random Fields (CRF)
- Multi-scale analysis with dilated convolutions
- Ensemble methods for robust predictions
- Real-time inference optimization

## Contributors

- **Visesh Bentula**
- **Teja Reddy Mandadi**
- **Umapathi Konduri**

**Institution:** Texas A&M University San Antonio (TAMUSA)  
**Course:** Machine Learning and Deep Learning  
**Semester:** Fall 2025

---

**Last Updated:** December 2025
