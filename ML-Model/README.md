# MedVision AI — Skin Disease Classification Model

Explainable deep learning pipeline for multi-class skin disease classification using EfficientNetB0 and Grad-CAM.

---

## 1. Project Overview

**MedVision AI** is an AI-assisted skin lesion screening and research system. It leverages transfer learning with **EfficientNetB0** to classify dermoscopic and clinical images into 8 distinct skin disease categories. It incorporates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize the exact spatial features influencing each classification decision.

---

## 2. Supported Skin Diseases (8 Classes)

The classifier operates on 8 clinically significant skin disease classes:

1. **Acne**
2. **Actinic Keratosis (AK)**
3. **Basal Cell Carcinoma (BCC)**
4. **Dermatofibroma (DF)**
5. **Melanocytic Nevus (Nevus)**
6. **Melanoma (MEL)**
7. **Seborrheic Keratosis (SK)**
8. **Squamous Cell Carcinoma (SCC)**

The class-to-index mapping is defined in `models/skin_class_names.json`.

---

## 3. Architecture & Performance

- **Backbone Architecture:** EfficientNetB0 (ImageNet pre-trained weights)
- **Input Dimensions:** 224 × 224 × 3 RGB
- **Classification Head:** GlobalAveragePooling2D → Dropout(0.3) → Dense(256, ReLU) → Dense(8, Softmax)
- **Test Set Accuracy:** **76.32%** (Evaluated on 2,280 test images)
- **Macro F1-Score:** **0.6851** | **Weighted F1-Score:** **0.7567**
- **Multi-Class ROC-AUC:** **0.9608**
- **Explainability:** Grad-CAM with 60/40 Jet colormap heatmap overlay

---

## 4. Directory Structure

```text
ML-Model/
├── config.py              # Central runtime and path configuration
├── config/
│   └── config.yaml        # Extended pipeline & hyperparameter settings
├── models/
│   ├── skin_model.keras   # Trained EfficientNetB0 model weights
│   └── skin_class_names.json # Target 8-class label mapping
├── inference/
│   └── predict.py         # Single-image prediction engine
├── explainability/
│   ├── explainability.py  # Grad-CAM heatmap & overlay generator
│   └── run_gradcam_evaluation.py # Multi-sample Grad-CAM generator
├── evaluation/
│   ├── evaluate.py        # Test split evaluator & classification report generator
│   └── error_analysis.py  # Diagnostic failure case & confidence analyzer
├── optimization/
│   └── optimize_model.py  # TFLite (FP32/FP16/INT8) and ONNX quantization suite
├── utils/                 # Shared data loaders, metrics, callbacks, and visualization
├── experiments/           # Transfer learning comparison & tuning experiments
├── tracking/              # MLflow run tracking
├── scripts/               # Dataset splitters & maintenance scripts
└── outputs/               # Evaluation reports, confusion matrices, and Grad-CAM samples
```

---

## 5. Quickstart & Usage

### Setup Environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Run Single-Image Inference
```bash
python inference/predict.py --image path/to/skin_image.jpg
```

### Generate Grad-CAM Visual Explanation
```bash
python explainability/explainability.py --image path/to/skin_image.jpg
```

### Run Full Test Evaluation
```bash
python evaluation/evaluate.py --dataset skin
```

---

## 6. Disclaimer

This system is designed for educational research and AI-assisted preliminary screening. It is **not** a substitute for professional medical diagnosis or clinical biopsy.
