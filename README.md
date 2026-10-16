# MedVision AI — Explainable Deep Learning-Based Skin Disease Classification System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![Model](https://img.shields.io/badge/Backbone-EfficientNetB0-brightgreen.svg)]()
[![Explainability](https://img.shields.io/badge/XAI-Grad--CAM-red.svg)]()

MedVision AI is an explainable deep learning pipeline for multi-class skin disease classification. Built with **EfficientNetB0** transfer learning and **Grad-CAM (Gradient-weighted Class Activation Mapping)**, it provides predictions and interpretable spatial heatmaps for clinical and dermoscopic skin disease screening.

---

## Key Features

- **8-Class Skin Classifier:** Diagnoses Acne, Actinic Keratosis, Basal Cell Carcinoma, Dermatofibroma, Melanocytic Nevus, Melanoma, Seborrheic Keratosis, and Squamous Cell Carcinoma.
- **Explainable AI (XAI):** Generates Grad-CAM visual explanations showing the exact image regions influencing predictions.
- **High Performance:** 76.32% test accuracy, 0.9608 multi-class ROC-AUC, 0.6851 macro F1, and 0.7567 weighted F1 across 2,280 verified test samples.
- **Edge-Ready Optimization:** Includes TFLite (FP32, FP16, INT8) and ONNX quantization pipelines.
- **Production Architecture:** Modular pipeline with dedicated modules for configuration, inference, evaluation, explainability, and optimization.

---

## Repository Structure

```text
MedvisionAi/
├── ML-Model/               # Core Machine Learning Pipeline
│   ├── config.py           # Central Python configuration
│   ├── config/config.yaml  # Extended research settings
│   ├── models/             # Trained EfficientNetB0 model & class mappings
│   ├── inference/          # Single-image prediction engine
│   ├── explainability/     # Grad-CAM heatmap & overlay generation
│   ├── evaluation/         # Comprehensive metric reports & error analysis
│   ├── optimization/       # TFLite and ONNX quantization suite
│   ├── utils/              # Data loaders, callbacks, and visualization
│   ├── experiments/        # Backbone comparisons and hyperparameter tuning
│   └── outputs/            # Evaluation benchmarks and visualization artifacts
├── Backend/                # Backend API service directory
│   └── README.md           # Backend service documentation
├── Frontend/               # Frontend user interface directory
│   └── README.md           # Frontend UI documentation
├── .gitignore              # Git ignore rules for datasets, environments, and logs
└── README.md               # Project overview and documentation
```

---

## Quickstart

For detailed instructions on running inference, generating Grad-CAM explanations, and running evaluation benchmarks, see [`ML-Model/README.md`](ML-Model/README.md).

```bash
# Run skin disease inference on an image
python ML-Model/inference/predict.py --image path/to/skin_image.jpg

# Generate Grad-CAM explainability heatmap
python ML-Model/explainability/explainability.py --image path/to/skin_image.jpg
```

---

## Disclaimer

This project is developed for educational research and preliminary screening. It is **not** a certified medical diagnostic device and should not replace evaluation by a qualified medical professional.
