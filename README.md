# Deep Learning for Flower Image Similarity Assessment

This repository contains a comprehensive implementation of deep learning approaches for flower image similarity assessment using the 102 Flower Category Dataset. The project explores different embedding-based architectures and their robustness to adversarial attacks.

## 📚 Project Overview

This educational project implements and compares three different deep learning approaches for image similarity assessment:

1. **Center Loss Classification** - Uses classification with center loss to learn discriminative features
2. **Siamese Networks** - Learns similarity through paired comparisons
3. **Triplet Loss Networks** - Optimizes embedding space using triplet margin loss

Each approach is evaluated for both standard performance and adversarial robustness, providing insights into different architectural choices for similarity learning tasks.

## 🗂️ Repository Structure

### Jupyter Notebooks

| Notebook | Description |
|----------|-------------|
| `R1_Data_Exploration_and_Analysis.ipynb` | Dataset exploration, visualization, and statistical analysis |
| `R2_Model_Selection_and_Backbone_Analysis.ipynb` | Backbone architecture selection and comparison |
| `R3_Center_Loss.ipynb` | Center loss implementation for classification-based similarity |
| `R4_Siamese_Network.ipynb` | Siamese network implementation and training |
| `R5_Triplet_Loss.ipynb` | Triplet loss network implementation and optimization |
| `R6_Adversarial_Robustness.ipynb` | Adversarial training and robustness evaluation |
| `R7_Explainability_Analysis.ipynb` | Model interpretability using Concept Activation Maps |

### Model Files

| Model Type | Standard Model | Robust Model |
|------------|----------------|--------------|
| **Center Loss** | `best_model_center_loss.pth` | `robust_model_center_loss.pth` |
| **Siamese Network** | `best_model_siamese_network.pth` | `robust_model_siamese_network.pth` |
| **Triplet Loss** | `best_model_triplet_loss.pth` | `robust_model_triplet_loss.pth` |

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision timm matplotlib seaborn scikit-learn pandas grad-cam opencv-python
```

### Data Access

All project files and datasets are available at:
📁 **[Google Drive Repository](https://drive.google.com/drive/folders/1Z4TOIbie2iCcJeSU3JRPbUQEjx2ZVohM)**

### Running the Notebooks

1. Start with `R1_Data_Exploration_and_Analysis.ipynb` for dataset understanding
2. Follow the numerical sequence (R1 → R2 → ... → R8) for complete workflow
3. Each notebook is self-contained with installation commands and data downloads

