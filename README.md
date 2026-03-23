# 👗 Fashion-MNIST Classification Using Machine Learning
### IT461 — Practical Machine Learning | Project Checkpoint

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Kaggle](https://img.shields.io/badge/Dataset-Fashion--MNIST-20BEFF?style=flat-square&logo=kaggle&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=flat-square)

---

## 📌 Overview

This project applies **Support Vector Machine (SVM)** classifiers to the **Fashion MNIST** dataset to classify grayscale images of clothing items into **10 categories**. The goal is to explore how well classical machine learning methods perform on image classification tasks, with systematic hyperparameter tuning and performance evaluation.

---

## 🎯 Task

| | |
|---|---|
| **Problem Type** | Multi-class Image Classification |
| **Input** | 28×28 grayscale image (784 features) |
| **Output** | One of 10 fashion category labels |
| **Dataset** | Fashion MNIST by Zalando Research |
| **Samples** | 70,000 (60,000 train / 10,000 test) |

---

## 🏷️ Class Labels

| Label | Category     | Label | Category    |
|-------|-------------|-------|-------------|
| 0     | T-shirt/Top | 5     | Sandal      |
| 1     | Trouser     | 6     | Shirt       |
| 2     | Pullover    | 7     | Sneaker     |
| 3     | Dress       | 8     | Bag         |
| 4     | Coat        | 9     | Ankle Boot  |

---

## 📁 Project Structure

```
fashion-mnist-svm/
│
├── notebook.ipynb          # Main Jupyter notebook (all 14 cells)
├── README.md               # Project documentation
│
└── figures/                # Output plots (auto-generated)
    ├── sample_images.png
    ├── samples_per_class.png
    ├── confusion_matrix_rbf.png
    ├── confusion_matrix_poly.png
    ├── predictions.png
    └── results_comparison.png
```

---

## 🔧 Methods

### Models Explored
- **SVM with RBF Kernel** — baseline and GridSearchCV-tuned
- **SVM with Polynomial Kernel** — baseline and GridSearchCV-tuned

### Pipeline
```
Raw Images (784 features)
        ↓
  Normalization (÷ 255)
        ↓
  PCA (784 → 100 components)
        ↓
  SVM Classifier (RBF / Poly)
        ↓
  Predicted Class Label
```

### Hyperparameter Tuning
GridSearchCV with **5-fold cross-validation** was used to find optimal parameters:

| Kernel     | Parameters Tuned              |
|------------|-------------------------------|
| RBF        | `C` ∈ {1, 5, 10}, `gamma` ∈ {0.0005, 0.001, 0.005} |
| Polynomial | `C` ∈ {0.1, 1, 5, 10}, `degree` ∈ {2, 3, 4}, `gamma` ∈ {scale, auto} |

---

## 📊 Results (Preliminary)

> ⚠️ Results will be updated after full experiments are completed.

| Model                    | Accuracy | F1 Score (Weighted) |
|--------------------------|----------|----------------------|
| SVM — RBF (baseline)     | TBD      | TBD                  |
| SVM — Poly (baseline)    | TBD      | TBD                  |
| SVM — Best RBF (tuned)   | TBD      | TBD                  |
| SVM — Best Poly (tuned)  | TBD      | TBD                  |

---

## 📦 Dependencies

| Package        | Purpose                         |
|----------------|---------------------------------|
| `numpy`        | Numerical operations            |
| `pandas`       | Data loading and manipulation   |
| `matplotlib`   | Plotting and visualization      |
| `seaborn`      | Heatmaps and styled plots       |
| `scikit-learn` | SVM, PCA, GridSearchCV, metrics |
| `kagglehub`    | Downloading the dataset         |

---

## 📓 Notebook Structure

| Cell | Description |
|------|-------------|
| 1  | Imports & Dataset Download |
| 2  | Load & Explore Data |
| 3  | Prepare Features & Labels + Normalization |
| 4  | Visualize Sample Images |
| 5  | Create Training Subset (10K samples) |
| 6  | Baseline SVM — RBF & Polynomial |
| 7  | GridSearchCV — Best RBF Kernel |
| 8  | GridSearchCV — Best Polynomial Kernel |
| 9  | Evaluate Best Models + Classification Report |
| 10 | Confusion Matrix Heatmap |
| 11 | Visualize Predictions (Correct / Incorrect) |
| 12 | One Sample Per Class (Dataset Overview) |
| 13 | Results Comparison Table — All 4 Models |
| 14 | Reusable Helper Functions (`evaluate_model`, `build_svm_pipeline`) |

---

## 👥 Team

| Name | Student ID |
|------|------------|
| Norah Aljayan | 444200832 |
| [Teammate Name] | [ID] |

**Course:** IT461 — Practical Machine Learning  
**Institution:** King Saud University  
**Semester:** Spring 2026

---

## 📄 Dataset Reference

> Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. [arXiv:1708.07747](https://arxiv.org/abs/1708.07747)

Dataset available at: [Kaggle — Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
