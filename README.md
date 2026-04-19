# 👗 Fashion-MNIST Classification Using Machine Learning
### IT461 — Practical Machine Learning | Project Checkpoint

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=flat-square)

---

## 📌 Overview

This project applies multiple machine learning methods to the **Fashion MNIST** dataset to classify grayscale images of clothing items into **10 categories**. We explore and compare classical ML methods (SVM, KNN) and a deep learning approach (CNN), with systematic hyperparameter tuning and performance evaluation across all methods.

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
fashion-mnist-classification/
│
├── notebook.ipynb          # Main Jupyter notebook (all sections)
├── README.md               # Project documentation
│
└── figures/                # Output plots (auto-generated)
    ├── sample_images.png
    ├── cnn_training_curves.png
    ├── confusion_matrix_cnn.png
    ├── confusion_matrix_knn.png
    ├── confusion_matrix_svm_rbf.png
    ├── confusion_matrix_svm_poly.png
    ├── predictions.png
    └── results_comparison.png
```

---

## 🔧 Methods

### Models Explored

| Method | Type | Tuning |
|--------|------|--------|
| **CNN** | Deep Learning | Learning rate search (0.01, 0.001) |
| **KNN** | Classical ML | GridSearchCV — k, distance metric |
| **SVM — RBF Kernel** | Classical ML | GridSearchCV — C, gamma |
| **SVM — Polynomial Kernel** | Classical ML | GridSearchCV — C, degree, gamma |

### Pipeline (CNN)
```
Raw Images (28×28)
        ↓
  Normalization (ToTensor)
        ↓
  Conv2d → ReLU → MaxPool (×2)
        ↓
  Fully Connected + Dropout
        ↓
  Predicted Class Label
```

### Pipeline (KNN & SVM)
```
Raw Images (784 features)
        ↓
  Normalization (÷ 255)
        ↓
  PCA (784 → 100 components, whitened)
        ↓
  KNN / SVM Classifier
        ↓
  Predicted Class Label
```

### Hyperparameter Tuning
GridSearchCV with **5-fold cross-validation** for all sklearn methods. Manual learning rate search for CNN.

| Method | Parameters Tuned |
|--------|-----------------|
| CNN | `lr` ∈ {0.01, 0.001}, epochs = 4 |
| KNN | `n_neighbors` ∈ {3, 5, 7, 9}, `metric` ∈ {euclidean, manhattan} |
| SVM RBF | `C` ∈ {1, 5, 10}, `gamma` ∈ {0.0005, 0.001, 0.005} |
| SVM Polynomial | `C` ∈ {0.1, 1, 5, 10}, `degree` ∈ {2, 3, 4}, `gamma` ∈ {scale, auto} |

---

## 📊 Results (Preliminary)

> ⚠️ Results will be updated after full experiments are completed.

| Model | Accuracy | F1 Score (Weighted) |
|-------|----------|----------------------|
| CNN (best LR) | TBD | TBD |
| KNN Baseline | TBD | TBD |
| KNN (GridSearchCV) | TBD | TBD |
| SVM RBF Baseline | TBD | TBD |
| SVM Poly Baseline | TBD | TBD |
| SVM RBF (GridSearchCV) | TBD | TBD |
| SVM Poly (GridSearchCV) | TBD | TBD |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | CNN model definition and training |
| `torchvision` | Fashion MNIST dataset loading |
| `numpy` | Numerical operations |
| `pandas` | Results tables and display |
| `matplotlib` | Plotting and visualization |
| `seaborn` | Heatmaps and styled plots |
| `scikit-learn` | KNN, SVM, PCA, GridSearchCV, metrics |

---

## 📓 Notebook Structure

| Section | Description |
|---------|-------------|
| Setup & Imports | All libraries and device configuration |
| 1. Dataset & Visualization | Load Fashion MNIST, plot sample grid |
| 2. CNN — Methods & Tuning | Define FashionCNN, train with learning rate search |
| 2B. KNN — Methods & Tuning | Baseline KNN + GridSearchCV with PCA pipeline |
| 2C. SVM — Methods & Tuning | Baseline SVM (RBF & Poly) + GridSearchCV with PCA pipeline |
| 2D. Reusable Helper Functions | `evaluate_sklearn_model()`, `build_sklearn_pipeline()` |
| 3. Preliminary Results | CNN training curves, confusion matrices for all models |
| 4. Final Testing & Visual Inference | Visual predictions on test batch (green = correct, red = wrong) |
| 5. Summary — All Methods | Combined comparison table across all 4 methods |

---

## 👥 Team

| Name | Student ID |
|------|------------|
| Norah Aljayan | 444200832 |
| Rana Alngashy | 444204737 |
| Ghalia Alkhaldi | 444200534 |
| Layan Alhowaimel | 444200969 |

**Course:** IT461 — Practical Machine Learning  
**Institution:** King Saud University  
**Semester:** Spring 2026

---

## 📄 Dataset Reference

> Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. [arXiv:1708.07747](https://arxiv.org/abs/1708.07747)

Dataset available at: [Kaggle — Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
