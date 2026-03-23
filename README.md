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
├── notebook.ipynb          # Main Jupyter notebook (all cells)
├── README.md               # Project documentation
│
└── figures/                # Output plots (auto-generated)
    ├── sample_images.png
    ├── confusion_matrix.png
    └── predictions.png
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

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/fashion-mnist-svm.git
cd fashion-mnist-svm
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn kagglehub
```

### 3. Run the notebook
```bash
jupyter notebook notebook.ipynb
```

> **Note:** The notebook will automatically download the Fashion MNIST dataset via `kagglehub` on first run. Make sure you have a Kaggle account and your API key configured.

### Kaggle API Setup
```bash
# Install kaggle CLI
pip install kaggle

# Place your kaggle.json in:
# ~/.kaggle/kaggle.json  (Linux/Mac)
# C:\Users\<user>\.kaggle\kaggle.json  (Windows)
```

---

## 📦 Dependencies

| Package       | Purpose                          |
|---------------|----------------------------------|
| `numpy`       | Numerical operations             |
| `pandas`      | Data loading and manipulation    |
| `matplotlib`  | Plotting and visualization       |
| `seaborn`     | Heatmaps and styled plots        |
| `scikit-learn`| SVM, PCA, GridSearchCV, metrics  |
| `kagglehub`   | Downloading the dataset          |

---

## 📓 Notebook Structure

| Cell | Description |
|------|-------------|
| 1 | Imports & Dataset Download |
| 2 | Load & Explore Data |
| 3 | Prepare Features & Labels + Normalization |
| 4 | Visualize Sample Images |
| 5 | Create Training Subset (10K samples) |
| 6 | Baseline SVM — RBF & Polynomial |
| 7 | GridSearchCV — Best RBF Kernel |
| 8 | GridSearchCV — Best Polynomial Kernel |
| 9 | Evaluate Best Models + Classification Report |
| 10 | Confusion Matrix Heatmap |
| 11 | Visualize Predictions (Correct/Incorrect) |

---

## 👥 Team

| Name | Student ID |
|------|------------|
| [Your Name] | [ID] |
| [Teammate Name] | [ID] |

**Course:** IT461 — Practical Machine Learning  
**Institution:** [Your University]  
**Semester:** Spring 2026

---

## 📄 Dataset Reference

> Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. [arXiv:1708.07747](https://arxiv.org/abs/1708.07747)

Dataset available at: [Kaggle — Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
