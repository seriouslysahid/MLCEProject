# 🛡️ Network Intrusion Detection System (NIDS)
### Machine Learning Approach to Cybersecurity Threat Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-FF6F00.svg)](https://www.tensorflow.org/)

---

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Results](#results)
- [Contributors](#contributors)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Problem Statement

Cybersecurity threats are evolving rapidly, with the average cost of a data breach reaching **$4.45 million** in 2023 (IBM Security). Traditional signature-based intrusion detection systems struggle to identify novel attack patterns, making machine learning a crucial tool for modern network security.

**This project aims to:**
- Build a binary classification model to distinguish between **normal network traffic** and **attack traffic**
- Evaluate multiple ML algorithms to identify the most effective approach for intrusion detection
- Provide actionable insights through comprehensive model evaluation and feature analysis

The goal is to develop a robust, production-ready model that can help security teams proactively identify and mitigate cyber threats.

---

## 📊 Dataset

**UNSW-NB15 Dataset**  
- **Source:** Australian Centre for Cyber Security (ACCS)
- **Size:** 257,673 records (varies by subset)
- **Features:** 49 attributes including flow features, basic features, content features, and time features
- **Target Variable:** Binary classification
  - `0` = Normal traffic
  - `1` = Attack traffic (9 attack categories consolidated)

**Key Features Include:**
- Protocol type, service, state
- Source/destination bytes, packets, TTL
- TCP window size, flags
- Duration, rate metrics
- And more...

The dataset provides a modern alternative to the dated KDD Cup 99 dataset, with realistic contemporary network traffic patterns.

---

## 🧠 Methodology

### Models Implemented
1. **Logistic Regression** – Baseline linear model with L2 regularization
2. **Support Vector Classifier (SVC)** – RBF kernel for non-linear decision boundaries
3. **Artificial Neural Network (ANN)** – Multi-layer perceptron with dropout regularization
4. **PCA + SVC** – Dimensionality reduction followed by SVM classification

### Pipeline Overview
```
Data Acquisition → EDA & Outlier Detection → Feature Engineering →
Train-Test Split (70-30 Stratified) → Model Training →
Evaluation (Parity Plots, Metrics) → Cross-Validation (10-Fold) →
Learning Curves & Feature Importance
```

### Evaluation Metrics
- **Accuracy** – Overall correctness
- **Precision** – Minimizing false positives (critical for alert fatigue)
- **Recall** – Maximizing attack detection (critical for security)
- **F1-Score** – Harmonic mean balancing precision and recall
- **ROC-AUC** – Model's ability to discriminate between classes

### Key Analyses
- ✅ Univariate outlier detection (IQR method, Z-scores)
- ✅ Correlation heatmaps to identify multicollinearity
- ✅ Parity plots (predicted probabilities vs actual labels)
- ✅ 10-fold stratified cross-validation for robust performance estimates
- ✅ Learning curves to diagnose bias-variance tradeoff
- ✅ Feature importance rankings

---

## 📁 Project Structure

```
MLCE/
│
├── data/                          # Dataset storage (not tracked in git)
│   ├── .gitkeep
│   └── README.md                  # Instructions for downloading data
│
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── 01_data_exploration_eda.ipynb
│   ├── 02_preprocessing_feature_engineering.ipynb
│   ├── 03_model_training_evaluation.ipynb
│   └── 04_cross_validation_analysis.ipynb
│
├── src/                           # Reusable Python modules (optional)
│   ├── __init__.py
│   ├── data_processing.py         # Data loading and cleaning utilities
│   ├── feature_engineering.py     # Feature transformation functions
│   ├── model_training.py          # Model training and evaluation
│   └── visualization.py           # Plotting utilities
│
├── models/                        # Saved trained models
│   ├── .gitkeep
│   └── README.md
│
├── outputs/                       # Generated figures and results
│   ├── .gitkeep
│   └── README.md
│
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- (Optional) Virtual environment tool (venv, conda)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd MLCE
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Follow instructions in `data/README.md`
   - Place the dataset files in the `data/` folder

5. **Launch Jupyter**
   ```bash
   jupyter notebook
   ```

---

## 💻 Usage

### Running the Analysis

Execute notebooks in order:

1. **01_data_exploration_eda.ipynb**  
   Start here to understand the dataset, detect outliers, and visualize distributions

2. **02_preprocessing_feature_engineering.ipynb**  
   Clean data, handle missing values, scale features, and engineer new variables

3. **03_model_training_evaluation.ipynb**  
   Train all four models, generate parity plots, and compute evaluation metrics

4. **04_cross_validation_analysis.ipynb**  
   Perform k-fold CV, plot learning curves, and analyze feature importance

### Quick Start Example
```python
# After running preprocessing notebook
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## 📈 Results

> **Note:** Results will be populated after model training and evaluation.

### Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | TBD | TBD | TBD | TBD | TBD |
| SVC (RBF) | TBD | TBD | TBD | TBD | TBD |
| ANN (MLP) | TBD | TBD | TBD | TBD | TBD |
| PCA + SVC | TBD | TBD | TBD | TBD | TBD |

### Key Findings
- 🔍 **Best performing model:** TBD
- ⚡ **Most important features:** TBD
- 🎯 **Cross-validation insights:** TBD

---

## 👥 Contributors

**Your Name**  
Machine Learning Course Project  
University Name | Fall 2025

Feel free to reach out for questions or collaboration!

---

## 🙏 Acknowledgments

- **UNSW-NB15 Dataset:** Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. *Military Communications and Information Systems Conference (MilCIS)*, 2015.
- **Course Instructor:** [Instructor Name]
- **IBM Security:** Cost of a Data Breach Report 2023
- **scikit-learn & TensorFlow communities** for excellent documentation

---

## 📝 License

This project is for academic purposes only.

---

**Last Updated:** November 2025
