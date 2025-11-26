# Network Intrusion Detection System (NIDS)

**Multi-class classification of network traffic using Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Project Overview

This project implements a **Network Intrusion Detection System** using machine learning to classify network traffic into attack categories. Built as an academic project for third-year engineering students, it demonstrates core ML concepts with production-quality code.

**Dataset:** CIC-IDS2017 (~844 MB, 8 CSV files)  
**Models:** Logistic Regression, SVC, PCA+LogReg  
**Approach:** Multi-class classification with 70-30 stratified split  
**Validation:** 5-fold cross-validation

---

## 🚀 Quick Start

```bash
# Clone repository
git clone <repository-url>
cd MLCEProject

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook
# 1. Open 01_data_exploration_eda.ipynb (downloads dataset automatically)
# 2. Run notebooks 1 → 2 → 3 → 4 sequentially
```

**Dataset downloads automatically** in Notebook 1 via Kaggle API.

---

## 📊 Problem Statement

Cybersecurity threats cost organizations an average of **$4.45 million per data breach** (IBM Security, 2023). Traditional signature-based intrusion detection systems cannot identify novel attack patterns, making machine learning crucial for modern network security.

**Objective:** Build a multi-class classifier to distinguish between normal traffic and various attack types (DoS, DDoS, PortScan, Brute Force, Web Attacks, etc.)

---

## 🎯 Professor Requirements

This project fulfills all 7 requirements:

1. ✅ Problem statement & motivation
2. ✅ EDA with manual IQR outlier detection & correlation heatmaps
3. ✅ I/O variables defined, multi-class classification chosen
4. ✅ 70-30 stratified train-test split, 3 models trained
5. ✅ Parity plots & classification metrics (Accuracy, Precision, Recall, F1)
6. ✅ 5-fold stratified cross-validation
7. ✅ Comprehensive conclusions with model comparison

---

## 📁 Project Structure

```
MLCEProject/
├── notebooks/
│   ├── 01_data_exploration_eda.ipynb              # EDA, outlier detection
│   ├── 02_preprocessing_feature_engineering.ipynb # Preprocessing, split
│   ├── 03_model_training_evaluation.ipynb        # Models, parity plots
│   └── 04_cross_validation_analysis.ipynb        # CV, conclusions
├── data/
│   └── CICIDS2017/                               # Dataset (auto-downloaded)
├── outputs/                                       # Generated plots
├── IMPLEMENTATION.md                              # Detailed guide
├── requirements.txt                               # Dependencies
└── README.md                                      # This file
```

---

## 🛠️ Methodology

### Data Pipeline
1. **Data Loading:** 8 CSV files from CIC-IDS2017 (~2.5M records)
2. **EDA:** Manual IQR outlier detection, correlation analysis
3. **Preprocessing:** Outlier capping, 3 engineered features, StandardScaler
4. **Splitting:** 70-30 stratified split (maintains class balance)

### Models
1. **Logistic Regression** - Baseline (multinomial)
2. **SVC** - RBF kernel for non-linear decision boundaries
3. **PCA + Logistic Regression** - Dimensionality reduction (95% variance)

### Evaluation
- **Metrics:** Accuracy, Precision (macro), Recall (macro), F1-Score
- **Visualization:** Parity plots, confusion matrices, ROC curves
- **Validation:** 5-fold stratified cross-validation

---

## 📈 Key Features

- **Automatic dataset download** - No manual Kaggle setup needed
- **Reproducible workflow** - 4 sequential notebooks
- **Academic-appropriate complexity** - Matches reference benchmark (~1,300 lines)
- **Team collaboration** - Clear member responsibilities
- **Git-optimized** - Large files excluded, fast clone/push

---

## 🔧 Dependencies

| Package | Purpose |
|---------|---------|
| numpy, pandas | Data manipulation |
| scikit-learn | ML models & preprocessing |
| matplotlib, seaborn | Visualization |
| kagglehub | Automatic dataset download |
| jupyter | Notebook environment |

**Install all:** `pip install -r requirements.txt`

---

## 👥 Team Workflow

### Notebook Assignments
- **Member 1:** Notebook 1 - EDA & outlier detection
- **Member 2:** Notebook 2 - Preprocessing & feature engineering
- **Member 3:** Notebook 3 - Model training & evaluation
- **Member 4:** Notebook 4 - Cross-validation & conclusions

### Git Workflow
```bash
git pull                  # Get latest code
# Run your assigned notebook
git add notebooks/        # Commit your changes
git commit -m "Updated notebook X"
git push                  # No large files, fast upload
```

Dataset (844 MB) is **excluded** from Git - downloads automatically.

---

## 📊 Results

*Results will be filled after running all notebooks*

**Best Model:** [To be determined after Notebook 4]  
**Test Accuracy:** [Fill after execution]  
**Cross-Validation Score:** [Fill after execution]

See Notebook 4 for comprehensive analysis and conclusions.

---

## 🎓 Academic Context

**Course:** Machine Learning in Chemical Engineering  
**Level:** Third-year engineering students  
**Complexity:** Medium (not too simple, not over-engineered)  
**Benchmark:** Matches REFERENCE.ipynb complexity level

---

## 📖 Documentation

- **IMPLEMENTATION.md** - Quick implementation guide
- **Notebook comments** - Inline explanations
- **Code templates** - Reusable patterns

---

## 🔬 Dataset

**CIC-IDS2017** by Canadian Institute for Cybersecurity  
**Source:** https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset  
**Size:** ~844 MB (8 CSV files)  
**Records:** ~2.5 million network flows  
**Features:** 78+ network traffic features  
**Labels:** BENIGN, DoS, DDoS, PortScan, Brute Force, Web Attack, Infiltration

**Citation:**  
Sharafaldin, I., Lashkari, A.H., & Ghorbani, A.A. (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. ICISSP.

---

## ⚡ Performance

**Expected Runtime:**
- Notebook 1: 5-10 minutes
- Notebook 2: 3-5 minutes
- Notebook 3: 15-30 minutes (SVC training)
- Notebook 4: 20-40 minutes (CV + GridSearch)

**Total:** 45-85 minutes (depending on CPU)

---

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `Dataset not found` | Run Notebook 1 (auto-downloads) |
| SVC training slow | Normal for large datasets (15-30 min) |
| Git shows CSV files | Verify `.gitignore` configuration |

---

## 📞 Support

See `IMPLEMENTATION.md` for detailed implementation guide.

---

## 📝 License

This project is created for academic purposes.

---

**Status:** ✅ Implementation Complete - Ready for Execution  
**Last Updated:** November 2025
