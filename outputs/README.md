# Outputs Directory

This folder stores all generated figures, plots, and analysis results from the project.

## 📊 Contents

### Exploratory Data Analysis (EDA)
- `univariate_boxplots.png` - Outlier detection box plots
- `univariate_histograms.png` - Feature distribution histograms
- `bivariate_analysis.png` - Feature distributions by class label
- `correlation_heatmap.png` - Feature correlation matrix
- `target_distribution.png` - Class balance visualization

### Model Evaluation
- `parity_plots.png` - Predicted probability vs actual class for all models
- `metrics_comparison.png` - Bar chart comparing all evaluation metrics
- `roc_curves.png` - ROC curves for all models
- `confusion_matrices.png` - Confusion matrices for all models

### Cross-Validation & Analysis
- `cv_results_comparison.png` - Cross-validation results with error bars
- `learning_curve_logistic_regression.png` - Learning curve for LR
- `learning_curve_svc_rbf.png` - Learning curve for SVC
- `learning_curve_ann_mlp.png` - Learning curve for ANN
- `learning_curve_pca_svc.png` - Learning curve for PCA+SVC
- `feature_importance.png` - Top feature importance rankings

## 📝 Naming Convention

All output files follow this convention:
- Lowercase with underscores
- Descriptive names indicating content
- PNG format for figures (300 DPI)
- Timestamped versions for iterative experiments (optional)

## 🔒 Git Configuration

Generated output files are excluded from git tracking (see `.gitignore`), but this README and folder structure are preserved.

---

**Note:** Run all notebooks to populate this directory with visualizations.
