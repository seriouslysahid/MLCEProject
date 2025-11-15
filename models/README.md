# Models Directory

This folder stores trained machine learning models and related artifacts.

## 💾 Contents

### Trained Models
- `logistic_regression.pkl` - Logistic Regression classifier
- `svc_rbf.pkl` - Support Vector Classifier with RBF kernel
- `ann_mlp.pkl` - Artificial Neural Network (Multi-layer Perceptron)
- `pca_svc.pkl` - PCA + SVC pipeline

### Preprocessing Artifacts
- `scaler.pkl` - StandardScaler/RobustScaler fitted on training data
- `label_encoders.pkl` - Label encoders for categorical features (if used)

## 📦 Model Format

All models are saved using `joblib` for efficient serialization of scikit-learn objects.

### Loading Models

```python
import joblib

# Load a trained model
model = joblib.load('models/logistic_regression.pkl')

# Load the scaler
scaler = joblib.load('models/scaler.pkl')

# Make predictions
X_new_scaled = scaler.transform(X_new)
predictions = model.predict(X_new_scaled)
```

## 🔒 Git Configuration

Model files are excluded from git tracking due to their size (see `.gitignore`), but this README and folder structure are preserved.

## 📊 Model Metadata

| Model | Size (approx) | Training Time | Best Metric |
|-------|---------------|---------------|-------------|
| Logistic Regression | ~1 MB | Fast | TBD |
| SVC (RBF) | ~5-10 MB | Moderate | TBD |
| ANN (MLP) | ~2-5 MB | Moderate | TBD |
| PCA + SVC | ~5-10 MB | Moderate | TBD |

---

**Note:** Run `03_model_training_evaluation.ipynb` to generate trained models.
