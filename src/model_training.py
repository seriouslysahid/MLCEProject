"""
Model Training and Evaluation Utilities

Functions for training models, making predictions, and computing metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)


def train_model(model, X_train, y_train, model_name: str = "Model") -> Any:
    """
    Train a machine learning model.
    
    Args:
        model: Scikit-learn compatible model
        X_train: Training features
        y_train: Training labels
        model_name: Name for logging
        
    Returns:
        Trained model
    """
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    print(f"✅ {model_name} trained successfully")
    return model


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> Dict[str, float]:
    """
    Evaluate a trained model and compute comprehensive metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name for results dictionary
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Compute metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
    }
    
    if y_proba is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_test, y_proba)
    
    return metrics


def print_evaluation_report(model, X_test, y_test, model_name: str = "Model"):
    """
    Print detailed evaluation report for a model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name for printing
    """
    y_pred = model.predict(X_test)
    
    print(f"\n{'='*70}")
    print(f"EVALUATION REPORT - {model_name}")
    print('='*70)
    
    # Metrics
    metrics = evaluate_model(model, X_test, y_test, model_name)
    for key, value in metrics.items():
        if key != 'Model':
            print(f"{key:20s}: {value:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Classification Report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
    
    print('='*70)


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Compare multiple models and return sorted results.
    
    Args:
        results: Dictionary of model results
        
    Returns:
        DataFrame with comparison results
    """
    df_results = pd.DataFrame(results).T
    
    # Sort by ROC-AUC (or F1-Score if ROC-AUC not available)
    sort_by = 'ROC-AUC' if 'ROC-AUC' in df_results.columns else 'F1-Score'
    df_results = df_results.sort_values(sort_by, ascending=False)
    
    return df_results


def get_feature_importance(model, feature_names: list, top_n: int = 20) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.
    
    Args:
        model: Trained model (must have coef_ or feature_importances_)
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        DataFrame with feature importance rankings
    """
    if hasattr(model, 'coef_'):
        # Linear models (Logistic Regression, Linear SVM)
        importance = np.abs(model.coef_[0])
    elif hasattr(model, 'feature_importances_'):
        # Tree-based models (Random Forest, etc.)
        importance = model.feature_importances_
    else:
        raise ValueError("Model does not have feature importance attributes")
    
    # Create DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return feature_importance.head(top_n)


def predict_with_confidence(model, X: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions with probability scores.
    
    Args:
        model: Trained model with predict_proba
        X: Input features
        threshold: Classification threshold
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    if not hasattr(model, 'predict_proba'):
        raise ValueError("Model must support probability predictions")
    
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    
    return predictions, probabilities
