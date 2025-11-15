"""
Visualization Utilities

Functions for creating plots and visualizations for the intrusion detection project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


def plot_correlation_heatmap(df: pd.DataFrame, 
                             figsize: Tuple[int, int] = (20, 16),
                             save_path: Optional[str] = None):
    """
    Plot correlation heatmap for numerical features.
    
    Args:
        df: Input DataFrame
        figsize: Figure size tuple
        save_path: Path to save figure (optional)
    """
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Compute correlation
    corr_matrix = df[numerical_cols].corr()
    
    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_class_distribution(y, 
                           labels: List[str] = ['Normal', 'Attack'],
                           save_path: Optional[str] = None):
    """
    Plot target variable class distribution.
    
    Args:
        y: Target variable
        labels: Class labels for display
        save_path: Path to save figure (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    counts = pd.Series(y).value_counts()
    ax1.bar(range(len(counts)), counts.values, color=['#2ecc71', '#e74c3c'])
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels)
    ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count')
    
    # Pie chart
    ax2.pie(counts.values, labels=labels, autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'], startangle=90)
    ax2.set_title('Class Distribution Percentage', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(feature_importance: pd.DataFrame,
                           top_n: int = 20,
                           save_path: Optional[str] = None):
    """
    Plot feature importance rankings.
    
    Args:
        feature_importance: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to display
        save_path: Path to save figure (optional)
    """
    top_features = feature_importance.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance', fontsize=13, fontweight='bold')
    plt.ylabel('Features', fontsize=13, fontweight='bold')
    plt.title(f'Top {top_n} Most Important Features', fontsize=15, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrices(models_predictions: dict,
                           y_test,
                           save_path: Optional[str] = None):
    """
    Plot confusion matrices for multiple models.
    
    Args:
        models_predictions: Dict of {model_name: predictions}
        y_test: True labels
        save_path: Path to save figure (optional)
    """
    from sklearn.metrics import confusion_matrix
    
    n_models = len(models_predictions)
    n_cols = 2
    n_rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6*n_rows))
    axes = axes.ravel() if n_models > 1 else [axes]
    
    for idx, (model_name, y_pred) in enumerate(models_predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Normal', 'Attack'],
                    yticklabels=['Normal', 'Attack'])
        axes[idx].set_title(f'{model_name}', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_roc_curves(models_probabilities: dict,
                   y_test,
                   save_path: Optional[str] = None):
    """
    Plot ROC curves for multiple models.
    
    Args:
        models_probabilities: Dict of {model_name: predicted_probabilities}
        y_test: True labels
        save_path: Path to save figure (optional)
    """
    from sklearn.metrics import roc_curve, auc
    
    plt.figure(figsize=(10, 8))
    
    for model_name, y_proba in models_probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    plt.title('ROC Curves - All Models', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_parity_plots(models_probabilities: dict,
                     y_test,
                     save_path: Optional[str] = None):
    """
    Plot parity plots (predicted probability vs actual class) for multiple models.
    
    Args:
        models_probabilities: Dict of {model_name: predicted_probabilities}
        y_test: True labels
        save_path: Path to save figure (optional)
    """
    n_models = len(models_probabilities)
    n_cols = 2
    n_rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6*n_rows))
    axes = axes.ravel() if n_models > 1 else [axes]
    
    for idx, (model_name, y_proba) in enumerate(models_probabilities.items()):
        axes[idx].scatter(y_test, y_proba, alpha=0.3, s=10)
        axes[idx].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
        axes[idx].set_xlabel('Actual Class', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Predicted Probability', fontsize=12, fontweight='bold')
        axes[idx].set_title(f'{model_name} - Parity Plot', fontsize=14, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim([-0.1, 1.1])
        axes[idx].set_ylim([-0.1, 1.1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_metrics_comparison(results_df: pd.DataFrame,
                           save_path: Optional[str] = None):
    """
    Plot bar chart comparing metrics across models.
    
    Args:
        results_df: DataFrame with model metrics
        save_path: Path to save figure (optional)
    """
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    available_metrics = [m for m in metrics_to_plot if m in results_df.columns]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    results_df[available_metrics].T.plot(kind='bar', ax=ax, width=0.8)
    ax.set_xlabel('Metrics', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.legend(title='Models', fontsize=11)
    ax.set_ylim([0.5, 1.0])
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
