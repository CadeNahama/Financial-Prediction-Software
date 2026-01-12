"""
Feature selection using importance-based methods.
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import yaml
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def select_features_importance(model, feature_names, top_k=None, threshold=0.01):
    """
    Select features based on model feature importance.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_k: Select top K features (None = use threshold)
        threshold: Minimum importance threshold
    
    Returns:
        List of selected feature names
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model doesn't have feature_importances_. Returning all features.")
        return feature_names
    
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    if top_k:
        selected = feature_importance.head(top_k)['feature'].tolist()
    else:
        selected = feature_importance[feature_importance['importance'] >= threshold]['feature'].tolist()
    
    print(f"Selected {len(selected)} features from {len(feature_names)} features")
    print(f"Top 10 features:")
    print(feature_importance.head(10))
    
    return selected


def select_features_statistical(X, y, feature_names, top_k=50, method='f_classif'):
    """
    Select features using statistical methods.
    
    Args:
        X: Feature matrix
        y: Labels
        feature_names: List of feature names
        top_k: Number of features to select
        method: 'f_classif' or 'mutual_info'
    
    Returns:
        List of selected feature names
    """
    if method == 'f_classif':
        selector = SelectKBest(f_classif, k=min(top_k, X.shape[1]))
    else:
        selector = SelectKBest(mutual_info_classif, k=min(top_k, X.shape[1]))
    
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    
    print(f"Selected {len(selected_features)} features using {method}")
    
    return selected_features


def combine_feature_selection(model, X, y, feature_names, top_k=50):
    """
    Combine model importance and statistical selection.
    
    Returns:
        List of selected feature names
    """
    # Get model-based selection
    if hasattr(model, 'feature_importances_'):
        model_selected = select_features_importance(model, feature_names, top_k=top_k)
    else:
        model_selected = feature_names
    
    # Get statistical selection
    stat_selected = select_features_statistical(X, y, feature_names, top_k=top_k)
    
    # Union of both methods
    combined = list(set(model_selected) | set(stat_selected))
    
    print(f"Combined selection: {len(combined)} features")
    
    return combined

