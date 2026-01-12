"""
Hyperparameter tuning using Optuna.
"""
import numpy as np
import yaml
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    print("Warning: optuna not installed. Install with: pip install optuna")
    OPTUNA_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score


def load_modeling_config():
    """Load modeling configuration."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['modeling']


def tune_xgboost(X_train, y_train, n_trials=50, cv=3):
    """Tune XGBoost hyperparameters using Optuna."""
    if not OPTUNA_AVAILABLE or not XGBOOST_AVAILABLE:
        raise ImportError("Optuna and XGBoost required for hyperparameter tuning")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
        
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"Best parameters: {study.best_params}")
    print(f"Best CV score: {study.best_value:.4f}")
    
    # Create best model
    best_params = study.best_params
    best_params.update({
        'random_state': 42,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    })
    best_model = xgb.XGBClassifier(**best_params)
    
    return best_model, study.best_params


def tune_lightgbm(X_train, y_train, n_trials=50, cv=3):
    """Tune LightGBM hyperparameters using Optuna."""
    if not OPTUNA_AVAILABLE or not LIGHTGBM_AVAILABLE:
        raise ImportError("Optuna and LightGBM required for hyperparameter tuning")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"Best parameters: {study.best_params}")
    print(f"Best CV score: {study.best_value:.4f}")
    
    # Create best model
    best_params = study.best_params
    best_params.update({
        'random_state': 42,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1
    })
    best_model = lgb.LGBMClassifier(**best_params)
    
    return best_model, study.best_params


def tune_model(X_train, y_train, algorithm='xgboost', n_trials=50, cv=3):
    """
    Tune model hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        algorithm: 'xgboost' or 'lightgbm'
        n_trials: Number of optimization trials
        cv: Cross-validation folds
    
    Returns:
        Best model and best parameters
    """
    if not OPTUNA_AVAILABLE:
        print("Optuna not available. Using default hyperparameters.")
        config = load_modeling_config()
        hyperparams = config.get('hyperparameters', {})
        
        if algorithm == 'xgboost' and XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(**hyperparams, random_state=42), hyperparams
        elif algorithm == 'lightgbm' and LIGHTGBM_AVAILABLE:
            return lgb.LGBMClassifier(**hyperparams, random_state=42), hyperparams
        else:
            raise ValueError(f"Algorithm {algorithm} not available")
    
    print(f"Tuning {algorithm} hyperparameters ({n_trials} trials)...")
    
    if algorithm == 'xgboost':
        return tune_xgboost(X_train, y_train, n_trials, cv)
    elif algorithm == 'lightgbm':
        return tune_lightgbm(X_train, y_train, n_trials, cv)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

