"""
Enhanced training pipeline with feature selection, hyperparameter tuning, and regime detection.
"""
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from datetime import datetime, date
import yaml
import pickle
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import get_db_engine, FeatureDaily, Label
from features.merge_features import load_all_features
from models.feature_selection import select_features_importance, combine_feature_selection
from models.hyperparameter_tuning import tune_model
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, log_loss
from tqdm import tqdm

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


def load_modeling_config():
    """Load modeling configuration."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['modeling']


def prepare_training_data(engine, tickers, start_date, end_date, horizon=63, use_triple_barrier=False):
    """Prepare training data with optional triple barrier labels."""
    all_X = []
    all_y = []
    all_dates = []
    all_tickers_list = []
    
    for ticker in tickers:
        features_df = load_all_features(engine, ticker, start_date, end_date)
        if features_df is None or features_df.empty:
            continue
        
        with Session(engine) as session:
            if use_triple_barrier:
                # Use triple barrier labels
                labels = session.query(Label).filter(
                    Label.ticker == ticker,
                    Label.date >= start_date,
                    Label.date <= end_date,
                    Label.triple_barrier_label.isnot(None)
                ).order_by(Label.date).all()
                
                if not labels:
                    continue
                
                labels_df = pd.DataFrame([{
                    'date': l.date,
                    'label': l.triple_barrier_label
                } for l in labels])
                labels_df['date'] = pd.to_datetime(labels_df['date'])
                
                # Convert triple barrier labels: 1=profit, -1=stop, 0=time -> binary: 1 if profit, 0 otherwise
                labels_df['label_binary'] = (labels_df['label'] == 1).astype(int)
            else:
                # Use forward return labels
                labels = session.query(Label).filter(
                    Label.ticker == ticker,
                    Label.date >= start_date,
                    Label.date <= end_date
                ).order_by(Label.date).all()
                
                if not labels:
                    continue
                
                labels_df = pd.DataFrame([{
                    'date': l.date,
                    'label_binary': getattr(l, f'label_horizon_{horizon}d')
                } for l in labels])
                labels_df['date'] = pd.to_datetime(labels_df['date'])
        
        # Merge features and labels
        merged = pd.merge(features_df, labels_df, on='date', how='inner')
        if merged.empty:
            continue
        
        # Extract feature columns
        feature_cols = [col for col in merged.columns 
                       if col not in ['ticker', 'date', 'label_binary', 'label']]
        feature_cols = [col for col in feature_cols if merged[col].notna().any()]
        
        X_ticker = merged[feature_cols].fillna(0)
        y_ticker = merged['label_binary'].values
        
        valid_mask = ~pd.isna(y_ticker)
        X_ticker = X_ticker[valid_mask]
        y_ticker = y_ticker[valid_mask]
        dates_ticker = merged['date'][valid_mask].values
        
        if len(X_ticker) > 0:
            all_X.append(X_ticker)
            all_y.append(y_ticker)
            all_dates.extend(dates_ticker)
            all_tickers_list.extend([ticker] * len(X_ticker))
    
    if not all_X:
        return None, None, None, None, None
    
    # Combine all tickers
    X = pd.concat(all_X, ignore_index=True)
    y = np.concatenate(all_y)
    dates = np.array(all_dates)
    tickers_array = np.array(all_tickers_list)
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    return X, y, dates, tickers_array, feature_names


def train_walk_forward_enhanced(engine, tickers, config, use_triple_barrier=False, tune_hyperparams=True, use_feature_selection=True):
    """Enhanced walk-forward training with feature selection and hyperparameter tuning."""
    wf_config = config['walk_forward']
    train_years = wf_config['train_years']
    validation_years = wf_config['validation_years']
    test_years = wf_config['test_years']
    step_years = wf_config['step_years']
    horizon = 63
    algorithm = config.get('algorithm', 'xgboost')
    
    # Get date range
    with Session(engine) as session:
        earliest = session.query(Label.date).order_by(Label.date).first()
        latest = session.query(Label.date).order_by(Label.date.desc()).first()
        
        if not earliest or not latest:
            raise ValueError("No label data found in database.")
        
        start_date = earliest[0]
        end_date = latest[0]
    
    print(f"Data range: {start_date} to {end_date}")
    
    current_start = start_date
    results = []
    models = []
    feature_columns_list = []
    
    while current_start < end_date:
        train_end = date(current_start.year + train_years, current_start.month, current_start.day)
        val_start = train_end
        val_end = date(val_start.year + validation_years, val_start.month, val_start.day)
        test_start = val_end
        test_end = date(test_start.year + test_years, test_start.month, test_start.day)
        
        if test_end > end_date:
            break
        
        print(f"\n=== Walk-forward window ===")
        print(f"Train: {current_start} to {train_end}")
        print(f"Val: {val_start} to {val_end}")
        print(f"Test: {test_start} to {test_end}")
        
        # Prepare data
        X_train, y_train, _, _, feature_names = prepare_training_data(
            engine, tickers, current_start, train_end, horizon, use_triple_barrier
        )
        X_val, y_val, _, _, _ = prepare_training_data(
            engine, tickers, val_start, val_end, horizon, use_triple_barrier
        )
        X_test, y_test, _, _, _ = prepare_training_data(
            engine, tickers, test_start, test_end, horizon, use_triple_barrier
        )
        
        if X_train is None or X_val is None or X_test is None:
            current_start = date(current_start.year + step_years, current_start.month, current_start.day)
            continue
        
        # Ensure consistent feature columns
        common_cols = list(set(X_train.columns) & set(X_val.columns) & set(X_test.columns))
        X_train = X_train[common_cols]
        X_val = X_val[common_cols]
        X_test = X_test[common_cols]
        feature_names = common_cols
        
        print(f"Training samples: {len(X_train)}, Features: {len(feature_names)}")
        
        # Feature selection
        selected_features = feature_names
        if use_feature_selection and len(feature_names) > 50:
            print("Applying feature selection...")
            # First train a basic model for feature importance
            if algorithm == 'xgboost' and XGBOOST_AVAILABLE:
                base_model = xgb.XGBClassifier(n_estimators=50, random_state=42)
            elif algorithm == 'lightgbm' and LIGHTGBM_AVAILABLE:
                base_model = lgb.LGBMClassifier(n_estimators=50, random_state=42)
            else:
                base_model = None
            
            if base_model:
                base_model.fit(X_train, y_train)
                selected_features = select_features_importance(
                    base_model, feature_names, top_k=50
                )
                X_train = X_train[selected_features]
                X_val = X_val[selected_features]
                X_test = X_test[selected_features]
                print(f"Selected {len(selected_features)} features")
        
        # Hyperparameter tuning
        if tune_hyperparams:
            print("Tuning hyperparameters...")
            try:
                model, best_params = tune_model(X_train, y_train, algorithm, n_trials=30, cv=3)
                model.fit(X_train, y_train)
                print(f"Best parameters: {best_params}")
            except Exception as e:
                print(f"Hyperparameter tuning failed: {e}, using defaults")
                if algorithm == 'xgboost' and XGBOOST_AVAILABLE:
                    model = xgb.XGBClassifier(**config.get('hyperparameters', {}), random_state=42)
                elif algorithm == 'lightgbm' and LIGHTGBM_AVAILABLE:
                    model = lgb.LGBMClassifier(**config.get('hyperparameters', {}), random_state=42)
                else:
                    raise ValueError(f"Algorithm {algorithm} not available")
                model.fit(X_train, y_train)
        else:
            # Use default hyperparameters
            if algorithm == 'xgboost' and XGBOOST_AVAILABLE:
                model = xgb.XGBClassifier(**config.get('hyperparameters', {}), random_state=42)
            elif algorithm == 'lightgbm' and LIGHTGBM_AVAILABLE:
                model = lgb.LGBMClassifier(**config.get('hyperparameters', {}), random_state=42)
            else:
                raise ValueError(f"Algorithm {algorithm} not available")
            model.fit(X_train, y_train)
        
        # Calibration
        if config.get('calibration', {}).get('enabled', False):
            cal_method = config['calibration'].get('method', 'isotonic')
            model = CalibratedClassifierCV(model, method=cal_method, cv=3)
            model.fit(X_train, y_train)
        
        # Evaluate
        for split_name, X_split, y_split in [('val', X_val, y_val), ('test', X_test, y_test)]:
            y_pred = model.predict(X_split)
            y_proba = model.predict_proba(X_split)[:, 1]
            
            accuracy = accuracy_score(y_split, y_pred)
            precision = precision_score(y_split, y_pred, zero_division=0)
            recall = recall_score(y_split, y_pred, zero_division=0)
            auc = roc_auc_score(y_split, y_proba) if len(np.unique(y_split)) > 1 else 0.0
            logloss = log_loss(y_split, y_proba)
            
            result = {
                'window': f"{current_start}_{test_end}",
                'split': split_name,
                'start_date': current_start.isoformat(),
                'test_end': test_end.isoformat(),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'auc': auc,
                'log_loss': logloss,
                'n_samples': len(X_split),
                'n_features': len(selected_features)
            }
            results.append(result)
            
            print(f"{split_name.upper()} - Accuracy: {accuracy:.3f}, AUC: {auc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
        
        models.append(model)
        feature_columns_list.append(selected_features)
        
        current_start = date(current_start.year + step_years, current_start.month, current_start.day)
    
    if not models:
        raise ValueError("No models trained.")
    
    final_model = models[-1]
    final_features = feature_columns_list[-1]
    
    return {
        'model': final_model,
        'feature_columns': final_features,
        'validation_results': pd.DataFrame(results),
        'all_models': models
    }


def train_model_enhanced(tickers=None, use_triple_barrier=False, tune_hyperparams=True, use_feature_selection=True):
    """Enhanced training function."""
    config = load_modeling_config()
    
    if tickers is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'tickers.yaml')
        with open(config_path, 'r') as f:
            ticker_config = yaml.safe_load(f)
        tickers = [t for t in ticker_config['tickers'] if t != 'SPY']
    
    engine = get_db_engine()
    
    print(f"Enhanced training for {len(tickers)} tickers...")
    print(f"Algorithm: {config.get('algorithm', 'xgboost')}")
    print(f"Triple barrier: {use_triple_barrier}")
    print(f"Hyperparameter tuning: {tune_hyperparams}")
    print(f"Feature selection: {use_feature_selection}")
    
    model_result = train_walk_forward_enhanced(
        engine, tickers, config, use_triple_barrier, tune_hyperparams, use_feature_selection
    )
    
    # Save model
    output_path = 'models/trained_model_enhanced.pkl'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_result['model'], f)
    
    metadata = {
        'feature_columns': model_result['feature_columns'],
        'trained_date': datetime.now().isoformat(),
        'validation_results': model_result['validation_results'].to_dict('records'),
        'use_triple_barrier': use_triple_barrier,
        'tune_hyperparams': tune_hyperparams,
        'use_feature_selection': use_feature_selection
    }
    
    metadata_path = output_path.replace('.pkl', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel saved to {output_path}")
    
    # Print summary
    val_results = model_result['validation_results']
    print("\n=== Validation Summary ===")
    print(val_results.groupby('split')[['accuracy', 'auc', 'precision', 'recall']].mean())
    
    return model_result


if __name__ == '__main__':
    train_model_enhanced()

