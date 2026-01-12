"""
Train model with walk-forward validation.
"""
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from datetime import datetime, date
import yaml
import pickle
import json
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, log_loss
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import get_db_engine, FeatureDaily, Label
from features.merge_features import load_all_features

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
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config['modeling']


def prepare_training_data(engine, tickers, start_date, end_date, horizon=63):
    """
    Prepare training data from database.
    
    Returns:
        X (features), y (labels), dates, tickers
    """
    all_X = []
    all_y = []
    all_dates = []
    all_tickers_list = []
    
    for ticker in tickers:
        # Load features
        features_df = load_all_features(engine, ticker, start_date, end_date)
        if features_df is None or features_df.empty:
            continue
        
        # Load labels
        with Session(engine) as session:
            labels = session.query(Label).filter(
                Label.ticker == ticker,
                Label.date >= start_date,
                Label.date <= end_date
            ).order_by(Label.date).all()
            
            if not labels:
                continue
            
            labels_df = pd.DataFrame([{
                'date': l.date,
                f'label_horizon_{horizon}d': getattr(l, f'label_horizon_{horizon}d')
            } for l in labels])
            labels_df['date'] = pd.to_datetime(labels_df['date'])
        
        # Merge features and labels
        merged = pd.merge(
            features_df,
            labels_df,
            on='date',
            how='inner'
        )
        
        if merged.empty:
            continue
        
        # Extract feature columns (exclude ticker, date, label)
        feature_cols = [col for col in merged.columns 
                       if col not in ['ticker', 'date', f'label_horizon_{horizon}d']]
        
        # Remove columns with all NaN
        feature_cols = [col for col in feature_cols if merged[col].notna().any()]
        
        X_ticker = merged[feature_cols].fillna(0)
        y_ticker = merged[f'label_horizon_{horizon}d'].values
        
        # Remove rows with NaN labels
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
        return None, None, None, None
    
    # Combine all tickers
    X = pd.concat(all_X, ignore_index=True)
    y = np.concatenate(all_y)
    dates = np.array(all_dates)
    tickers_array = np.array(all_tickers_list)
    
    # Ensure feature columns are consistent
    # (in case different tickers have different features)
    print(f"Features: {len(X.columns)} columns, {len(X)} samples")
    print(f"Label distribution: {np.bincount(y.astype(int))}")
    
    return X, y, dates, tickers_array


def create_model(config):
    """Create model based on configuration."""
    algorithm = config.get('algorithm', 'xgboost')
    hyperparams = config.get('hyperparameters', {})
    
    if algorithm == 'xgboost' and XGBOOST_AVAILABLE:
        model = xgb.XGBClassifier(**hyperparams, random_state=42)
    elif algorithm == 'lightgbm' and LIGHTGBM_AVAILABLE:
        model = lgb.LGBMClassifier(**hyperparams, random_state=42)
    else:
        raise ValueError(f"Algorithm {algorithm} not available. Install xgboost or lightgbm.")
    
    return model


def train_walk_forward(engine, tickers, config):
    """
    Train model with walk-forward validation.
    
    Returns:
        Dictionary with model, feature columns, and validation results
    """
    wf_config = config['walk_forward']
    train_years = wf_config['train_years']
    validation_years = wf_config['validation_years']
    test_years = wf_config['test_years']
    step_years = wf_config['step_years']
    horizon = 63  # Default to 63-day horizon
    
    # Get date range from data
    with Session(engine) as session:
        earliest = session.query(Label.date).order_by(Label.date).first()
        latest = session.query(Label.date).order_by(Label.date.desc()).first()
        
        if not earliest or not latest:
            raise ValueError("No label data found in database.")
        
        start_date = earliest[0]
        end_date = latest[0]
    
    print(f"Data range: {start_date} to {end_date}")
    
    # Walk-forward windows
    current_start = start_date
    results = []
    models = []
    feature_columns_list = []
    
    while current_start < end_date:
        # Define windows
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
        X_train, y_train, _, _ = prepare_training_data(engine, tickers, current_start, train_end, horizon)
        X_val, y_val, _, _ = prepare_training_data(engine, tickers, val_start, val_end, horizon)
        X_test, y_test, _, _ = prepare_training_data(engine, tickers, test_start, test_end, horizon)
        
        if X_train is None or X_val is None or X_test is None:
            print("Skipping window: insufficient data")
            current_start = date(current_start.year + step_years, current_start.month, current_start.day)
            continue
        
        # Ensure consistent feature columns
        common_cols = list(set(X_train.columns) & set(X_val.columns) & set(X_test.columns))
        X_train = X_train[common_cols]
        X_val = X_val[common_cols]
        X_test = X_test[common_cols]
        
        print(f"Training samples: {len(X_train)}, Features: {len(common_cols)}")
        
        # Create and train model
        model = create_model(config)
        model.fit(X_train, y_train)
        
        # Calibrate if enabled
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
                'n_samples': len(X_split)
            }
            results.append(result)
            
            print(f"{split_name.upper()} - Accuracy: {accuracy:.3f}, AUC: {auc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
        
        # Store model and features for latest window
        models.append(model)
        feature_columns_list.append(common_cols)
        
        # Move window forward
        current_start = date(current_start.year + step_years, current_start.month, current_start.day)
    
    # Use the last model for production
    if not models:
        raise ValueError("No models trained. Check data availability.")
    
    final_model = models[-1]
    final_features = feature_columns_list[-1]
    
    return {
        'model': final_model,
        'feature_columns': final_features,
        'validation_results': pd.DataFrame(results),
        'all_models': models
    }


def save_model(model_result, output_path='models/trained_model.pkl'):
    """Save trained model and metadata."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save model
    with open(output_path, 'wb') as f:
        pickle.dump(model_result['model'], f)
    
    # Save metadata
    metadata = {
        'feature_columns': model_result['feature_columns'],
        'trained_date': datetime.now().isoformat(),
        'validation_results': model_result['validation_results'].to_dict('records')
    }
    
    metadata_path = output_path.replace('.pkl', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {output_path}")
    print(f"Metadata saved to {metadata_path}")


def train_model(tickers=None):
    """
    Main training function.
    
    Args:
        tickers: List of tickers (None = all in database)
    """
    # Load configuration
    config = load_modeling_config()
    
    # Load tickers
    if tickers is None:
        with open('configs/tickers.yaml', 'r') as f:
            ticker_config = yaml.safe_load(f)
        tickers = [t for t in ticker_config['tickers'] if t != 'SPY']  # Exclude benchmark
    
    engine = get_db_engine()
    
    print(f"Training model for {len(tickers)} tickers...")
    print(f"Algorithm: {config.get('algorithm', 'xgboost')}")
    
    # Train with walk-forward validation
    model_result = train_walk_forward(engine, tickers, config)
    
    # Save model
    save_model(model_result)
    
    # Print summary
    val_results = model_result['validation_results']
    print("\n=== Validation Summary ===")
    print(val_results.groupby('split')[['accuracy', 'auc', 'precision', 'recall']].mean())
    
    return model_result


if __name__ == '__main__':
    train_model()

