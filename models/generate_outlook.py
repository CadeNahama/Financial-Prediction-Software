"""
Generate daily outlook for tickers.
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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import get_db_engine, FeatureDaily, PriceDaily
from features.merge_features import load_all_features


def load_model(model_path='models/trained_model.pkl'):
    """Load trained model and metadata."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    metadata_path = model_path.replace('.pkl', '_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return model, metadata


def load_thresholds():
    """Load thresholds from config."""
    with open('configs/tickers.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config.get('thresholds', {'long': 0.55, 'no_trade': 0.50})


def get_current_price(engine, ticker):
    """Get most recent price for position sizing."""
    with Session(engine) as session:
        price = session.query(PriceDaily).filter(
            PriceDaily.ticker == ticker
        ).order_by(PriceDaily.date.desc()).first()
        
        if price:
            return price.adj_close
        return None


def calculate_position_size(p_up, volatility, max_size=0.15, vol_cap=0.50):
    """
    Calculate position size based on probability and risk.
    
    Args:
        p_up: Probability of positive return
        volatility: Current volatility (annualized)
        max_size: Maximum position size (15% default)
        vol_cap: Volatility cap (avoid positions with >50% vol)
    
    Returns:
        Suggested position size (0 to max_size)
    """
    # Base size from probability
    base_size = (p_up - 0.50) * 2 * max_size  # Scale from 0 to max_size
    
    # Reduce size if volatility is high
    if volatility > vol_cap:
        vol_penalty = vol_cap / volatility
        base_size *= vol_penalty
    
    # Clamp to [0, max_size]
    return max(0, min(max_size, base_size))


def get_key_feature_drivers(model, features_df, feature_cols, top_n=5):
    """
    Get top feature drivers for prediction.
    
    Note: This is a simplified version. For tree models, you'd use
    feature_importances_ or SHAP values for better interpretability.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = dict(zip(feature_cols, importances))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return {feat: float(imp) for feat, imp in top_features}
    else:
        # Fallback: return None or use correlation
        return None


def generate_outlook(engine, ticker, model, metadata, prediction_date=None):
    """
    Generate outlook for a single ticker.
    
    Returns:
        Dictionary with outlook data
    """
    if prediction_date is None:
        prediction_date = date.today()
    
    # Load features for most recent date
    features_df = load_all_features(engine, ticker)
    
    if features_df is None or features_df.empty:
        return None
    
    # Get most recent features (before prediction_date)
    features_df = features_df[features_df['date'] <= pd.Timestamp(prediction_date)]
    
    if features_df.empty:
        return None
    
    latest_features = features_df.iloc[-1]
    feature_cols = metadata['feature_columns']
    
    # Extract features in correct order
    X = pd.DataFrame([latest_features[feature_cols]], columns=feature_cols)
    X = X.fillna(0)
    
    # Predict
    p_up_63d = model.predict_proba(X)[0, 1]
    p_up_84d = p_up_63d  # For now, use same probability (could train separate model)
    
    # Get current volatility for position sizing
    volatility = latest_features.get('volatility_20d', 0.20)  # Default 20% if not available
    
    # Calculate position size
    thresholds = load_thresholds()
    suggested_size = calculate_position_size(p_up_63d, volatility)
    
    # Determine action
    if p_up_63d > thresholds['long']:
        action = 'Long'
    elif p_up_63d > thresholds['no_trade']:
        action = 'No Trade'
    else:
        action = 'Reduce'
    
    # Get key drivers
    key_drivers = get_key_feature_drivers(model, latest_features, feature_cols)
    
    # Get current price
    current_price = get_current_price(engine, ticker)
    
    outlook = {
        'ticker': ticker,
        'date': prediction_date.isoformat(),
        'p_up_63d': float(p_up_63d),
        'p_up_84d': float(p_up_84d),
        'recommended_action': action,
        'suggested_size': float(suggested_size),
        'current_price': float(current_price) if current_price else None,
        'volatility': float(volatility),
        'key_feature_drivers': key_drivers
    }
    
    return outlook


def generate_all_outlooks(tickers=None, output_path='reports/outlook.json'):
    """
    Generate outlook for all tickers.
    
    Args:
        tickers: List of tickers (None = all in config)
        output_path: Path to save outlook JSON
    """
    # Load model
    model, metadata = load_model()
    
    # Load tickers
    if tickers is None:
        with open('configs/tickers.yaml', 'r') as f:
            config = yaml.safe_load(f)
        tickers = [t for t in config['tickers'] if t != 'SPY']
    
    engine = get_db_engine()
    
    print(f"Generating outlook for {len(tickers)} tickers...")
    
    outlooks = []
    
    for ticker in tickers:
        outlook = generate_outlook(engine, ticker, model, metadata)
        if outlook:
            outlooks.append(outlook)
            print(f"  {ticker}: {outlook['recommended_action']} (p={outlook['p_up_63d']:.3f})")
        else:
            print(f"  {ticker}: No data available")
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(outlooks, f, indent=2)
    
    print(f"\nOutlook saved to {output_path}")
    
    return outlooks


if __name__ == '__main__':
    generate_all_outlooks()

