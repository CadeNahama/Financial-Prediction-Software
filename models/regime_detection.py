"""
Market regime detection and classification.
"""
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from datetime import date
import yaml
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import get_db_engine, FeatureDaily, PriceDaily
from data.ingest_macro import MacroDaily


def detect_market_regime(engine, ticker='SPY'):
    """
    Detect market regime based on SPY price action and macro indicators.
    
    Regimes:
    - Bull (trending up, low vol)
    - Bear (trending down, high vol)
    - Sideways (range-bound, moderate vol)
    - High Volatility (VIX > 25)
    - Low Volatility (VIX < 15)
    
    Returns:
        DataFrame with regime labels
    """
    # Get SPY data
    with Session(engine) as session:
        prices = session.query(PriceDaily).filter(
            PriceDaily.ticker == ticker
        ).order_by(PriceDaily.date).all()
        
        if not prices:
            return None
        
        spy_df = pd.DataFrame([{
            'date': p.date,
            'adj_close': p.adj_close
        } for p in prices])
        spy_df['date'] = pd.to_datetime(spy_df['date'])
        spy_df = spy_df.sort_values('date').reset_index(drop=True)
        
        # Get macro data
        macro = session.query(MacroDaily).order_by(MacroDaily.date).all()
        if macro:
            macro_df = pd.DataFrame([{
                'date': m.date,
                'vix': m.vix,
                'rate_spread': m.rate_spread
            } for m in macro])
            macro_df['date'] = pd.to_datetime(macro_df['date'])
            macro_df = macro_df.sort_values('date').reset_index(drop=True)
            
            # Merge
            merged = pd.merge(spy_df, macro_df, on='date', how='left')
        else:
            merged = spy_df.copy()
            merged['vix'] = np.nan
            merged['rate_spread'] = np.nan
    
    # Calculate features for regime detection
    merged['ma_200'] = merged['adj_close'].rolling(200).mean()
    merged['ma_50'] = merged['adj_close'].rolling(50).mean()
    merged['return_63d'] = merged['adj_close'].pct_change(63)
    merged['volatility_63d'] = merged['adj_close'].pct_change().rolling(63).std() * np.sqrt(252)
    
    # Regime classification
    regimes = pd.DataFrame(index=merged.index)
    regimes['date'] = merged['date']
    
    # Trend-based regimes
    merged['above_200ma'] = (merged['adj_close'] > merged['ma_200']).astype(int)
    merged['above_50ma'] = (merged['adj_close'] > merged['ma_50']).astype(int)
    merged['bull_regime'] = (merged['above_200ma'] & merged['above_50ma'] & (merged['return_63d'] > 0.05)).astype(int)
    merged['bear_regime'] = ((~merged['above_200ma']) & (merged['return_63d'] < -0.05)).astype(int)
    merged['sideways_regime'] = ((merged['return_63d'].abs() < 0.05) & (~merged['bull_regime']) & (~merged['bear_regime'])).astype(int)
    
    # Volatility-based regimes
    if 'vix' in merged.columns:
        merged['vix'].fillna(method='ffill', inplace=True)
        merged['high_vol_regime'] = (merged['vix'] > 25).astype(int)
        merged['low_vol_regime'] = (merged['vix'] < 15).astype(int)
        merged['extreme_vol_regime'] = (merged['vix'] > 30).astype(int)
    else:
        merged['high_vol_regime'] = 0
        merged['low_vol_regime'] = 0
        merged['extreme_vol_regime'] = 0
    
    # Combined regime label
    def classify_regime(row):
        if row['extreme_vol_regime']:
            return 'extreme_volatility'
        elif row['bear_regime']:
            return 'bear_market'
        elif row['bull_regime']:
            return 'bull_market'
        elif row['high_vol_regime']:
            return 'high_volatility'
        elif row['sideways_regime']:
            return 'sideways'
        elif row['low_vol_regime']:
            return 'low_volatility'
        else:
            return 'normal'
    
    regimes['regime'] = merged.apply(classify_regime, axis=1)
    regimes['regime_code'] = regimes['regime'].map({
        'bull_market': 1,
        'bear_market': -1,
        'sideways': 0,
        'high_volatility': 2,
        'extreme_volatility': 3,
        'low_volatility': -2,
        'normal': 0
    })
    
    # Add feature columns
    feature_cols = ['bull_regime', 'bear_regime', 'sideways_regime', 
                   'high_vol_regime', 'low_vol_regime', 'extreme_vol_regime',
                   'above_200ma', 'above_50ma', 'return_63d', 'volatility_63d']
    for col in feature_cols:
        if col in merged.columns:
            regimes[col] = merged[col].values
    
    if 'vix' in merged.columns:
        regimes['vix'] = merged['vix'].values
    
    return regimes


def add_regime_features_to_features(engine):
    """Add regime features to all ticker features."""
    # Get regime data
    regime_df = detect_market_regime(engine)
    if regime_df is None or regime_df.empty:
        print("No regime data available.")
        return
    
    # Get all tickers
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'tickers.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    tickers = config['tickers']
    
    regime_features = ['regime_code', 'bull_regime', 'bear_regime', 'sideways_regime',
                      'high_vol_regime', 'low_vol_regime', 'extreme_vol_regime',
                      'above_200ma', 'above_50ma']
    
    with Session(engine) as session:
        for ticker in tickers:
            if ticker == 'SPY':
                continue
            
            feature_records = session.query(FeatureDaily).filter(
                FeatureDaily.ticker == ticker
            ).order_by(FeatureDaily.date).all()
            
            for feat_record in feature_records:
                if isinstance(feat_record.features, dict):
                    feat_dict = feat_record.features.copy()
                else:
                    feat_dict = {}
                
                # Add regime features for this date
                regime_row = regime_df[regime_df['date'] == pd.Timestamp(feat_record.date)]
                if not regime_row.empty:
                    for col in regime_features:
                        if col in regime_row.columns:
                            val = regime_row[col].iloc[0]
                            if pd.notna(val):
                                feat_dict[col] = float(val) if col != 'regime' else str(val)
                    
                    feat_record.features = feat_dict
            
            session.commit()
    
    print("Regime features added to all tickers.")


if __name__ == '__main__':
    engine = get_db_engine()
    add_regime_features_to_features(engine)

