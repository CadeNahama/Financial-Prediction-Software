"""
Build macro features from macro data.
"""
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from datetime import date
import yaml
from tqdm import tqdm
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import get_db_engine, FeatureDaily, init_database
from data.ingest_macro import MacroDaily


def get_macro_data(engine, start_date=None):
    """Get macro data from database."""
    with Session(engine) as session:
        query = session.query(MacroDaily).order_by(MacroDaily.date)
        if start_date:
            query = query.filter(MacroDaily.date >= start_date)
        
        macro = query.all()
        
        if not macro:
            return None
        
        df = pd.DataFrame([{
            'date': m.date,
            'vix': m.vix,
            'rate_10y': m.rate_10y,
            'rate_2y': m.rate_2y,
            'rate_spread': m.rate_spread,
            'dxy': m.dxy
        } for m in macro])
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        return df


def build_macro_features(macro_df):
    """Build macro features from raw macro data."""
    features = {}
    
    # VIX features
    if 'vix' in macro_df.columns:
        features['vix'] = macro_df['vix']
        features['vix_ma_20d'] = macro_df['vix'].rolling(20).mean()
        features['vix_ma_63d'] = macro_df['vix'].rolling(63).mean()
        features['vix_change_5d'] = macro_df['vix'].pct_change(5)
        features['vix_change_20d'] = macro_df['vix'].pct_change(20)
        features['vix_zscore_20d'] = (
            (macro_df['vix'] - macro_df['vix'].rolling(20).mean()) /
            (macro_df['vix'].rolling(20).std() + 1e-6)
        )
    
    # Interest rate features
    if 'rate_10y' in macro_df.columns:
        features['rate_10y'] = macro_df['rate_10y']
        features['rate_10y_change_5d'] = macro_df['rate_10y'].diff(5)
        features['rate_10y_change_20d'] = macro_df['rate_10y'].diff(20)
    
    if 'rate_2y' in macro_df.columns:
        features['rate_2y'] = macro_df['rate_2y']
        features['rate_2y_change_5d'] = macro_df['rate_2y'].diff(5)
    
    if 'rate_spread' in macro_df.columns:
        features['rate_spread'] = macro_df['rate_spread']
        features['rate_spread_change_5d'] = macro_df['rate_spread'].diff(5)
        # Inverted yield curve indicator
        features['yield_curve_inverted'] = (macro_df['rate_spread'] < 0).astype(int)
    
    # Dollar index features
    if 'dxy' in macro_df.columns:
        features['dxy'] = macro_df['dxy']
        features['dxy_change_5d'] = macro_df['dxy'].pct_change(5)
        features['dxy_change_20d'] = macro_df['dxy'].pct_change(20)
    
    # Regime indicators
    if 'vix' in macro_df.columns:
        # Risk-off indicator (VIX > 20)
        features['risk_off_regime'] = (macro_df['vix'] > 20).astype(int)
        # High volatility regime
        features['high_vol_regime'] = (macro_df['vix'] > 30).astype(int)
    
    # Combine into DataFrame
    if features:
        features_df = pd.DataFrame(features, index=macro_df.index)
        features_df['date'] = macro_df['date'].values
        return features_df
    else:
        return pd.DataFrame({'date': macro_df['date']})


def merge_macro_features_with_market_features(engine):
    """Merge macro features with existing market features."""
    macro_df = get_macro_data(engine)
    if macro_df is None or macro_df.empty:
        print("No macro data available.")
        return
    
    macro_features_df = build_macro_features(macro_df)
    
    # Get all tickers
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'tickers.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    tickers = config['tickers']
    
    with Session(engine) as session:
        for ticker in tickers:
            if ticker == 'SPY':
                continue
            
            # Get existing features for this ticker
            feature_records = session.query(FeatureDaily).filter(
                FeatureDaily.ticker == ticker
            ).order_by(FeatureDaily.date).all()
            
            if not feature_records:
                continue
            
            # Merge macro features with market features
            for feat_record in feature_records:
                if isinstance(feat_record.features, dict):
                    feat_dict = feat_record.features.copy()
                else:
                    feat_dict = {}
                
                # Add macro features for this date
                macro_row = macro_features_df[macro_features_df['date'] == pd.Timestamp(feat_record.date)]
                if not macro_row.empty:
                    for col in macro_features_df.columns:
                        if col != 'date' and col in macro_row.columns:
                            val = macro_row[col].iloc[0]
                            if pd.notna(val):
                                feat_dict[col] = float(val)
                    
                    feat_record.features = feat_dict
            
            session.commit()


def build_all_macro_features():
    """Build and merge macro features for all tickers."""
    init_database()
    engine = get_db_engine()
    
    print("Building macro features...")
    merge_macro_features_with_market_features(engine)
    print("Macro feature engineering complete!")


if __name__ == '__main__':
    build_all_macro_features()

