"""
Build media features from sentiment data.
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

from data.database import get_db_engine, SentimentDaily, FeatureDaily, init_database
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert


def load_feature_config():
    """Load feature engineering configuration."""
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config['features']['media']


def get_sentiment_data(engine, ticker):
    """Get sentiment data for a ticker."""
    with Session(engine) as session:
        sentiments = session.query(SentimentDaily).filter(
            SentimentDaily.ticker == ticker
        ).order_by(SentimentDaily.date).all()
        
        if not sentiments:
            return None
        
        df = pd.DataFrame([{
            'date': s.date,
            'n_articles': s.n_articles,
            'mean_pos': s.mean_pos,
            'mean_neg': s.mean_neg,
            'mean_neu': s.mean_neu,
            'mean_net': s.mean_net,
            'sentiment_change_5d': s.sentiment_change_5d,
            'sentiment_change_20d': s.sentiment_change_20d,
            'coverage_zscore': s.coverage_zscore,
            'volume_spike': s.volume_spike
        } for s in sentiments])
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        return df


def build_media_features(engine, ticker):
    """
    Build media features from sentiment data.
    
    Returns:
        DataFrame with media features
    """
    sentiment_df = get_sentiment_data(engine, ticker)
    
    if sentiment_df is None or sentiment_df.empty:
        return None
    
    features = {}
    
    # Basic sentiment features
    features['n_articles'] = sentiment_df['n_articles']
    features['mean_pos'] = sentiment_df['mean_pos']
    features['mean_neg'] = sentiment_df['mean_neg']
    features['mean_neu'] = sentiment_df['mean_neu']
    features['mean_net'] = sentiment_df['mean_net']
    
    # Sentiment changes
    features['sentiment_change_5d'] = sentiment_df['sentiment_change_5d']
    features['sentiment_change_20d'] = sentiment_df['sentiment_change_20d']
    
    # Coverage features
    features['coverage_zscore'] = sentiment_df['coverage_zscore']
    features['volume_spike'] = sentiment_df['volume_spike']
    
    # Rolling aggregates
    config = load_feature_config()
    windows = config.get('sentiment_windows', [5, 20])
    
    for window in windows:
        features[f'sentiment_ma_{window}d'] = sentiment_df['mean_net'].rolling(window).mean()
        features[f'sentiment_std_{window}d'] = sentiment_df['mean_net'].rolling(window).std()
        features[f'article_count_ma_{window}d'] = sentiment_df['n_articles'].rolling(window).mean()
    
    # Sentiment momentum
    features['sentiment_momentum_5d'] = sentiment_df['mean_net'].diff(5)
    features['sentiment_momentum_20d'] = sentiment_df['mean_net'].diff(20)
    
    # Positive/negative ratio
    features['pos_ratio'] = sentiment_df['mean_pos'] / (sentiment_df['mean_pos'] + sentiment_df['mean_neg'] + 1e-6)
    features['neg_ratio'] = sentiment_df['mean_neg'] / (sentiment_df['mean_pos'] + sentiment_df['mean_neg'] + 1e-6)
    
    # Coverage spikes (binary indicators)
    volume_spike_threshold = config.get('volume_spike_window', 20)
    features['coverage_spike'] = (sentiment_df['coverage_zscore'] > 2).astype(int)
    features['coverage_drop'] = (sentiment_df['coverage_zscore'] < -2).astype(int)
    
    # Combine into DataFrame
    features_df = pd.DataFrame(features, index=sentiment_df.index)
    features_df['ticker'] = ticker
    features_df['date'] = sentiment_df['date'].values
    
    # Reset index
    features_df = features_df.reset_index(drop=True)
    
    return features_df


def merge_with_market_features(engine, ticker, media_features_df):
    """
    Merge media features with existing market features.
    """
    from sqlalchemy import and_
    
    with Session(engine) as session:
        # Get existing market features
        feature_records = session.query(FeatureDaily).filter(
            FeatureDaily.ticker == ticker
        ).order_by(FeatureDaily.date).all()
        
        if not feature_records:
            # No market features yet, just store media features
            return media_features_df
        
        # Convert to DataFrame
        market_features_list = []
        for record in feature_records:
            feat_dict = record.features if isinstance(record.features, dict) else {}
            feat_dict['ticker'] = record.ticker
            feat_dict['date'] = record.date
            market_features_list.append(feat_dict)
        
        market_df = pd.DataFrame(market_features_list)
        market_df['date'] = pd.to_datetime(market_df['date'])
        
        # Merge
        merged = pd.merge(
            market_df,
            media_features_df,
            on=['ticker', 'date'],
            how='outer',
            suffixes=('', '_media')
        )
        
        # Combine feature dictionaries
        # For now, we'll update the features JSONB with media features
        return merged


def upsert_media_features(df, engine):
    """Upsert media features, merging with existing market features."""
    # Get existing features and merge
    ticker = df['ticker'].iloc[0]
    merged_df = merge_with_market_features(engine, ticker, df)
    
    # Extract media feature columns
    media_cols = [col for col in df.columns if col not in ['ticker', 'date']]
    
    # Update existing feature records
    with Session(engine) as session:
        for _, row in merged_df.iterrows():
            # Get or create feature record
            feature_record = session.query(FeatureDaily).filter(
                FeatureDaily.ticker == row['ticker'],
                FeatureDaily.date == row['date']
            ).first()
            
            if feature_record:
                # Update existing features dict
                if isinstance(feature_record.features, dict):
                    features_dict = feature_record.features.copy()
                else:
                    features_dict = {}
                
                # Add media features
                for col in media_cols:
                    if col in row and pd.notna(row[col]):
                        features_dict[col] = float(row[col])
                
                feature_record.features = features_dict
            else:
                # Create new record with just media features
                features_dict = {col: float(row[col]) for col in media_cols if pd.notna(row[col])}
                feature_record = FeatureDaily(
                    ticker=row['ticker'],
                    date=row['date'],
                    features=features_dict
                )
                session.add(feature_record)
        
        session.commit()


def build_all_media_features(tickers=None):
    """
    Build media features for all tickers.
    
    Args:
        tickers: List of tickers (None = all in database)
    """
    init_database()
    
    engine = get_db_engine()
    
    # Load tickers
    if tickers is None:
        with open('configs/tickers.yaml', 'r') as f:
            config = yaml.safe_load(f)
        tickers = config['tickers']
    
    print(f"Building media features for {len(tickers)} tickers...")
    
    for ticker in tqdm(tickers, desc="Building media features"):
        features_df = build_media_features(engine, ticker)
        
        if features_df is not None and not features_df.empty:
            print(f"  {ticker}: {len(features_df)} days, {len(features_df.columns)-2} features")
            upsert_media_features(features_df, engine)
        else:
            print(f"  {ticker}: No sentiment data available")
    
    print("Media feature engineering complete!")


if __name__ == '__main__':
    build_all_media_features()

