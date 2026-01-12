"""
Merge market and media features into a single feature matrix.
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
from sqlalchemy import create_engine


def load_all_features(engine, ticker, start_date=None, end_date=None):
    """
    Load all features (market + media) for a ticker.
    
    Returns:
        DataFrame with all features as columns
    """
    with Session(engine) as session:
        query = session.query(FeatureDaily).filter(FeatureDaily.ticker == ticker)
        
        if start_date:
            query = query.filter(FeatureDaily.date >= start_date)
        if end_date:
            query = query.filter(FeatureDaily.date <= end_date)
        
        features = query.order_by(FeatureDaily.date).all()
        
        if not features:
            return None
        
        # Convert JSONB features to flat DataFrame
        records = []
        for feat in features:
            record = {'ticker': feat.ticker, 'date': feat.date}
            if isinstance(feat.features, dict):
                record.update(feat.features)
            records.append(record)
        
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        return df


def merge_features_for_ticker(engine, ticker):
    """Merge features for a single ticker (already done in database, this is for export)."""
    return load_all_features(engine, ticker)


def merge_features_for_all_tickers(tickers=None, output_path=None):
    """
    Merge features for all tickers and optionally save to file.
    
    Args:
        tickers: List of tickers (None = all in database)
        output_path: Optional path to save merged features (CSV or Parquet)
    """
    init_database()
    
    engine = get_db_engine()
    
    # Load tickers
    if tickers is None:
        with open('configs/tickers.yaml', 'r') as f:
            config = yaml.safe_load(f)
        tickers = config['tickers']
    
    print(f"Merging features for {len(tickers)} tickers...")
    
    all_features = []
    
    for ticker in tqdm(tickers, desc="Merging features"):
        df = merge_features_for_ticker(engine, ticker)
        if df is not None and not df.empty:
            all_features.append(df)
            print(f"  {ticker}: {len(df)} days, {len(df.columns)-2} features")
    
    if not all_features:
        print("No features found.")
        return None
    
    # Combine all tickers
    combined = pd.concat(all_features, ignore_index=True)
    
    print(f"\nTotal: {len(combined)} records, {len(combined.columns)-2} features")
    
    # Save if requested
    if output_path:
        if output_path.endswith('.parquet'):
            combined.to_parquet(output_path, index=False)
        else:
            combined.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
    
    return combined


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge features for all tickers')
    parser.add_argument('--output', '-o', help='Output file path (CSV or Parquet)')
    args = parser.parse_args()
    
    merge_features_for_all_tickers(output_path=args.output)

