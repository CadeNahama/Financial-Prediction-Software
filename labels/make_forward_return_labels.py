"""
Create forward return labels for 3-4 month horizon.
"""
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from datetime import date, timedelta
import yaml
from tqdm import tqdm
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import get_db_engine, PriceDaily, Label, init_database
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert


def load_labeling_config():
    """Load labeling configuration."""
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config['labeling']


def get_price_data(engine, ticker):
    """Get price data for a ticker."""
    with Session(engine) as session:
        prices = session.query(PriceDaily).filter(
            PriceDaily.ticker == ticker
        ).order_by(PriceDaily.date).all()
        
        if not prices:
            return None
        
        df = pd.DataFrame([{
            'date': p.date,
            'adj_close': p.adj_close
        } for p in prices])
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        return df


def calculate_forward_returns(price_df, horizons=[63, 84], benchmark_df=None):
    """
    Calculate forward returns and labels.
    
    Args:
        price_df: DataFrame with date and adj_close
        horizons: List of forward horizons in trading days
        benchmark_df: Optional benchmark price data
    
    Returns:
        DataFrame with forward returns and labels
    """
    result = price_df[['date']].copy()
    result['ticker'] = price_df.get('ticker', 'UNKNOWN').iloc[0] if 'ticker' in price_df.columns else 'UNKNOWN'
    
    # Calculate forward returns for each horizon
    for horizon in horizons:
        # Forward return
        forward_price = price_df['adj_close'].shift(-horizon)
        forward_return = (forward_price - price_df['adj_close']) / price_df['adj_close']
        result[f'forward_return_{horizon}d'] = forward_return
        
        # Binary label (1 if positive return, 0 otherwise)
        result[f'label_horizon_{horizon}d'] = (forward_return > 0).astype(int)
    
    # Benchmark returns if provided
    if benchmark_df is not None and not benchmark_df.empty:
        # Merge on date
        merged = pd.merge(
            price_df[['date', 'adj_close']],
            benchmark_df[['date', 'adj_close']],
            on='date',
            suffixes=('', '_bench')
        ).sort_values('date')
        
        for horizon in horizons:
            # Benchmark forward return
            bench_forward_price = merged['adj_close_bench'].shift(-horizon)
            bench_forward_return = (bench_forward_price - merged['adj_close_bench']) / merged['adj_close_bench']
            
            # Relative return (ticker vs benchmark)
            ticker_return = result[f'forward_return_{horizon}d'].values
            relative_return = ticker_return - bench_forward_return.values
            
            # Label: 1 if beats benchmark, 0 otherwise
            result[f'benchmark_return_{horizon}d'] = bench_forward_return.values
            result[f'label_horizon_{horizon}d'] = (relative_return > 0).astype(int)
    
    return result


def create_labels(engine, ticker, benchmark_ticker='SPY'):
    """
    Create labels for a ticker.
    
    Returns:
        DataFrame with labels
    """
    # Get price data
    price_df = get_price_data(engine, ticker)
    if price_df is None or price_df.empty:
        return None
    
    price_df['ticker'] = ticker
    
    # Get benchmark data
    benchmark_df = get_price_data(engine, benchmark_ticker)
    
    # Load config
    config = load_labeling_config()
    horizons = config.get('horizons', [63, 84])
    use_benchmark = config.get('benchmark') is not None
    
    if not use_benchmark:
        benchmark_df = None
    
    # Calculate labels
    labels_df = calculate_forward_returns(price_df, horizons, benchmark_df)
    
    # Standardize column names
    column_mapping = {}
    for horizon in horizons:
        column_mapping[f'forward_return_{horizon}d'] = f'forward_return_{horizon}d'
        column_mapping[f'label_horizon_{horizon}d'] = f'label_horizon_{horizon}d'
        if benchmark_df is not None:
            column_mapping[f'benchmark_return_{horizon}d'] = f'benchmark_return_{horizon}d'
    
    # Select and rename columns for database
    result_cols = ['ticker', 'date']
    for horizon in horizons:
        result_cols.append(f'forward_return_{horizon}d')
        result_cols.append(f'label_horizon_{horizon}d')
        if benchmark_df is not None:
            result_cols.append(f'benchmark_return_{horizon}d')
    
    labels_df = labels_df[result_cols].copy()
    
    # Add null columns for triple barrier (if not using)
    labels_df['triple_barrier_label'] = None
    labels_df['triple_barrier_outcome'] = None
    
    return labels_df


def upsert_labels(df, engine):
    """Upsert labels to database."""
    dialect = engine.dialect.name
    
    if dialect == 'postgresql':
        from sqlalchemy.dialects.postgresql import insert
        for _, row in df.iterrows():
            # Convert NaN to None
            row_dict = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
            
            stmt = insert(Label).values(**row_dict)
            stmt = stmt.on_conflict_do_update(
                index_elements=['ticker', 'date'],
                set_=dict(
                    label_horizon_63d=stmt.excluded.label_horizon_63d,
                    label_horizon_84d=stmt.excluded.label_horizon_84d,
                    forward_return_63d=stmt.excluded.forward_return_63d,
                    forward_return_84d=stmt.excluded.forward_return_84d,
                    benchmark_return_63d=stmt.excluded.benchmark_return_63d,
                    benchmark_return_84d=stmt.excluded.benchmark_return_84d
                )
            )
            engine.execute(stmt)
    else:
        with Session(engine) as session:
            for _, row in df.iterrows():
                row_dict = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
                
                # Delete existing
                session.query(Label).filter(
                    Label.ticker == row_dict['ticker'],
                    Label.date == row_dict['date']
                ).delete()
                
                label = Label(**row_dict)
                session.add(label)
            
            session.commit()


def create_all_labels(tickers=None):
    """
    Create labels for all tickers.
    
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
    
    config = load_labeling_config()
    benchmark = config.get('benchmark', 'SPY')
    
    print(f"Creating labels for {len(tickers)} tickers...")
    print(f"Benchmark: {benchmark}")
    
    for ticker in tqdm(tickers, desc="Creating labels"):
        if ticker == benchmark:
            continue  # Skip benchmark
        
        labels_df = create_labels(engine, ticker, benchmark)
        
        if labels_df is not None and not labels_df.empty:
            # Remove rows with NaN labels (not enough forward data)
            labels_df = labels_df.dropna(subset=['label_horizon_63d'])
            print(f"  {ticker}: {len(labels_df)} labeled dates")
            upsert_labels(labels_df, engine)
        else:
            print(f"  {ticker}: No data available")
    
    print("Label creation complete!")


if __name__ == '__main__':
    create_all_labels()

