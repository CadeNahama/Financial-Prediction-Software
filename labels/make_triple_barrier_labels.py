"""
Create triple barrier labels using MlFinLab.
Triple barrier method: profit-take, stop-loss, or time-barrier (whichever hits first).
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

from data.database import get_db_engine, PriceDaily, Label, init_database

try:
    from mlfinlab.labeling import triple_barrier_events
    MLFINLAB_AVAILABLE = True
except ImportError:
    # mlfinlab is optional - we have a manual fallback implementation
    MLFINLAB_AVAILABLE = False


def load_labeling_config():
    """Load labeling configuration."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
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
            'open': p.open,
            'high': p.high,
            'low': p.low,
            'close': p.close,
            'adj_close': p.adj_close,
            'volume': p.volume
        } for p in prices])
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df.set_index('date', inplace=True)
        
        return df


def create_triple_barrier_labels(price_df, profit_take=0.05, stop_loss=0.03, time_barrier=63):
    """
    Create triple barrier labels.
    
    Args:
        price_df: DataFrame with OHLCV data, indexed by date
        profit_take: Profit target (e.g., 0.05 = 5%)
        stop_loss: Stop loss (e.g., 0.03 = 3%)
        time_barrier: Maximum holding period in days
    
    Returns:
        DataFrame with triple barrier labels
    """
    if not MLFINLAB_AVAILABLE:
        raise ImportError("MlFinLab required for triple barrier labeling")
    
    # Calculate volatility for dynamic barriers (optional - using fixed for now)
    # You could use volatility-adjusted barriers: pt_dynamic = volatility * profit_take
    
    # Create events (entry points) - every day is a potential entry
    events = pd.DataFrame(index=price_df.index)
    events['t1'] = events.index + pd.Timedelta(days=time_barrier)  # Time barrier
    events['trgt'] = 1.0  # Unit target (we'll scale by profit_take/stop_loss)
    
    # Generate triple barrier labels
    # Note: mlfinlab expects close prices and uses volatility for dynamic barriers
    try:
        labels = triple_barrier_events(
            close=price_df['adj_close'],
            t_events=events.index,
            pt_sl=[profit_take, stop_loss],  # [profit_take, stop_loss]
            trgt=events['trgt'],
            min_ret=0.0,
            num_threads=1,
            t1=events['t1']
        )
    except Exception as e:
        print(f"Error in triple_barrier_events: {e}")
        # Fallback: manual implementation
        labels = create_triple_barrier_manual(price_df, profit_take, stop_loss, time_barrier)
    
    # Convert labels to our format
    # Labels: 1 = profit taken, -1 = stop loss, 0 = time barrier hit
    result = pd.DataFrame(index=price_df.index)
    result['date'] = result.index
    result['triple_barrier_label'] = labels['bin'].values
    result['triple_barrier_outcome'] = labels['bin'].map({1: 'profit', -1: 'stop', 0: 'time'})
    result['triple_barrier_return'] = labels['ret'].values
    result['triple_barrier_time'] = labels['t1'] - labels.index
    
    return result


def create_triple_barrier_manual(price_df, profit_take=0.05, stop_loss=0.03, time_barrier=63):
    """
    Manual implementation of triple barrier labeling (fallback).
    """
    labels = pd.DataFrame(index=price_df.index)
    labels['bin'] = 0
    labels['ret'] = 0.0
    labels['t1'] = labels.index + pd.Timedelta(days=time_barrier)
    
    for i in range(len(price_df) - time_barrier):
        entry_idx = price_df.index[i]
        entry_price = price_df.loc[entry_idx, 'adj_close']
        
        # Look ahead
        window = price_df.iloc[i+1:i+1+time_barrier]
        
        if window.empty:
            continue
        
        # Check profit take
        profit_price = entry_price * (1 + profit_take)
        profit_hit = window[window['high'] >= profit_price]
        
        # Check stop loss
        stop_price = entry_price * (1 - stop_loss)
        stop_hit = window[window['low'] <= stop_price]
        
        # Determine outcome
        if not profit_hit.empty and not stop_hit.empty:
            # Both hit - whichever came first
            if profit_hit.index[0] <= stop_hit.index[0]:
                labels.loc[entry_idx, 'bin'] = 1  # Profit
                labels.loc[entry_idx, 'ret'] = profit_take
                labels.loc[entry_idx, 't1'] = profit_hit.index[0]
            else:
                labels.loc[entry_idx, 'bin'] = -1  # Stop
                labels.loc[entry_idx, 'ret'] = -stop_loss
                labels.loc[entry_idx, 't1'] = stop_hit.index[0]
        elif not profit_hit.empty:
            labels.loc[entry_idx, 'bin'] = 1  # Profit
            labels.loc[entry_idx, 'ret'] = profit_take
            labels.loc[entry_idx, 't1'] = profit_hit.index[0]
        elif not stop_hit.empty:
            labels.loc[entry_idx, 'bin'] = -1  # Stop
            labels.loc[entry_idx, 'ret'] = -stop_loss
            labels.loc[entry_idx, 't1'] = stop_hit.index[0]
        else:
            # Time barrier hit
            labels.loc[entry_idx, 'bin'] = 0  # Time
            final_price = window.iloc[-1]['adj_close']
            labels.loc[entry_idx, 'ret'] = (final_price - entry_price) / entry_price
            labels.loc[entry_idx, 't1'] = window.index[-1]
    
    return labels


def create_triple_barrier_for_ticker(engine, ticker):
    """Create triple barrier labels for a single ticker."""
    price_df = get_price_data(engine, ticker)
    if price_df is None or price_df.empty:
        return None
    
    config = load_labeling_config()
    tb_config = config.get('triple_barrier', {})
    profit_take = tb_config.get('profit_take', 0.05)
    stop_loss = tb_config.get('stop_loss', 0.03)
    time_barrier = tb_config.get('time_barrier', 63)
    
    labels_df = create_triple_barrier_labels(price_df, profit_take, stop_loss, time_barrier)
    labels_df['ticker'] = ticker
    labels_df.reset_index(drop=True, inplace=True)
    
    return labels_df


def upsert_triple_barrier_labels(df, engine):
    """Update existing labels with triple barrier data."""
    with Session(engine) as session:
        for _, row in df.iterrows():
            # Update existing label record
            label = session.query(Label).filter(
                Label.ticker == row['ticker'],
                Label.date == row['date']
            ).first()
            
            if label:
                label.triple_barrier_label = int(row['triple_barrier_label']) if pd.notna(row['triple_barrier_label']) else None
                label.triple_barrier_outcome = str(row['triple_barrier_outcome']) if pd.notna(row['triple_barrier_outcome']) else None
            else:
                # Create new label record (shouldn't happen if forward returns exist)
                label = Label(
                    ticker=row['ticker'],
                    date=row['date'],
                    triple_barrier_label=int(row['triple_barrier_label']) if pd.notna(row['triple_barrier_label']) else None,
                    triple_barrier_outcome=str(row['triple_barrier_outcome']) if pd.notna(row['triple_barrier_outcome']) else None
                )
                session.add(label)
        
        session.commit()


def create_all_triple_barrier_labels(tickers=None):
    """Create triple barrier labels for all tickers."""
    if not MLFINLAB_AVAILABLE:
        print("MlFinLab not available. Skipping triple barrier labeling.")
        return
    
    init_database()
    engine = get_db_engine()
    
    if tickers is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'tickers.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        tickers = config['tickers']
    
    print(f"Creating triple barrier labels for {len(tickers)} tickers...")
    
    for ticker in tqdm(tickers, desc="Creating triple barrier labels"):
        if ticker == 'SPY':  # Skip benchmark
            continue
        
        labels_df = create_triple_barrier_for_ticker(engine, ticker)
        if labels_df is not None and not labels_df.empty:
            print(f"  {ticker}: {len(labels_df)} labeled dates")
            upsert_triple_barrier_labels(labels_df, engine)
        else:
            print(f"  {ticker}: No data available")
    
    print("Triple barrier labeling complete!")


if __name__ == '__main__':
    create_all_triple_barrier_labels()

