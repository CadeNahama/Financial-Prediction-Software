"""
Build market features from price data.
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

from data.database import get_db_engine, PriceDaily, FeatureDaily, init_database
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert


def load_feature_config():
    """Load feature engineering configuration."""
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config['features']['market']


def get_price_data(engine, ticker, start_date=None):
    """Get price data for a ticker."""
    with Session(engine) as session:
        query = session.query(PriceDaily).filter(PriceDaily.ticker == ticker)
        if start_date:
            query = query.filter(PriceDaily.date >= start_date)
        
        prices = query.order_by(PriceDaily.date).all()
        
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
        
        return df


def get_benchmark_data(engine, benchmark_ticker='SPY'):
    """Get benchmark price data."""
    return get_price_data(engine, benchmark_ticker)


def calculate_momentum_features(df, windows=[20, 63, 126]):
    """Calculate momentum features."""
    features = {}
    
    for window in windows:
        # Returns
        features[f'return_{window}d'] = df['adj_close'].pct_change(window)
        
        # Log returns
        features[f'log_return_{window}d'] = np.log(df['adj_close'] / df['adj_close'].shift(window))
    
    return pd.DataFrame(features, index=df.index)


def calculate_trend_features(df, windows=[20, 63, 200]):
    """Calculate trend features."""
    features = {}
    
    for window in windows:
        ma = df['adj_close'].rolling(window).mean()
        features[f'ma_{window}d'] = ma
        features[f'price_vs_ma_{window}d'] = (df['adj_close'] - ma) / ma
        features[f'ma_slope_{window}d'] = ma.diff(5) / ma.shift(5)  # 5-day slope
    
    # MA crossovers
    if 20 in windows and 63 in windows:
        ma20 = df['adj_close'].rolling(20).mean()
        ma63 = df['adj_close'].rolling(63).mean()
        features['ma_cross_20_63'] = (ma20 - ma63) / ma63
        features['ma_cross_signal'] = (ma20 > ma63).astype(int)
    
    return pd.DataFrame(features, index=df.index)


def calculate_volatility_features(df, windows=[20, 63]):
    """Calculate volatility features."""
    features = {}
    
    returns = df['adj_close'].pct_change()
    
    for window in windows:
        # Realized volatility (annualized)
        vol = returns.rolling(window).std() * np.sqrt(252)
        features[f'volatility_{window}d'] = vol
        
        # High-low range
        hl_range = (df['high'] - df['low']) / df['adj_close']
        features[f'hl_range_{window}d'] = hl_range.rolling(window).mean()
    
    return pd.DataFrame(features, index=df.index)


def calculate_drawdown_features(df, window=252):
    """Calculate drawdown features."""
    features = {}
    
    # Rolling max
    rolling_max = df['adj_close'].rolling(window, min_periods=1).max()
    
    # Drawdown
    drawdown = (df['adj_close'] - rolling_max) / rolling_max
    features['drawdown'] = drawdown
    
    # Max drawdown in window
    features['max_drawdown_252d'] = drawdown.rolling(window).min()
    
    # Days since peak
    peak_idx = df['adj_close'].rolling(window).apply(lambda x: x.idxmax() if len(x) > 0 else np.nan, raw=False)
    features['days_since_peak'] = df.index - peak_idx
    features['days_since_peak'] = features['days_since_peak'].fillna(0)
    
    return pd.DataFrame(features, index=df.index)


def calculate_relative_strength_features(df, benchmark_df, windows=[20, 63, 126]):
    """Calculate relative strength vs benchmark."""
    features = {}
    
    if benchmark_df is None or benchmark_df.empty:
        # Return zeros if no benchmark
        for window in windows:
            features[f'relative_return_{window}d'] = 0.0
        return pd.DataFrame(features, index=df.index)
    
    # Align dates
    merged = pd.merge(
        df[['date', 'adj_close']],
        benchmark_df[['date', 'adj_close']],
        on='date',
        suffixes=('', '_bench')
    ).sort_values('date')
    
    merged['ticker_return'] = merged['adj_close'].pct_change()
    merged['bench_return'] = merged['adj_close_bench'].pct_change()
    merged['excess_return'] = merged['ticker_return'] - merged['bench_return']
    
    for window in windows:
        features[f'relative_return_{window}d'] = merged['excess_return'].rolling(window).sum()
    
    # Align back to original index
    result = pd.DataFrame(features, index=merged.index)
    result = result.reindex(df.index, method='ffill').fillna(0)
    
    return result


def calculate_liquidity_features(df, windows=[20, 63]):
    """Calculate liquidity features."""
    features = {}
    
    # Dollar volume
    df['dollar_volume'] = df['adj_close'] * df['volume']
    
    for window in windows:
        features[f'avg_dollar_volume_{window}d'] = df['dollar_volume'].rolling(window).mean()
        features[f'volume_change_{window}d'] = df['volume'].pct_change(window)
        features[f'volume_ratio_{window}d'] = df['volume'] / df['volume'].rolling(window).mean()
    
    return pd.DataFrame(features, index=df.index)


def build_market_features(engine, ticker, benchmark_ticker='SPY'):
    """
    Build all market features for a ticker.
    
    Returns:
        DataFrame with features
    """
    # Get price data
    price_df = get_price_data(engine, ticker)
    if price_df is None or price_df.empty:
        return None
    
    # Get benchmark data
    benchmark_df = get_benchmark_data(engine, benchmark_ticker)
    
    # Load config
    config = load_feature_config()
    momentum_windows = config.get('momentum_windows', [20, 63, 126])
    volatility_windows = config.get('volatility_windows', [20, 63])
    ma_windows = config.get('ma_windows', [20, 63, 200])
    
    # Calculate all feature groups
    momentum = calculate_momentum_features(price_df, momentum_windows)
    trend = calculate_trend_features(price_df, ma_windows)
    volatility = calculate_volatility_features(price_df, volatility_windows)
    drawdown = calculate_drawdown_features(price_df)
    relative_strength = calculate_relative_strength_features(price_df, benchmark_df, momentum_windows)
    liquidity = calculate_liquidity_features(price_df)
    
    # Combine all features
    all_features = pd.concat([
        momentum,
        trend,
        volatility,
        drawdown,
        relative_strength,
        liquidity
    ], axis=1)
    
    # Add ticker and date
    all_features['ticker'] = ticker
    all_features['date'] = price_df['date'].values
    
    # Reset index
    all_features = all_features.reset_index(drop=True)
    
    # Convert to dict for JSONB storage (or keep as separate columns)
    # For now, we'll store as separate columns in a flattened structure
    # In production, you might want to use JSONB for flexibility
    
    return all_features


def upsert_features(df, engine):
    """Upsert features to database."""
    # Convert features to dict format for JSONB
    feature_cols = [col for col in df.columns if col not in ['ticker', 'date']]
    
    records = []
    for _, row in df.iterrows():
        feature_dict = {col: float(row[col]) if pd.notna(row[col]) else None for col in feature_cols}
        records.append({
            'ticker': row['ticker'],
            'date': row['date'],
            'features': feature_dict
        })
    
    dialect = engine.dialect.name
    
    if dialect == 'postgresql':
        from sqlalchemy.dialects.postgresql import insert
        for record in records:
            stmt = insert(FeatureDaily).values(**record)
            stmt = stmt.on_conflict_do_update(
                index_elements=['ticker', 'date'],
                set_={'features': stmt.excluded.features}
            )
            engine.execute(stmt)
    else:
        with Session(engine) as session:
            for record in records:
                # Delete existing
                session.query(FeatureDaily).filter(
                    FeatureDaily.ticker == record['ticker'],
                    FeatureDaily.date == record['date']
                ).delete()
                
                feature = FeatureDaily(**record)
                session.add(feature)
            
            session.commit()


def build_all_market_features(tickers=None):
    """
    Build market features for all tickers.
    
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
    
    config = load_feature_config()
    benchmark = config.get('relative_strength_benchmark', 'SPY')
    
    print(f"Building market features for {len(tickers)} tickers...")
    print(f"Benchmark: {benchmark}")
    
    for ticker in tqdm(tickers, desc="Building features"):
        if ticker == benchmark:
            continue  # Skip benchmark itself
        
        features_df = build_market_features(engine, ticker, benchmark)
        
        if features_df is not None and not features_df.empty:
            print(f"  {ticker}: {len(features_df)} days, {len(features_df.columns)-2} features")
            upsert_features(features_df, engine)
        else:
            print(f"  {ticker}: No data available")
    
    print("Market feature engineering complete!")


if __name__ == '__main__':
    build_all_market_features()

