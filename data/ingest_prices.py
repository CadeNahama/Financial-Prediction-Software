"""
Ingest daily price data from yfinance.
"""
import os
import yfinance as yf
import pandas as pd
from sqlalchemy.orm import Session
from datetime import datetime, date
import yaml
from tqdm import tqdm
from data.database import get_db_engine, PriceDaily, init_database
from sqlalchemy import create_engine


def load_tickers():
    """Load ticker list from config."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'tickers.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['tickers']


def load_data_config():
    """Load data source configuration."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['data_sources']['prices']


def fetch_price_data(ticker, start_date, end_date=None):
    """
    Fetch price data for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD) or None for today
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"Warning: No data for {ticker}")
            return None
        
        # Reset index to get date as column
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        
        # Rename columns to match database schema
        df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        
        # Add adj_close (use Close if not available)
        if 'Dividends' in df.columns:
            df['adj_close'] = df['close']  # yfinance already adjusts
        else:
            df['adj_close'] = df['close']
        
        df['ticker'] = ticker
        
        # Select only needed columns
        df = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]
        
        return df
    
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


def upsert_prices(df, engine):
    """
    Upsert price data into database.
    Uses PostgreSQL upsert or SQLite replace.
    """
    # Convert date to datetime for SQLAlchemy
    df_to_insert = df.copy()
    
    # Use pandas to_sql with if_exists='append' and handle duplicates
    # For better performance, we'll use raw SQL upsert for PostgreSQL
    # or check for SQLite
    
    dialect = engine.dialect.name
    
    if dialect == 'postgresql':
        # PostgreSQL upsert
        from sqlalchemy.dialects.postgresql import insert
        for _, row in df_to_insert.iterrows():
            stmt = insert(PriceDaily).values(**row.to_dict())
            stmt = stmt.on_conflict_do_update(
                index_elements=['ticker', 'date'],
                set_=dict(
                    open=stmt.excluded.open,
                    high=stmt.excluded.high,
                    low=stmt.excluded.low,
                    close=stmt.excluded.close,
                    adj_close=stmt.excluded.adj_close,
                    volume=stmt.excluded.volume
                )
            )
            with engine.connect() as conn:
                conn.execute(stmt)
                conn.commit()
    else:
        # SQLite: delete existing and insert
        with Session(engine) as session:
            for _, row in df_to_insert.iterrows():
                # Delete existing
                session.query(PriceDaily).filter(
                    PriceDaily.ticker == row['ticker'],
                    PriceDaily.date == row['date']
                ).delete()
                
                # Insert new
                price_record = PriceDaily(**row.to_dict())
                session.add(price_record)
            
            session.commit()


def ingest_prices(tickers=None, start_date=None, end_date=None):
    """
    Main ingestion function.
    
    Args:
        tickers: List of tickers (None = load from config)
        start_date: Start date (None = load from config)
        end_date: End date (None = today)
    """
    # Initialize database if needed
    init_database()
    
    # Load configuration
    if tickers is None:
        tickers = load_tickers()
    
    data_config = load_data_config()
    if start_date is None:
        start_date = data_config.get('start_date', '2010-01-01')
    if end_date is None:
        end_date = data_config.get('end_date', None)
    
    engine = get_db_engine()
    
    print(f"Ingesting price data for {len(tickers)} tickers...")
    print(f"Date range: {start_date} to {end_date or 'today'}")
    
    all_data = []
    
    for ticker in tqdm(tickers, desc="Fetching prices"):
        df = fetch_price_data(ticker, start_date, end_date)
        if df is not None and not df.empty:
            all_data.append(df)
    
    if not all_data:
        print("No data fetched. Exiting.")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"\nFetched {len(combined_df)} records")
    print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    
    # Upsert to database
    print("\nWriting to database...")
    upsert_prices(combined_df, engine)
    
    print("Price ingestion complete!")


if __name__ == '__main__':
    ingest_prices()

