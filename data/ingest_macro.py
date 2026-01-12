"""
Ingest macro economic data (VIX, interest rates, etc.)
"""
import yfinance as yf
import pandas as pd
from sqlalchemy.orm import Session
from datetime import datetime, date
import yaml
from tqdm import tqdm
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import get_db_engine, init_database, Base
from sqlalchemy import Column, Integer, Float, Date, Index


class MacroDaily(Base):
    """Daily macro economic data."""
    __tablename__ = 'macro_daily'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    vix = Column(Float)  # VIX (volatility index)
    rate_10y = Column(Float)  # 10-year Treasury rate
    rate_2y = Column(Float)  # 2-year Treasury rate
    rate_spread = Column(Float)  # 10Y - 2Y spread
    dxy = Column(Float)  # Dollar index
    
    __table_args__ = (
        Index('idx_macro_date', 'date', unique=True),
    )


def fetch_vix(start_date, end_date=None):
    """Fetch VIX data from yfinance."""
    try:
        vix = yf.Ticker("^VIX")
        df = vix.history(start=start_date, end=end_date)
        if df.empty:
            return None
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df = df.rename(columns={'Date': 'date', 'Close': 'vix'})
        return df[['date', 'vix']]
    except Exception as e:
        print(f"Error fetching VIX: {e}")
        return None


def fetch_treasury_rates(start_date, end_date=None):
    """Fetch Treasury rates (10Y and 2Y) from yfinance."""
    try:
        # 10-year Treasury
        tnx = yf.Ticker("^TNX")
        df_10y = tnx.history(start=start_date, end=end_date)
        
        # 2-year Treasury
        irx = yf.Ticker("^IRX")
        df_2y = irx.history(start=start_date, end=end_date)
        
        if df_10y.empty or df_2y.empty:
            return None
        
        df_10y.reset_index(inplace=True)
        df_2y.reset_index(inplace=True)
        df_10y['Date'] = pd.to_datetime(df_10y['Date']).dt.date
        df_2y['Date'] = pd.to_datetime(df_2y['Date']).dt.date
        
        merged = pd.merge(
            df_10y[['Date', 'Close']],
            df_2y[['Date', 'Close']],
            on='Date',
            suffixes=('_10y', '_2y')
        )
        merged = merged.rename(columns={'Date': 'date', 'Close_10y': 'rate_10y', 'Close_2y': 'rate_2y'})
        merged['rate_spread'] = merged['rate_10y'] - merged['rate_2y']
        
        return merged[['date', 'rate_10y', 'rate_2y', 'rate_spread']]
    except Exception as e:
        print(f"Error fetching Treasury rates: {e}")
        return None


def fetch_dxy(start_date, end_date=None):
    """Fetch Dollar Index from yfinance."""
    try:
        dxy = yf.Ticker("DX-Y.NYB")  # Dollar Index
        df = dxy.history(start=start_date, end=end_date)
        if df.empty:
            # Try alternative ticker
            dxy = yf.Ticker("UUP")  # Dollar ETF as proxy
            df = dxy.history(start=start_date, end=end_date)
        if df.empty:
            return None
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df = df.rename(columns={'Date': 'date', 'Close': 'dxy'})
        return df[['date', 'dxy']]
    except Exception as e:
        print(f"Error fetching DXY: {e}")
        return None


def ingest_macro_data(start_date='2010-01-01', end_date=None):
    """Ingest all macro data."""
    init_database()
    
    # Create macro_daily table if it doesn't exist
    engine = get_db_engine()
    Base.metadata.create_all(engine)
    
    print("Fetching macro data...")
    
    # Fetch VIX
    print("  Fetching VIX...")
    vix_df = fetch_vix(start_date, end_date)
    
    # Fetch Treasury rates
    print("  Fetching Treasury rates...")
    rates_df = fetch_treasury_rates(start_date, end_date)
    
    # Fetch DXY
    print("  Fetching Dollar Index...")
    dxy_df = fetch_dxy(start_date, end_date)
    
    # Merge all data
    macro_df = pd.DataFrame({'date': pd.date_range(start=start_date, end=end_date or date.today(), freq='D')})
    macro_df['date'] = pd.to_datetime(macro_df['date']).dt.date
    
    if vix_df is not None:
        macro_df = pd.merge(macro_df, vix_df, on='date', how='left')
    
    if rates_df is not None:
        macro_df = pd.merge(macro_df, rates_df, on='date', how='left')
    
    if dxy_df is not None:
        macro_df = pd.merge(macro_df, dxy_df, on='date', how='left')
    
    # Keep only dates with at least one value
    macro_df = macro_df[macro_df[['vix', 'rate_10y', 'dxy']].notna().any(axis=1)]
    
    # Upsert to database
    print(f"Writing {len(macro_df)} macro data records to database...")
    with Session(engine) as session:
        for _, row in macro_df.iterrows():
            existing = session.query(MacroDaily).filter(MacroDaily.date == row['date']).first()
            if existing:
                for col in ['vix', 'rate_10y', 'rate_2y', 'rate_spread', 'dxy']:
                    if col in row and pd.notna(row[col]):
                        setattr(existing, col, row[col])
            else:
                macro = MacroDaily(**{k: v for k, v in row.to_dict().items() if pd.notna(v)})
                session.add(macro)
        session.commit()
    
    print("Macro data ingestion complete!")


if __name__ == '__main__':
    ingest_macro_data()
