"""
Ingest news articles from GDELT Doc API.
"""
import os
import pandas as pd
from sqlalchemy.orm import Session
from datetime import datetime, date, timedelta
import yaml
from tqdm import tqdm
from data.database import get_db_engine, GDELTArticle, init_database
from sqlalchemy import create_engine
import time

try:
    from gdeltdoc import GdeltDoc, Filters
except ImportError:
    print("Warning: gdeltdoc not installed. Install with: pip install gdeltdoc")
    GdeltDoc = None
    Filters = None


def load_ticker_aliases():
    """Load ticker to search term mappings."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'tickers.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('ticker_aliases', {})


def load_data_config():
    """Load GDELT configuration."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['data_sources']['gdelt']


def get_market_date(article_datetime):
    """
    Convert article datetime to market close date.
    Articles during day t are usable at end of day t (or next open).
    """
    if isinstance(article_datetime, str):
        article_datetime = datetime.fromisoformat(article_datetime.replace('Z', '+00:00'))
    
    # If article is before 4 PM ET (market close), assign to that day
    # Otherwise, assign to next trading day
    # For simplicity, we'll assign to the date of the article
    # and handle market alignment in feature engineering
    return article_datetime.date()


def query_gdelt(ticker, aliases, start_date, end_date):
    """
    Query GDELT Doc API for articles about a ticker.
    
    Args:
        ticker: Stock ticker
        aliases: List of search terms
        start_date: Start date
        end_date: End date
    
    Returns:
        DataFrame with articles
    """
    if GdeltDoc is None:
        print("gdeltdoc library not available. Skipping GDELT ingestion.")
        return None
    
    try:
        # Combine aliases into search query
        # GDELT Doc API uses OR logic for multiple terms
        search_terms = ' OR '.join([f'"{alias}"' for alias in aliases])
        
        f = Filters(
            keyword=search_terms,
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d')
        )
        
        gd = GdeltDoc()
        articles = gd.article_search(f)
        
        if not articles or len(articles) == 0:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(articles)
        
        # Standardize column names
        df['ticker'] = ticker
        
        # Parse datetime
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        elif 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])
        else:
            print(f"Warning: No datetime column found for {ticker}")
            return None
        
        # Add market date
        df['date'] = df['datetime'].apply(get_market_date)
        
        # Standardize column names
        column_mapping = {
            'title': 'title',
            'url': 'url',
            'source': 'source',
            'language': 'language',
            'country': 'country'
        }
        
        for col in column_mapping:
            if col not in df.columns:
                df[col] = None
        
        # Store additional GDELT fields as JSON
        gdelt_fields = {}
        for col in df.columns:
            if col not in ['ticker', 'datetime', 'date', 'title', 'url', 'source', 'language', 'country']:
                gdelt_fields[col] = df[col].tolist()
        
        # Select and rename columns
        df = df[['ticker', 'datetime', 'date', 'title', 'url', 'source', 'language', 'country']].copy()
        
        # Add gdelt_fields as a dict per row (will be converted to JSONB)
        df['gdelt_fields'] = [{} for _ in range(len(df))]
        
        return df
    
    except Exception as e:
        print(f"Error querying GDELT for {ticker}: {e}")
        return None


def upsert_articles(df, engine):
    """Upsert articles into database."""
    dialect = engine.dialect.name
    
    if dialect == 'postgresql':
        from sqlalchemy.dialects.postgresql import insert
        # For PostgreSQL, convert datetime to string for JSONB compatibility
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            # Convert datetime to string for storage
            if 'datetime' in row_dict and pd.notna(row_dict['datetime']):
                row_dict['datetime'] = row_dict['datetime'].isoformat() if isinstance(row_dict['datetime'], (datetime, pd.Timestamp)) else row_dict['datetime']
            
            stmt = insert(GDELTArticle).values(**row_dict)
            # Simple insert (allow duplicates for now, can dedupe later)
            with engine.connect() as conn:
                conn.execute(stmt)
                conn.commit()
    else:
        # SQLite
        with Session(engine) as session:
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                if 'datetime' in row_dict and pd.notna(row_dict['datetime']):
                    if isinstance(row_dict['datetime'], str):
                        row_dict['datetime'] = datetime.fromisoformat(row_dict['datetime'])
                
                article = GDELTArticle(**row_dict)
                session.add(article)
            
            session.commit()


def ingest_gdelt(tickers=None, lookback_days=None):
    """
    Main GDELT ingestion function.
    
    Args:
        tickers: List of tickers (None = load from config)
        lookback_days: Days to look back (None = load from config)
    """
    init_database()
    
    if GdeltDoc is None:
        print("GDELT ingestion requires gdeltdoc library.")
        print("Install with: pip install gdeltdoc")
        return
    
    # Load configuration
    ticker_aliases = load_ticker_aliases()
    
    if tickers is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'tickers.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        tickers = config['tickers']
    
    gdelt_config = load_data_config()
    if lookback_days is None:
        lookback_days = gdelt_config.get('lookback_days', 90)
    
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)
    
    engine = get_db_engine()
    
    print(f"Ingesting GDELT articles for {len(tickers)} tickers...")
    print(f"Date range: {start_date} to {end_date}")
    
    for ticker in tqdm(tickers, desc="Fetching GDELT"):
        if ticker not in ticker_aliases:
            print(f"Warning: No aliases found for {ticker}, skipping")
            continue
        
        aliases = ticker_aliases[ticker]
        df = query_gdelt(ticker, aliases, start_date, end_date)
        
        if df is not None and not df.empty:
            print(f"  {ticker}: {len(df)} articles")
            upsert_articles(df, engine)
            # Rate limiting
            time.sleep(1)
        else:
            print(f"  {ticker}: No articles found")
    
    print("GDELT ingestion complete!")


if __name__ == '__main__':
    ingest_gdelt()

