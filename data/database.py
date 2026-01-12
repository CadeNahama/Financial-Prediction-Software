"""
Database connection and schema setup.
"""
import os
from sqlalchemy import create_engine, Column, String, Float, Integer, Date, DateTime, Text, Index, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import yaml
from dotenv import load_dotenv

load_dotenv()

# Determine JSON type based on database config
def get_json_type():
    """Get appropriate JSON type based on database configuration."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        db_type = config.get('database', {}).get('type', 'sqlite')
        
        if db_type == 'postgresql':
            from sqlalchemy.dialects.postgresql import JSONB
            return JSONB
        else:
            return JSON
    except:
        # Default to JSON (SQLite compatible)
        return JSON

JSON_TYPE = get_json_type()

Base = declarative_base()


class PriceDaily(Base):
    """Daily OHLCV price data."""
    __tablename__ = 'prices_daily'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Integer)
    
    __table_args__ = (
        Index('idx_ticker_date', 'ticker', 'date', unique=True),
    )


class FeatureDaily(Base):
    """Daily engineered features."""
    __tablename__ = 'features_daily'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    
    # Market features (will be populated dynamically)
    # Using JSON/JSONB for flexibility during development
    features = Column(JSON_TYPE)
    
    __table_args__ = (
        Index('idx_feature_ticker_date', 'ticker', 'date', unique=True),
    )


class GDELTArticle(Base):
    """GDELT news articles."""
    __tablename__ = 'gdelt_articles'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    datetime = Column(DateTime, nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)  # Market close date
    title = Column(Text)
    url = Column(Text)
    source = Column(String(200))
    language = Column(String(10))
    country = Column(String(10))
    gdelt_fields = Column(JSON_TYPE)  # Store additional GDELT fields
    
    __table_args__ = (
        Index('idx_gdelt_ticker_date', 'ticker', 'date'),
    )


class SentimentDaily(Base):
    """Daily aggregated sentiment scores."""
    __tablename__ = 'sentiment_daily'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    n_articles = Column(Integer)
    mean_pos = Column(Float)
    mean_neg = Column(Float)
    mean_neu = Column(Float)
    mean_net = Column(Float)  # mean_pos - mean_neg
    sentiment_change_5d = Column(Float)
    sentiment_change_20d = Column(Float)
    coverage_zscore = Column(Float)
    volume_spike = Column(Float)
    
    __table_args__ = (
        Index('idx_sentiment_ticker_date', 'ticker', 'date', unique=True),
    )


class Label(Base):
    """Forward return labels and triple barrier labels."""
    __tablename__ = 'labels'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    label_horizon_63d = Column(Integer)  # 1 if positive return, 0 otherwise
    label_horizon_84d = Column(Integer)
    forward_return_63d = Column(Float)
    forward_return_84d = Column(Float)
    benchmark_return_63d = Column(Float)
    benchmark_return_84d = Column(Float)
    triple_barrier_label = Column(Integer)  # -1, 0, 1
    triple_barrier_outcome = Column(String(20))  # 'profit', 'stop', 'time'
    
    __table_args__ = (
        Index('idx_label_ticker_date', 'ticker', 'date', unique=True),
    )


def get_db_engine():
    """Create database engine from config."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database']
    
    if db_config['type'] == 'sqlite':
        db_path = db_config.get('path', 'data/stock_predictor.db')
        # Make path absolute relative to project root
        if not os.path.isabs(db_path):
            project_root = os.path.dirname(os.path.dirname(__file__))
            db_path = os.path.join(project_root, db_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        engine = create_engine(f'sqlite:///{db_path}', echo=False)
    else:
        password = os.getenv('DB_PASSWORD', db_config.get('password', ''))
        connection_string = (
            f"postgresql://{db_config['user']}:{password}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        engine = create_engine(connection_string, echo=False)
    
    return engine


def init_database():
    """Initialize database tables."""
    engine = get_db_engine()
    Base.metadata.create_all(engine)
    print("Database tables created successfully.")


if __name__ == '__main__':
    init_database()

