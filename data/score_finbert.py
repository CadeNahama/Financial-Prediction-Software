"""
Score news articles with FinBERT sentiment model.
"""
import pandas as pd
from sqlalchemy.orm import Session
from datetime import date
import yaml
from tqdm import tqdm
from data.database import get_db_engine, GDELTArticle, SentimentDaily, init_database
from sqlalchemy import create_engine, func
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers library not available.")
    TRANSFORMERS_AVAILABLE = False


class FinBERTScorer:
    """FinBERT sentiment scorer."""
    
    def __init__(self, model_name="ProsusAI/finbert", device="auto"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for FinBERT scoring")
        
        self.model_name = model_name
        self.device = device
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading FinBERT model: {model_name}")
        print(f"Device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # FinBERT labels: positive, negative, neutral
        self.label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
    
    def score_text(self, text):
        """
        Score a single text with FinBERT.
        
        Returns:
            dict with 'positive', 'negative', 'neutral' probabilities
        """
        if not text or pd.isna(text):
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
        
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            probs = probs.cpu().numpy()[0]
            
            return {
                'positive': float(probs[0]),
                'negative': float(probs[1]),
                'neutral': float(probs[2])
            }
        except Exception as e:
            print(f"Error scoring text: {e}")
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
    
    def score_batch(self, texts, batch_size=32):
        """
        Score a batch of texts.
        
        Returns:
            List of probability dicts
        """
        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Scoring batch"):
            batch = texts[i:i+batch_size]
            batch_results = [self.score_text(text) for text in batch]
            results.extend(batch_results)
        return results


def load_finbert_config():
    """Load FinBERT configuration."""
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config['data_sources']['finbert']


def get_unscored_articles(engine, ticker=None):
    """Get articles that haven't been scored yet."""
    with Session(engine) as session:
        query = session.query(GDELTArticle).filter(
            GDELTArticle.title.isnot(None)
        )
        
        if ticker:
            query = query.filter(GDELTArticle.ticker == ticker)
        
        articles = query.all()
        
        # Convert to DataFrame
        data = []
        for article in articles:
            data.append({
                'id': article.id,
                'ticker': article.ticker,
                'date': article.date,
                'title': article.title
            })
        
        return pd.DataFrame(data)


def aggregate_sentiment_daily(engine, ticker, scorer):
    """
    Score articles and aggregate to daily sentiment features.
    """
    with Session(engine) as session:
        # Get all articles for ticker
        articles = session.query(GDELTArticle).filter(
            GDELTArticle.ticker == ticker,
            GDELTArticle.title.isnot(None)
        ).all()
        
        if not articles:
            return None
        
        # Score articles
        texts = [article.title for article in articles]
        scores = scorer.score_batch(texts)
        
        # Create DataFrame
        df = pd.DataFrame({
            'ticker': [article.ticker for article in articles],
            'date': [article.date for article in articles],
            'pos': [s['positive'] for s in scores],
            'neg': [s['negative'] for s in scores],
            'neu': [s['neutral'] for s in scores]
        })
        
        # Aggregate by date
        daily = df.groupby('date').agg({
            'pos': ['mean', 'count'],
            'neg': 'mean',
            'neu': 'mean'
        }).reset_index()
        
        daily.columns = ['date', 'mean_pos', 'n_articles', 'mean_neg', 'mean_neu']
        daily['ticker'] = ticker
        daily['mean_net'] = daily['mean_pos'] - daily['mean_neg']
        
        # Calculate rolling changes
        daily = daily.sort_values('date')
        daily['sentiment_change_5d'] = daily['mean_net'].diff(5)
        daily['sentiment_change_20d'] = daily['mean_net'].diff(20)
        
        # Calculate coverage z-score (volume spike)
        window = 20
        daily['coverage_zscore'] = (
            (daily['n_articles'] - daily['n_articles'].rolling(window, min_periods=1).mean()) /
            (daily['n_articles'].rolling(window, min_periods=1).std() + 1e-6)
        )
        daily['volume_spike'] = daily['coverage_zscore']
        
        return daily[['ticker', 'date', 'n_articles', 'mean_pos', 'mean_neg', 
                      'mean_neu', 'mean_net', 'sentiment_change_5d', 
                      'sentiment_change_20d', 'coverage_zscore', 'volume_spike']]


def upsert_sentiment(df, engine):
    """Upsert sentiment data."""
    dialect = engine.dialect.name
    
    if dialect == 'postgresql':
        from sqlalchemy.dialects.postgresql import insert
        for _, row in df.iterrows():
            stmt = insert(SentimentDaily).values(**row.to_dict())
            stmt = stmt.on_conflict_do_update(
                index_elements=['ticker', 'date'],
                set_=dict(
                    n_articles=stmt.excluded.n_articles,
                    mean_pos=stmt.excluded.mean_pos,
                    mean_neg=stmt.excluded.mean_neg,
                    mean_neu=stmt.excluded.mean_neu,
                    mean_net=stmt.excluded.mean_net,
                    sentiment_change_5d=stmt.excluded.sentiment_change_5d,
                    sentiment_change_20d=stmt.excluded.sentiment_change_20d,
                    coverage_zscore=stmt.excluded.coverage_zscore,
                    volume_spike=stmt.excluded.volume_spike
                )
            )
            engine.execute(stmt)
    else:
        with Session(engine) as session:
            for _, row in df.iterrows():
                # Delete existing
                session.query(SentimentDaily).filter(
                    SentimentDaily.ticker == row['ticker'],
                    SentimentDaily.date == row['date']
                ).delete()
                
                sentiment = SentimentDaily(**row.to_dict())
                session.add(sentiment)
            
            session.commit()


def score_finbert(tickers=None):
    """
    Main FinBERT scoring function.
    
    Args:
        tickers: List of tickers (None = all in database)
    """
    init_database()
    
    if not TRANSFORMERS_AVAILABLE:
        print("FinBERT scoring requires transformers library.")
        print("Install with: pip install transformers torch")
        return
    
    # Load configuration
    finbert_config = load_finbert_config()
    model_name = finbert_config.get('model_name', 'ProsusAI/finbert')
    device = finbert_config.get('device', 'auto')
    batch_size = finbert_config.get('batch_size', 32)
    
    engine = get_db_engine()
    
    # Load tickers
    if tickers is None:
        with open('configs/tickers.yaml', 'r') as f:
            config = yaml.safe_load(f)
        tickers = config['tickers']
    
    # Initialize scorer
    scorer = FinBERTScorer(model_name=model_name, device=device)
    scorer.model.config.batch_size = batch_size
    
    print(f"Scoring sentiment for {len(tickers)} tickers...")
    
    for ticker in tqdm(tickers, desc="Processing tickers"):
        daily_df = aggregate_sentiment_daily(engine, ticker, scorer)
        
        if daily_df is not None and not daily_df.empty:
            print(f"  {ticker}: {len(daily_df)} days of sentiment data")
            upsert_sentiment(daily_df, engine)
        else:
            print(f"  {ticker}: No articles to score")
    
    print("FinBERT scoring complete!")


if __name__ == '__main__':
    score_finbert()

