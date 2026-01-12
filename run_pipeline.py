#!/usr/bin/env python3
"""
Main entry point to run the full pipeline for a list of tickers.

Usage:
    python run_pipeline.py                    # Use tickers from config
    python run_pipeline.py AAPL MSFT GOOGL   # Specify tickers
"""
import sys
import os
import yaml
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data.database import init_database
from data.ingest_prices import ingest_prices
from data.ingest_gdelt import ingest_gdelt
from data.score_finbert import score_finbert
from features.build_market_features import build_all_market_features
from features.build_media_features import build_all_media_features
from labels.make_forward_return_labels import create_all_labels
from models.train_model import train_model
from models.generate_outlook import generate_all_outlooks


def update_tickers_config(tickers):
    """Update tickers.yaml with provided tickers and return the requested tickers."""
    config_path = 'configs/tickers.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add tickers if not present
    existing_tickers = set(config.get('tickers', []))
    new_tickers = [t for t in tickers if t not in existing_tickers]
    
    if new_tickers:
        config['tickers'].extend(new_tickers)
        
        # Add basic aliases for new tickers
        if 'ticker_aliases' not in config:
            config['ticker_aliases'] = {}
        
        for ticker in new_tickers:
            if ticker not in config['ticker_aliases']:
                config['ticker_aliases'][ticker] = [ticker]
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Added {len(new_tickers)} new tickers to config: {new_tickers}")
    
    # Return only the requested tickers, not all tickers in config
    return tickers


def run_pipeline(tickers=None, skip_media=False):
    """
    Run the full pipeline.
    
    Args:
        tickers: List of tickers (None = use config)
        skip_media: Skip GDELT/FinBERT steps (faster for testing)
    """
    print("=" * 60)
    print("Stock Predictor Pipeline")
    print("=" * 60)
    
    # Initialize database
    print("\n[1/7] Initializing database...")
    init_database()
    
    # Update config if tickers provided
    if tickers:
        tickers = update_tickers_config(tickers)
    else:
        with open('configs/tickers.yaml', 'r') as f:
            config = yaml.safe_load(f)
        tickers = config['tickers']
    
    print(f"\nProcessing {len(tickers)} tickers: {', '.join(tickers)}")
    
    # Ingest prices
    print("\n[2/7] Ingesting price data...")
    ingest_prices(tickers=tickers)
    
    # Ingest media (optional)
    if not skip_media:
        print("\n[3/7] Ingesting GDELT articles...")
        try:
            ingest_gdelt(tickers=tickers)
        except Exception as e:
            print(f"Warning: GDELT ingestion failed: {e}")
            print("Continuing without media data...")
        
        print("\n[4/7] Scoring articles with FinBERT...")
        try:
            score_finbert(tickers=tickers)
        except Exception as e:
            print(f"Warning: FinBERT scoring failed: {e}")
            print("Continuing without sentiment data...")
    else:
        print("\n[3-4/7] Skipping media ingestion (skip_media=True)")
    
    # Build features
    print("\n[5/7] Building market features...")
    build_all_market_features(tickers=tickers)
    
    if not skip_media:
        print("\n[5b/7] Building media features...")
        try:
            build_all_media_features(tickers=tickers)
        except Exception as e:
            print(f"Warning: Media feature building failed: {e}")
    
    # Create labels
    print("\n[6/7] Creating labels...")
    create_all_labels(tickers=tickers)
    
    # Train model
    print("\n[7/7] Training model...")
    try:
        train_model(tickers=tickers)
    except Exception as e:
        print(f"Error training model: {e}")
        print("Model training failed. Check data availability.")
        return
    
    # Generate outlook
    print("\n[8/8] Generating outlook...")
    generate_all_outlooks(tickers=tickers)
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Check reports/outlook.json for predictions")
    print("  - Review model performance in models/trained_model_metadata.json")
    print("  - Set up LEAN backtesting (see lean/Algorithm.py)")


if __name__ == '__main__':
    # Parse command line arguments
    if len(sys.argv) > 1:
        tickers = sys.argv[1:]
        skip_media = '--skip-media' in tickers
        if skip_media:
            tickers.remove('--skip-media')
        run_pipeline(tickers=tickers if tickers else None, skip_media=skip_media)
    else:
        run_pipeline()

