#!/usr/bin/env python3
"""
Enhanced pipeline with all new features:
- Triple barrier labeling
- Macro features (VIX, rates)
- Feature selection
- Hyperparameter tuning
- Regime detection
- Visualizations
"""
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data.database import init_database
from data.ingest_prices import ingest_prices
from data.ingest_macro import ingest_macro_data
from data.ingest_gdelt import ingest_gdelt
from data.score_finbert import score_finbert
from features.build_market_features import build_all_market_features
from features.build_media_features import build_all_media_features
from features.build_macro_features import build_all_macro_features
from labels.make_forward_return_labels import create_all_labels
from labels.make_triple_barrier_labels import create_all_triple_barrier_labels
from models.regime_detection import add_regime_features_to_features
from models.train_model_enhanced import train_model_enhanced
from models.generate_outlook import generate_all_outlooks
from reports.visualizations import generate_all_visualizations
import yaml


def run_enhanced_pipeline(tickers=None, skip_media=False, use_triple_barrier=True, 
                         tune_hyperparams=True, use_feature_selection=True):
    """
    Run enhanced pipeline with all new features.
    
    Args:
        tickers: List of tickers (None = use config)
        skip_media: Skip GDELT/FinBERT steps
        use_triple_barrier: Use triple barrier labels (default: True)
        tune_hyperparams: Tune hyperparameters (default: True)
        use_feature_selection: Use feature selection (default: True)
    """
    print("=" * 70)
    print("Enhanced Stock Predictor Pipeline")
    print("=" * 70)
    
    # Initialize database
    print("\n[1/11] Initializing database...")
    init_database()
    
    # Load tickers
    if tickers:
        config_path = 'configs/tickers.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        existing_tickers = set(config.get('tickers', []))
        new_tickers = [t for t in tickers if t not in existing_tickers]
        if new_tickers:
            config['tickers'].extend(new_tickers)
            if 'ticker_aliases' not in config:
                config['ticker_aliases'] = {}
            for ticker in new_tickers:
                if ticker not in config['ticker_aliases']:
                    config['ticker_aliases'][ticker] = [ticker]
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        tickers = tickers
    else:
        with open('configs/tickers.yaml', 'r') as f:
            config = yaml.safe_load(f)
        tickers = config['tickers']
    
    print(f"\nProcessing {len(tickers)} tickers: {', '.join(tickers)}")
    
    # Ingest prices
    print("\n[2/11] Ingesting price data...")
    ingest_prices(tickers=tickers)
    
    # Ingest macro data
    print("\n[3/11] Ingesting macro data (VIX, rates)...")
    try:
        ingest_macro_data()
    except Exception as e:
        print(f"Warning: Macro data ingestion failed: {e}")
        print("Continuing without macro data...")
    
    # Ingest media (optional)
    if not skip_media:
        print("\n[4/11] Ingesting GDELT articles...")
        try:
            ingest_gdelt(tickers=tickers)
        except Exception as e:
            print(f"Warning: GDELT ingestion failed: {e}")
            print("Continuing without media data...")
        
        print("\n[5/11] Scoring articles with FinBERT...")
        try:
            score_finbert(tickers=tickers)
        except Exception as e:
            print(f"Warning: FinBERT scoring failed: {e}")
            print("Continuing without sentiment data...")
    else:
        print("\n[4-5/11] Skipping media ingestion (skip_media=True)")
    
    # Build features
    print("\n[6/11] Building market features...")
    build_all_market_features(tickers=tickers)
    
    if not skip_media:
        print("\n[7/11] Building media features...")
        try:
            build_all_media_features(tickers=tickers)
        except Exception as e:
            print(f"Warning: Media feature building failed: {e}")
    
    print("\n[8/11] Building macro features...")
    try:
        build_all_macro_features()
    except Exception as e:
        print(f"Warning: Macro feature building failed: {e}")
    
    print("\n[9/11] Adding regime features...")
    try:
        from data.database import get_db_engine
        engine = get_db_engine()
        add_regime_features_to_features(engine)
    except Exception as e:
        print(f"Warning: Regime detection failed: {e}")
    
    # Create labels
    print("\n[10/11] Creating forward return labels...")
    create_all_labels(tickers=tickers)
    
    triple_barrier_available = False
    if use_triple_barrier:
        print("\n[10b/11] Creating triple barrier labels...")
        try:
            create_all_triple_barrier_labels(tickers=tickers)
            # Check if triple barrier labels were actually created
            from data.database import get_db_engine, Label
            from sqlalchemy.orm import Session
            engine_check = get_db_engine()
            with Session(engine_check) as session:
                count = session.query(Label).filter(Label.triple_barrier_label.isnot(None)).count()
                if count > 0:
                    print(f"Triple barrier labels created: {count} labels")
                else:
                    print("No triple barrier labels created, using forward return labels")
                    use_triple_barrier = False
        except Exception as e:
            print(f"Warning: Triple barrier labeling failed: {e}")
            print("Falling back to forward return labels only...")
            use_triple_barrier = False
    
    # Train model
    print("\n[11/11] Training enhanced model...")
    print(f"  - Triple barrier: {use_triple_barrier}")
    print(f"  - Hyperparameter tuning: {tune_hyperparams}")
    print(f"  - Feature selection: {use_feature_selection}")
    
    try:
        train_model_enhanced(
            tickers=tickers,
            use_triple_barrier=use_triple_barrier,
            tune_hyperparams=tune_hyperparams,
            use_feature_selection=use_feature_selection
        )
    except Exception as e:
        print(f"Error training model: {e}")
        print("Model training failed. Check data availability.")
        import traceback
        traceback.print_exc()
        return
    
    # Generate outlook
    print("\n[12/12] Generating outlook...")
    try:
        generate_all_outlooks(tickers=tickers)
    except Exception as e:
        print(f"Warning: Outlook generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate visualizations
    print("\n[13/13] Generating visualizations...")
    try:
        generate_all_visualizations()
    except Exception as e:
        print(f"Warning: Visualization generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Enhanced pipeline complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Check reports/outlook.json for predictions")
    print("  - View reports/predictions_2d.png and predictions_3d.html")
    print("  - Review model performance in models/trained_model_enhanced_metadata.json")
    print("  - Check feature importance in reports/feature_importance.png")


if __name__ == '__main__':
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Run enhanced stock predictor pipeline')
    parser.add_argument('tickers', nargs='*', help='Ticker symbols (e.g., AAPL MSFT GOOGL)')
    parser.add_argument('--skip-media', action='store_true', help='Skip GDELT/FinBERT steps')
    parser.add_argument('--no-triple-barrier', action='store_true', help='Use forward return labels instead of triple barrier')
    parser.add_argument('--no-tuning', action='store_true', help='Skip hyperparameter tuning')
    parser.add_argument('--no-feature-selection', action='store_true', help='Skip feature selection')
    
    args = parser.parse_args()
    
    run_enhanced_pipeline(
        tickers=args.tickers if args.tickers else None,
        skip_media=args.skip_media,
        use_triple_barrier=not args.no_triple_barrier,
        tune_hyperparams=not args.no_tuning,
        use_feature_selection=not args.no_feature_selection
    )

