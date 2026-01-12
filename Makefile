.PHONY: help ingest prices gdelt finbert features market-features media-features labels train outlook backtest report all clean init-db

help:
	@echo "Available commands:"
	@echo "  make init-db          - Initialize database tables"
	@echo "  make ingest           - Ingest all data (prices + GDELT + FinBERT)"
	@echo "  make prices           - Ingest price data only"
	@echo "  make gdelt            - Ingest GDELT news articles"
	@echo "  make finbert          - Score articles with FinBERT"
	@echo "  make features         - Build all features (market + media)"
	@echo "  make market-features  - Build market features only"
	@echo "  make media-features   - Build media features only"
	@echo "  make labels           - Create forward return labels"
	@echo "  make train            - Train model with walk-forward validation"
	@echo "  make outlook          - Generate daily outlook for all tickers"
	@echo "  make backtest         - Run LEAN backtest (requires LEAN setup)"
	@echo "  make report           - Generate backtest report"
	@echo "  make all              - Run full pipeline: ingest -> features -> labels -> train -> outlook"
	@echo "  make clean            - Clean generated files"

init-db:
	@echo "Initializing database..."
	@cd data && python database.py

ingest: prices gdelt finbert
	@echo "Data ingestion complete!"

prices:
	@echo "Ingesting price data..."
	@cd data && python ingest_prices.py

gdelt:
	@echo "Ingesting GDELT articles..."
	@cd data && python ingest_gdelt.py

finbert:
	@echo "Scoring articles with FinBERT..."
	@cd data && python score_finbert.py

features: market-features media-features
	@echo "Feature engineering complete!"

market-features:
	@echo "Building market features..."
	@cd features && python build_market_features.py

media-features:
	@echo "Building media features..."
	@cd features && python build_media_features.py

labels:
	@echo "Creating labels..."
	@cd labels && python make_forward_return_labels.py

train:
	@echo "Training model..."
	@cd models && python train_model.py

outlook:
	@echo "Generating outlook..."
	@cd models && python generate_outlook.py

backtest:
	@echo "Running LEAN backtest..."
	@echo "Note: LEAN integration requires additional setup"
	@echo "See lean/ directory for LEAN algorithm implementation"

report:
	@echo "Generating backtest report..."
	@echo "Note: Requires backtest results from LEAN"

all: ingest features labels train outlook
	@echo "Full pipeline complete!"

clean:
	@echo "Cleaning generated files..."
	@rm -f models/*.pkl models/*.json
	@rm -f reports/*.json reports/*.html reports/*.pdf
	@rm -f data/*.db data/*.sqlite
	@find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true

