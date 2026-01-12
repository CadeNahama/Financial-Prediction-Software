# Financial Prediction Software: 3-4 Month Stock Outlook System

A comprehensive machine learning system for predicting 3-4 month directional stock outlooks using market data, macroeconomic indicators, and media sentiment analysis. Built with rigorous time-series validation, risk-aware position sizing, and production-ready backtesting capabilities.

## Overview

This system combines quantitative finance, natural language processing, and machine learning to generate probabilistic predictions about stock price movements over 63-84 trading days (approximately 3-4 months). The system is designed for institutional-quality analysis with proper validation, risk management, and realistic backtesting.

### Key Features

- **Multi-Source Data Integration**: Market prices, macroeconomic indicators (VIX, interest rates, dollar index), and news sentiment
- **Advanced Feature Engineering**: 80+ engineered features including momentum, trend, volatility, drawdown, relative strength, and regime indicators
- **Machine Learning Pipeline**: XGBoost/LightGBM classifiers with walk-forward validation, hyperparameter tuning, and feature selection
- **Sentiment Analysis**: GDELT news ingestion with FinBERT financial sentiment scoring
- **Risk Management**: Volatility-based position sizing, maximum drawdown limits, and portfolio constraints
- **Visualizations**: Interactive 2D and 3D plots for predictions, feature importance, and performance analysis
- **Production Ready**: SQLite (development) or PostgreSQL (production) database, configurable pipelines, and comprehensive error handling

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [Data Pipeline](#data-pipeline)
- [Model Details](#model-details)
- [Output Format](#output-format)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning repository)

### Step 1: Clone Repository

```bash
git clone https://github.com/CadeNahama/Financial-Prediction-Software.git
cd Financial-Prediction-Software
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Initialize Database

```bash
python setup.py
```

Or using Makefile:

```bash
make init-db
```

This will:
- Create necessary directories (data/, reports/, models/)
- Initialize SQLite database (or connect to PostgreSQL if configured)
- Create all database tables
- Verify dependencies

### Optional: PostgreSQL Setup

For production use, configure PostgreSQL in `configs/config.yaml`:

```yaml
database:
  type: postgresql
  host: localhost
  port: 5432
  database: stock_predictor
  user: postgres
  password: ${DB_PASSWORD}  # Set in .env file
```

Create `.env` file:

```bash
cp .env.example .env
# Edit .env with your database credentials
```

## Quick Start

### 1. Configure Tickers

Edit `configs/tickers.yaml` to add your stocks:

```yaml
tickers:
  - AAPL
  - MSFT
  - GOOGL
  - AMZN

ticker_aliases:
  AAPL:
    - "Apple"
    - "AAPL"
    - "Tim Cook"
  MSFT:
    - "Microsoft"
    - "MSFT"
    - "Satya Nadella"
```

### 2. Run Full Pipeline

**Option A: Standard Pipeline (Recommended for First Run)**

```bash
python run_pipeline.py AAPL MSFT GOOGL
```

**Option B: Enhanced Pipeline (With Advanced Features)**

```bash
python run_enhanced_pipeline.py AAPL MSFT GOOGL
```

**Option C: Skip Media Ingestion (Faster, Market-Only Model)**

```bash
python run_pipeline.py AAPL MSFT GOOGL --skip-media
```

**Option D: Using Makefile**

```bash
make all
```

### 3. View Results

After completion, check:

- **Predictions**: `reports/outlook.json`
- **2D Plot**: `reports/predictions_2d.png`
- **3D Interactive Plot**: `reports/predictions_3d.html` (open in browser)
- **Performance Metrics**: Console output and saved validation results

## Configuration

### Main Configuration File (`configs/config.yaml`)

Key settings:

```yaml
# Database
database:
  type: sqlite  # or postgresql
  path: data/stock_predictor.db

# Data Sources
data_sources:
  prices:
    start_date: "2010-01-01"  # Historical data start
    end_date: null  # null = today
  gdelt:
    lookback_days: 90  # News history to fetch
  finbert:
    model_name: "ProsusAI/finbert"
    batch_size: 32

# Feature Engineering
features:
  market:
    momentum_windows: [20, 63, 126]
    volatility_windows: [20, 63]
    ma_windows: [20, 63, 200]
  media:
    sentiment_windows: [5, 20]

# Labeling
labeling:
  horizons: [63, 84]  # Trading days
  benchmark: SPY

# Modeling
modeling:
  algorithm: xgboost  # or lightgbm
  walk_forward:
    train_years: 6
    validation_years: 1
    test_years: 1
    step_years: 1

# Risk Management
risk:
  max_drawdown: 0.20  # 20% max drawdown
  max_position_size: 0.15  # 15% per position
```

### Ticker Configuration (`configs/tickers.yaml`)

Define stocks and their aliases for news queries:

```yaml
tickers:
  - AAPL
  - MSFT
  # Add more...

ticker_aliases:
  AAPL:
    - "Apple"
    - "AAPL"
    - "Tim Cook"
    # Add company name, ticker, CEO, keywords
```

## Usage

### Basic Workflow

#### 1. Add New Tickers

Edit `configs/tickers.yaml`, then run:

```bash
# Ingest price data
make prices

# Ingest news (optional)
make gdelt

# Score sentiment (optional)
make finbert
```

#### 2. Build Features

```bash
# Market features (always needed)
make market-features

# Media features (if using news)
make media-features

# Macro features (VIX, rates, DXY)
make macro-features
```

#### 3. Create Labels

```bash
# Forward return labels
make labels

# Triple barrier labels (optional, requires mlfinlab)
make triple-barrier-labels
```

#### 4. Train Model

```bash
# Standard training
make train

# Enhanced training (with tuning and selection)
python run_enhanced_pipeline.py --no-triple-barrier
```

#### 5. Generate Predictions

```bash
make outlook
```

Output saved to `reports/outlook.json`

### Advanced Usage

#### Step-by-Step Execution

```bash
# 1. Initialize database
make init-db

# 2. Ingest data
make prices
make macro-data          # VIX, rates, DXY
make gdelt               # Optional: news articles
make finbert             # Optional: sentiment scoring

# 3. Build features
make market-features
make macro-features
make media-features      # Optional
make regime-features     # Optional: regime detection

# 4. Create labels
make labels

# 5. Train model
make train

# 6. Generate outlook
make outlook
```

#### Enhanced Pipeline Options

```bash
# Full enhanced pipeline
python run_enhanced_pipeline.py AAPL MSFT GOOGL

# Disable specific features
python run_enhanced_pipeline.py AAPL MSFT GOOGL --no-triple-barrier
python run_enhanced_pipeline.py AAPL MSFT GOOGL --skip-media
python run_enhanced_pipeline.py AAPL MSFT GOOGL --no-tuning
python run_enhanced_pipeline.py AAPL MSFT GOOGL --no-feature-selection
```

#### Generate Visualizations

```bash
python -c "from reports.visualizations import generate_all_visualizations; generate_all_visualizations()"
```

Or use existing outlook.json:

```bash
python -c "from reports.visualizations import load_outlook_json, plot_predictions_2d, plot_predictions_3d; data = load_outlook_json('reports/outlook.json'); plot_predictions_2d(data, 'reports/predictions_2d.png'); plot_predictions_3d(data, 'reports/predictions_3d.html')"
```

## Architecture

### System Overview

```
┌─────────────────┐
│  Configuration  │  tickers.yaml, config.yaml
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Ingestion │  Prices, Macro, News, Sentiment
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Database     │  SQLite/PostgreSQL
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Features     │  Market, Media, Macro, Regime
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Labels      │  Forward Returns / Triple Barrier
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Training      │  Walk-Forward Validation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Prediction    │  Outlook Generation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Visualization   │  2D/3D Plots, Reports
└─────────────────┘
```

### Project Structure

```
Financial-Prediction-Software/
├── configs/                      # Configuration files
│   ├── tickers.yaml             # Ticker list and aliases
│   └── config.yaml              # Main configuration
├── data/                        # Data ingestion
│   ├── database.py              # Database schema and connection
│   ├── ingest_prices.py         # Price data from yfinance
│   ├── ingest_macro.py          # Macro data (VIX, rates, DXY)
│   ├── ingest_gdelt.py          # News articles from GDELT
│   └── score_finbert.py         # FinBERT sentiment scoring
├── features/                    # Feature engineering
│   ├── build_market_features.py # Market technical features
│   ├── build_media_features.py  # News sentiment features
│   ├── build_macro_features.py  # Macro economic features
│   └── merge_features.py        # Feature merging logic
├── labels/                      # Labeling strategies
│   ├── make_forward_return_labels.py    # Forward return labels
│   └── make_triple_barrier_labels.py    # Triple barrier labels
├── models/                      # Model training and prediction
│   ├── train_model.py           # Standard training
│   ├── train_model_enhanced.py  # Enhanced training pipeline
│   ├── feature_selection.py     # Feature importance selection
│   ├── hyperparameter_tuning.py # Optuna-based tuning
│   ├── regime_detection.py      # Market regime classification
│   └── generate_outlook.py      # Prediction generation
├── reports/                     # Generated outputs
│   ├── outlook.json             # Prediction results
│   ├── predictions_2d.png       # 2D visualization
│   ├── predictions_3d.html      # 3D interactive plot
│   └── visualizations.py        # Visualization code
├── lean/                        # LEAN backtesting integration
├── run_pipeline.py              # Standard pipeline runner
├── run_enhanced_pipeline.py     # Enhanced pipeline runner
├── setup.py                     # Setup script
├── utils.py                     # Utility functions
├── Makefile                     # CLI commands
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Data Pipeline

### 1. Price Data Ingestion

**Source**: Yahoo Finance via `yfinance`

**Data Collected**:
- Open, High, Low, Close prices
- Volume
- Adjusted close prices

**Storage**: `prices_daily` table

**Command**: `python data/ingest_prices.py` or `make prices`

### 2. Macro Data Ingestion

**Source**: Yahoo Finance

**Data Collected**:
- VIX (Volatility Index)
- 10-Year Treasury Rate
- 2-Year Treasury Rate
- Dollar Index (DXY)

**Storage**: `macro_daily` table

**Command**: `python data/ingest_macro.py` or `make macro-data`

### 3. News Data Ingestion

**Source**: GDELT Doc API

**Data Collected**:
- News articles mentioning ticker keywords
- Article metadata (date, source, title)

**Storage**: `gdelt_articles` table

**Command**: `python data/ingest_gdelt.py` or `make gdelt`

**Note**: Requires `gdeltdoc` package and ticker aliases in `tickers.yaml`

### 4. Sentiment Scoring

**Model**: FinBERT (ProsusAI/finbert)

**Process**:
- Loads FinBERT transformer model
- Scores each article headline
- Generates positive/negative/neutral probabilities

**Storage**: `sentiment_daily` table (aggregated daily)

**Command**: `python data/score_finbert.py` or `make finbert`

**Note**: First run downloads ~438MB model. Uses CPU by default.

## Model Details

### Feature Engineering

#### Market Features (33 features)

- **Momentum**: Returns over 20/63/126 trading days
- **Trend**: Moving averages (20/63/200), crossovers, slopes
- **Volatility**: 20/63-day realized volatility (annualized)
- **Drawdown**: Rolling maximum drawdown, days since peak
- **Relative Strength**: Stock returns minus SPY benchmark returns
- **Liquidity**: Dollar volume, volume ratios, turnover

#### Macro Features (15+ features)

- **VIX Features**: Current value, moving averages, changes, z-scores
- **Interest Rate Features**: 10Y/2Y rates, yield spread, changes
- **Dollar Index**: Changes, trends, relative levels
- **Regime Indicators**: Risk-off flags (VIX > 20), inverted yield curve

#### Media Features (10+ features)

- **Coverage**: Article counts, coverage z-scores
- **Sentiment Aggregates**: Mean positive/negative/net sentiment
- **Sentiment Momentum**: 5-day and 20-day sentiment changes
- **Volume Spikes**: Abnormal coverage detection

#### Regime Features (9 features)

- **Market Regime**: Bull/Bear/Sideways classification
- **Volatility Regime**: High/Low/Extreme volatility
- **Trend Indicators**: Above/below moving averages

**Total**: 80+ features per ticker per day

### Labeling Strategies

#### Forward Return Labels

**Method**: Calculate forward return over 63/84 trading days

**Label**: Binary (1 = positive return, 0 = negative return)

**Variants**:
- Absolute: Positive if stock return > 0
- Relative: Positive if stock return > SPY return

**Storage**: `labels` table, columns `label_horizon_63d`, `label_horizon_84d`

#### Triple Barrier Labels

**Method**: Model actual trade outcomes with barriers

**Barriers**:
- **Profit Take**: 5% gain (label = 1)
- **Stop Loss**: 3% loss (label = -1)
- **Time Barrier**: 63 days without hitting barriers (label = 0)

**Advantage**: More realistic for trading, accounts for risk management

**Storage**: `labels` table, column `triple_barrier_label`

**Note**: Requires `mlfinlab` package (has fallback implementation)

### Training Process

#### Walk-Forward Validation

**Purpose**: Prevent overfitting and data leakage in time-series data

**Method**:
1. Split data chronologically (no random splits)
2. Train on past data, validate on recent data, test on future data
3. Step forward in time and repeat

**Example Window**:
- Train: 2010-2016 (6 years)
- Validate: 2016-2017 (1 year)
- Test: 2017-2018 (1 year)
- Step: 1 year forward
- Repeat for all available data

**Benefits**:
- Realistic performance estimation
- Accounts for non-stationarity
- Prevents look-ahead bias

#### Model Algorithms

**XGBoost** (Default):
- Gradient boosting framework
- Handles non-linear relationships
- Feature importance built-in

**LightGBM** (Alternative):
- Faster training
- Better memory efficiency
- Similar performance to XGBoost

#### Hyperparameter Tuning

**Tool**: Optuna (Bayesian optimization)

**Parameters Tuned**:
- `n_estimators`: Number of trees
- `max_depth`: Maximum tree depth
- `learning_rate`: Step size
- `subsample`: Row sampling
- `colsample_bytree`: Column sampling

**Typical Improvement**: 5-10% over default parameters

**Command**: Enabled by default in `run_enhanced_pipeline.py`

#### Feature Selection

**Method**: Importance-based selection using trained model

**Process**:
1. Train base model to get feature importance
2. Select top N features (default: 50)
3. Retrain final model on selected features

**Benefits**: Reduces noise, improves signal, faster training

**Command**: Enabled by default in `run_enhanced_pipeline.py`

#### Probability Calibration

**Purpose**: Ensure predicted probabilities are meaningful

**Methods**:
- **Isotonic Regression**: Non-parametric calibration
- **Platt Scaling**: Logistic regression calibration

**Result**: When model predicts 60% probability, it wins ~60% of the time

**Configuration**: `configs/config.yaml` → `modeling.calibration`

### Model Output

The trained model outputs a probability (0.0 to 1.0) representing the likelihood of a positive return over the next 63-84 trading days.

**Interpretation**:
- `0.50` = 50% chance (coin flip, no edge)
- `> 0.55` = Bullish signal (more likely to go up)
- `< 0.50` = Bearish signal (more likely to go down)

## Output Format

### Outlook JSON Structure

The system generates `reports/outlook.json` with predictions for all tickers:

```json
[
  {
    "ticker": "AAPL",
    "date": "2026-01-11",
    "p_up_63d": 0.574,
    "p_up_84d": 0.574,
    "recommended_action": "Long",
    "suggested_size": 0.0223,
    "current_price": 259.37,
    "volatility": 0.110,
    "key_feature_drivers": null
  }
]
```

### Field Explanations

- **`ticker`**: Stock symbol
- **`date`**: Prediction date
- **`p_up_63d`**: Probability of positive return over 63 trading days (0.0-1.0)
- **`p_up_84d`**: Probability of positive return over 84 trading days (0.0-1.0)
- **`recommended_action`**: Trading recommendation
  - `"Long"`: p_up > 0.55 (buy/hold)
  - `"No Trade"`: 0.50 < p_up ≤ 0.55 (neutral/wait)
  - `"Reduce"`: p_up ≤ 0.50 (sell/avoid)
- **`suggested_size`**: Position size as fraction of portfolio (0.0-0.15, max 15%)
- **`current_price`**: Latest closing price (USD)
- **`volatility`**: Annualized volatility (20-day rolling)
- **`key_feature_drivers`**: Top features driving prediction (if available)

### Visualizations

**2D Plot** (`reports/predictions_2d.png`):
- Scatter plot: Probability vs Volatility
- Color-coded by recommended action
- Useful for quick overview

**3D Plot** (`reports/predictions_3d.html`):
- Interactive Plotly visualization
- Axes: Probability, Volatility, Position Size
- Hover for ticker details
- Rotate, zoom, pan interactively

**Performance Charts**:
- Feature importance rankings
- Model performance over time
- Validation metrics by window

## Advanced Features

### Enhanced Pipeline

The enhanced pipeline (`run_enhanced_pipeline.py`) includes:

1. **Triple Barrier Labeling**: More realistic trade outcome modeling
2. **Macro Features**: VIX, interest rates, dollar index integration
3. **Feature Selection**: Importance-based feature reduction
4. **Hyperparameter Tuning**: Optuna-based optimization
5. **Regime Detection**: Market condition classification

**Usage**:

```bash
# Full enhanced pipeline
python run_enhanced_pipeline.py AAPL MSFT GOOGL

# Disable specific features
python run_enhanced_pipeline.py AAPL MSFT GOOGL --no-triple-barrier
python run_enhanced_pipeline.py AAPL MSFT GOOGL --no-tuning
python run_enhanced_pipeline.py AAPL MSFT GOOGL --no-feature-selection
```

### Regime Detection

The system classifies market regimes:

- **Bull Market**: SPY above 200MA, positive momentum
- **Bear Market**: SPY below 200MA, negative momentum
- **Sideways**: Mixed signals
- **High Volatility**: VIX > 30
- **Low Volatility**: VIX < 15

Regime features are added to all ticker features, allowing the model to adapt to different market conditions.

### Backtesting Integration

LEAN backtesting integration is available in the `lean/` directory. See QuantConnect LEAN documentation for setup:

- https://github.com/QuantConnect/Lean
- https://github.com/QuantConnect/lean-cli

The backtesting includes:
- Transaction costs (0.1% per trade)
- Slippage (0.05%)
- Position sizing constraints
- Risk limits (max drawdown, volatility caps)

## Troubleshooting

### Common Issues

#### 1. Database Connection Errors

**SQLite**: Ensure `data/` directory exists and is writable

```bash
mkdir -p data
chmod 755 data
```

**PostgreSQL**: Verify credentials in `.env` file and database exists

```bash
createdb stock_predictor
```

#### 2. Missing Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

For optional features:

```bash
pip install gdeltdoc optuna  # For news and hyperparameter tuning
```

#### 3. GDELT API Errors

GDELT may reject queries with special characters. Ensure ticker aliases in `tickers.yaml` don't contain illegal characters.

If GDELT fails, use `--skip-media` flag:

```bash
python run_pipeline.py AAPL MSFT GOOGL --skip-media
```

#### 4. FinBERT Model Download

First run downloads ~438MB model. Ensure sufficient disk space and internet connection.

To use GPU (if available):

```yaml
# configs/config.yaml
finbert:
  device: cuda
```

#### 5. Training Fails with "No models trained"

**Cause**: No training data available or all windows skipped

**Solutions**:
- Ensure labels exist: `make labels`
- Check date range in database
- Verify features exist: `make market-features`
- Try standard pipeline first: `python run_pipeline.py`

#### 6. Triple Barrier Labels Not Created

Triple barrier requires `mlfinlab` package. If not installed, system falls back to forward return labels automatically.

To install mlfinlab (requires separate setup):

```bash
# Follow mlfinlab installation instructions
# https://github.com/hudson-and-thames/mlfinlab
```

#### 7. Visualizations Not Generated

Ensure outlook.json exists:

```bash
ls reports/outlook.json
```

Manually generate:

```bash
python -c "from reports.visualizations import generate_all_visualizations; generate_all_visualizations()"
```

### Performance Optimization

#### Speed Improvements

1. **Skip Media**: Use `--skip-media` to skip news ingestion (fastest)
2. **Reduce Data Range**: Edit `config.yaml` → `data_sources.prices.start_date`
3. **Fewer Tickers**: Process fewer stocks at once
4. **SQLite vs PostgreSQL**: SQLite is faster for development, PostgreSQL for production
5. **GPU for FinBERT**: Use CUDA if available (significantly faster)

#### Memory Optimization

1. **Batch Processing**: Process tickers in smaller batches
2. **Reduce Feature Windows**: Fewer lookback periods in `config.yaml`
3. **Feature Selection**: Enable to reduce memory usage
4. **Database Cleanup**: Archive old data periodically

### Getting Help

1. Check error messages in console output
2. Review logs in database or console
3. Verify configuration files are valid YAML
4. Ensure all dependencies are installed
5. Try running individual pipeline steps with Makefile

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Financial-Prediction-Software.git
cd Financial-Prediction-Software

# Install development dependencies
pip install -r requirements.txt

# Run tests (if available)
make test

# Run linting
make lint
```

### Code Style

- Follow PEP 8 for Python code
- Use type hints where possible
- Add docstrings to functions and classes
- Keep functions focused and modular

## License

[Specify your license here]

## Acknowledgments

- **yfinance**: Market data provider
- **GDELT**: Global news database
- **FinBERT**: ProsusAI for financial sentiment model
- **XGBoost/LightGBM**: Gradient boosting frameworks
- **QuantConnect LEAN**: Backtesting framework
- **Optuna**: Hyperparameter optimization

## Citation

If you use this software in research, please cite:

```bibtex
@software{financial_prediction_software,
  title = {Financial Prediction Software: 3-4 Month Stock Outlook System},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/CadeNahama/Financial-Prediction-Software}
}
```

---

For detailed usage examples, see the [OUTLOOK_EXPLANATION.md](OUTLOOK_EXPLANATION.md) file.
