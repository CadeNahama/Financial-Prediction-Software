# Understanding the Outlook Report

The outlook.json file contains predictions for each ticker with the following fields:

## Field Explanations

### `ticker`
The stock symbol (e.g., AAPL, MSFT, GOOGL)

### `date`
The prediction date (when the forecast was generated)

### `p_up_63d` / `p_up_84d`
**Probability of positive return** over the next 63 or 84 trading days (~3-4 months)

- **Range**: 0.0 to 1.0 (0% to 100%)
- **0.50 = 50%** (coin flip - no edge)
- **> 0.55 = Bullish** (more likely to go up)
- **< 0.50 = Bearish** (more likely to go down)

**Example**: `p_up_63d: 0.574` means 57.4% probability the stock will have a positive return over the next 63 trading days.

### `recommended_action`
**Trading recommendation** based on probability thresholds:

- **"Long"**: `p_up > 0.55` - Buy/hold (positive signal)
- **"No Trade"**: `0.50 < p_up ≤ 0.55` - Neutral/wait (weak signal)
- **"Reduce"**: `p_up ≤ 0.50` - Sell/avoid (negative signal)

**Example**: AAPL shows "Long" because `p_up_63d: 0.574 > 0.55`

### `suggested_size`
**Position size as fraction of portfolio** (0.0 to 1.0)

- **Range**: 0.0 to 0.15 (0% to 15% of portfolio per position)
- Based on probability and volatility
- Higher probability + lower volatility = larger position
- Capped at 15% for risk management

**Example**: `suggested_size: 0.0223` means allocate 2.23% of your portfolio to this stock.

### `current_price`
Latest closing price (in USD)

### `volatility`
**Annualized volatility** (20-day rolling window)

- Higher volatility = more risk = smaller position size
- Typical range: 0.10 (10%) to 0.50 (50%)
- TSLA shows 40% volatility (very high risk)
- AAPL shows 11% volatility (moderate risk)

### `key_feature_drivers`
**Top features driving the prediction** (currently null - feature importance not implemented)

This would show which market/sentiment features are most important for the prediction.

## Interpreting Your Results

Looking at your current outlook:

1. **AAPL** (Apple)
   - 57.4% probability of positive return
   - **Long** recommendation
   - Low volatility (11%) → reasonable position size (2.2%)

2. **MSFT** (Microsoft)
   - 54.8% probability (just below Long threshold)
   - **No Trade** (weak signal)
   - Moderate volatility (14%)

3. **GOOGL** (Google)
   - 57.4% probability
   - **Long** recommendation
   - Higher volatility (22%) → smaller position size

4. **AMZN** (Amazon)
   - 60.8% probability (highest)
   - **Long** recommendation
   - High volatility (23%) → moderate position size (3.2%)

5. **TSLA** (Tesla)
   - 54.8% probability
   - **No Trade** (very high volatility at 40%)
   - Volatility cap limits position size

6. **NVDA** (NVIDIA)
   - 57.4% probability
   - **Long** recommendation
   - High volatility (30%)

7. **META** (Meta/Facebook)
   - 54.9% probability
   - **No Trade** (just below threshold)

## Important Notes

- These are **probabilistic forecasts**, not guarantees
- The model predicts **directional probability** over 3-4 months, not exact prices
- **Position sizing** accounts for volatility (higher risk = smaller positions)
- The model was trained on historical data - past performance doesn't guarantee future results
- Always use proper risk management and diversification

## Next Steps

1. Review predictions in context of your portfolio
2. Consider diversification (don't put everything in tech)
3. Use position sizing suggestions for risk management
4. Monitor and re-evaluate periodically (model retrains as new data comes in)

