A comprehensive algorithmic trading bot with backtesting, paper trading, and live trading capabilities.

## Features

- 📈 **Multiple Trading Strategies**
  - Moving Average Crossover
  - Mean Reversion
  - Extensible strategy framework

- 🛡️ **Risk Management**
  - Position sizing
  - Stop loss / Take profit
  - Drawdown protection
  - Daily loss limits

- 📊 **Data & Analytics**
  - Real-time market data via Yahoo Finance
  - Technical indicators (RSI, Bollinger Bands, Moving Averages)
  - Performance tracking and metrics

- 🔍 **Trading Modes**
  - **Backtest**: Test strategies on historical data
  - **Paper Trading**: Real-time simulation without real money
  - **Live Trading**: Execute real trades (use with caution!)

- 📱 **Dashboard**
  - Real-time portfolio monitoring
  - Performance analytics
  - Trade history visualization

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd trading_bot

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `config/config.yaml` to customize:
- Trading symbols
- Strategy parameters
- Risk management settings
- Initial capital

### 3. Running the Bot

**Backtest (recommended first step):**
```bash
python main.py --mode backtest --period 6mo
```

**Paper Trading:**
```bash
python main.py --mode paper
```

**Live Trading (real money!):**
```bash
python main.py --mode live
```

**Launch Dashboard:**
```bash
python main.py --dashboard
```

## Testing Your Bot

### 1. Unit Tests
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src

# Run specific test
python -m pytest tests/test_strategies.py::TestMovingAverageCrossoverStrategy::test_buy_signal_generation
```

### 2. Strategy Backtesting
```bash
# Quick backtest (3 months)
python main.py --mode backtest --period 3mo

# Extended backtest (1 year)
python main.py --mode backtest --period 1y

# Custom config
python main.py --mode backtest --config config/custom_config.yaml
```

### 3. Paper Trading Test
```bash
# Start paper trading
python main.py --mode paper

# Monitor in dashboard (separate terminal)
python main.py --dashboard
```

## Project Structure

```
trading_bot/
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py      # Market data collection
│   ├── strategies.py        # Trading strategies  
│   ├── risk_manager.py      # Risk management
│   ├── portfolio.py         # Portfolio tracking
│   ├── bot.py              # Main trading bot
│   └── dashboard.py        # Performance dashboard
├── config/
│   └── config.yaml         # Configuration settings
├── data/
│   └── trades.csv          # Trade history
├── tests/
│   └── test_strategies.py  # Unit tests
├── logs/                   # Log files
├── requirements.txt
├── README.md
└── main.py                 # Entry point
```

## Configuration

Key configuration options in `config/config.yaml`:

```yaml
trading:
  symbols: ["AAPL", "MSFT", "GOOGL"]  # Stocks to trade
  initial_capital: 100000              # Starting capital
  strategy: "MovingAverageCrossover"   # Strategy to use

risk_management:
  max_position_size: 0.1      # Max 10% per position
  stop_loss_pct: 0.05         # 5% stop loss
  take_profit_pct: 0.10       # 10% take profit
```

## Safety Guidelines

### Before Live Trading:

1. **Thorough Backtesting**
   ```bash
   python main.py --mode backtest --period 1y
   ```

2. **Extended Paper Trading**
   ```bash
   python main.py --mode paper
   # Let it run for at least 1-2 weeks
   ```

3. **Start Small**
   - Use minimal capital initially
   - Test with 1-2 symbols first
   - Monitor closely for the first few days

4. **Risk Management**
   - Never risk more than you can afford to lose
   - Set conservative position sizes
   - Use stop losses
   - Monitor drawdown limits

## Common Issues & Solutions

### Data Issues
```bash
# If you get data fetch errors:
pip install --upgrade yfinance

# Check internet connection and try again
```

### Performance Issues
```bash
# For slow backtests, reduce data:
# Edit config.yaml - use fewer symbols or shorter periods
```

### Import Errors
```bash
# Make sure you're in the project directory:
cd trading_bot
python main.py --mode backtest
```

## Extending the Bot

### Adding New Strategies

1. Create new strategy class in `src/strategies.py`:
```python
class MyCustomStrategy(BaseStrategy):
    def generate_signals(self, symbol, data, indicators):
        # Your strategy logic here
        return {'action': 'BUY', 'confidence': 70, 'reason': 'Custom signal'}
```

2. Register in strategy factory:
```python
def create_strategy(strategy_name, config):
    strategies = {
        'MovingAverageCrossover': MovingAverageCrossoverStrategy,
        'MeanReversion': MeanReversionStrategy,
        'MyCustom': MyCustomStrategy,  # Add here
    }
```

3. Update config:
```yaml
trading:
  strategy: "MyCustom"
```

### Adding New Indicators

Add to `data_fetcher.py`:
```python
def get_technical_indicators(self, symbol, window=20):
    # Add your custom indicator
    my_indicator = calculate_my_indicator(hist_data)
    return {
        'sma': sma,
        'rsi': rsi,
        'my_indicator': my_indicator  # Add here
    }
```

## Support

- Check logs in `logs/trading_bot.log`
- Review trade history in `data/trades.csv`
- Use dashboard for real-time monitoring
- Run tests to verify functionality

## Disclaimer

This software is for educational purposes. Trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.
                