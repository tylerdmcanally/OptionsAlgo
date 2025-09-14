# 🚀 Options AI Trading System

A comprehensive options trading system powered by AI sentiment analysis, real-time market data, and advanced backtesting capabilities.

## 🎯 Core Features

- **AI Sentiment Analysis**: FinBERT-powered news sentiment analysis
- **Options Scanner**: Real-time options opportunity discovery
- **Strategy Framework**: Multiple trading strategies with risk management
- **Backtesting Engine**: Comprehensive historical performance testing
- **Live Dashboard**: Real-time portfolio monitoring via Streamlit
- **Risk Management**: Advanced position sizing and risk controls

## 🏗️ Project Structure

```
options_ai/
├── src/
│   ├── main.py                    # Main entry point
│   ├── config.py                  # Configuration management
│   ├── polygon_manager.py         # Market data API integration
│   ├── sentiment_analyzer.py      # AI sentiment analysis
│   ├── options_scanner.py         # Options opportunity scanner
│   ├── options_pricing.py         # Options pricing models
│   ├── scoring_engine.py          # Opportunity scoring system
│   ├── strategy_framework.py      # Trading strategies
│   ├── backtest_engine.py         # Backtesting framework
│   ├── streamlit_integration.py   # Dashboard integration
│   └── universe.py               # Stock universe definition
├── dashboard.py                   # Streamlit dashboard UI
├── requirements.txt              # Python dependencies
└── .env.example                 # Environment variables template
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd options_ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
POLYGON_API_KEY=your_polygon_api_key_here
```

### 3. Run the System

**Start the Dashboard:**
```bash
python -m streamlit run dashboard.py --server.port 8501
```

**Run Backtesting:**
```bash
cd src
python main.py
```

## 🎛️ Dashboard Integration

The system includes a comprehensive Streamlit dashboard for real-time monitoring:

### Push Opportunities
```python
from streamlit_integration import StreamlitIntegration
dashboard = StreamlitIntegration()

opportunities = [{
    'symbol': 'AAPL',
    'contract_type': 'CALL', 
    'strike_price': 175.0,
    'expiration_date': '2025-01-17',
    'premium': 2.50,
    'delta': 0.65
}]

dashboard.push_opportunities_to_dashboard(opportunities)
```

### Track Trades
```python
trade = {
    'symbol': 'AAPL',
    'contract_type': 'CALL',
    'action': 'BUY',
    'quantity': 2,
    'price': 2.45,
    'date': '2025-09-14'
}

dashboard.push_trade_to_dashboard(trade)
```

### Monitor Portfolio
```python
data = dashboard.get_dashboard_data()
print(f"Current P&L: ${data['pnl_breakdown']['total_unrealized_pnl']:,.2f}")

# Use in trading decisions
if data['pnl_breakdown']['total_unrealized_pnl'] > 1000:
    # Trigger profit-taking strategy
    pass
```

## 📊 Sentiment Analysis Integration

The system integrates AI-powered sentiment analysis:

```python
from sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer(polygon_api_key="your_key")

# Get sentiment for a symbol
sentiment_score = await analyzer.get_sentiment("AAPL")

# Push to dashboard
analyzer.push_opportunities_to_dashboard(opportunities)
analyzer.push_trade_to_dashboard(trade)

# Check profit-taking triggers
should_take_profits = analyzer.check_profit_taking_trigger(threshold=1000)
```

## 🎯 Trading Strategies

The system includes multiple trading strategies:

- **SentimentMomentum**: News sentiment-driven momentum trading
- **VolatilityCapture**: Volatility-based options strategies
- **MeanReversion**: Mean reversion opportunities

### Strategy Configuration
```python
# Configure in src/config.py
STRATEGY_CONFIG = {
    'max_positions': 3,
    'position_size_pct': 0.02,
    'sentiment_threshold': 0.5,
    'confidence_threshold': 0.45
}
```

## 🔧 Risk Management

Built-in risk management features:

- **Position Sizing**: Maximum 2% of portfolio per position
- **Stop Losses**: Automatic exit conditions
- **Time Decay Protection**: Exits before excessive theta decay
- **Drawdown Controls**: Maximum drawdown limits

## 📈 Backtesting

Comprehensive backtesting framework:

```python
from backtest_engine import OptionsBacktester

backtester = OptionsBacktester(config=backtest_config)
results = await backtester.run_backtest(market_data)

# Analyze results
print(f"Total Return: {results['total_return']:.2%}")
print(f"Win Rate: {results['win_rate']:.1%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

## 🛠️ Dependencies

Key dependencies include:
- `streamlit` - Dashboard UI
- `plotly` - Interactive charts
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `aiohttp` - Async HTTP requests
- `transformers` - FinBERT sentiment analysis
- `loguru` - Logging

## 📝 Environment Variables

Required environment variables:

```env
POLYGON_API_KEY=your_polygon_api_key_here
LOG_LEVEL=INFO
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results.
