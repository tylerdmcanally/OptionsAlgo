# options_ai

Production-ready scanner that surfaces **strongest options plays** (CALL/PUT, expiry, strike, bid/ask/last, score) for a **10–30 day slightly OTM** style.  
- **Options data:** Yahoo (prefers `yahooquery`, falls back to `yfinance`)  
- **News:** Polygon News v2 (primary), Yahoo headlines (fallback) → **FinBERT** sentiment  
- **Universe:** S&P 500 + S&P 400 (live from Wikipedia)  
- **No position sizing** (you choose size manually)

## Install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
```bash
python src/main.py
```

## Configuration
Create a `.env` file:
```
POLYGON_API_KEY=your_polygon_api_key_here
```

## Features
- Real-time S&P 500 + S&P 400 universe from Wikipedia
- Options chain analysis with Greeks calculation
- FinBERT-powered sentiment analysis from news
- 10-30 day slightly OTM strategy focus
- Production-ready caching and error handling
- Comprehensive scoring algorithm for option plays
