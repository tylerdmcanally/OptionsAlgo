"""
Polygon API Manager - Enhanced integration with Polygon.io
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
from loguru import logger

try:
    from polygon import RESTClient
    from polygon.rest.models import TickerNews, TickerSnapshot, DailyOpenCloseAgg
    POLYGON_CLIENT_AVAILABLE = True
except ImportError:
    POLYGON_CLIENT_AVAILABLE = False
    logger.warning("Polygon client not available")


class PolygonAPIManager:
    """Enhanced Polygon API integration for market data"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self.ready = False
        
        if POLYGON_CLIENT_AVAILABLE and api_key and api_key != "your_polygon_api_key_here":
            try:
                self.client = RESTClient(api_key=api_key)
                self.ready = True
                logger.info("Polygon API Manager initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Polygon client: {e}")
                self.ready = False
        else:
            logger.warning("Polygon API key not configured or client not available")
    
    async def get_ticker_news(self, symbol: str, days_back: int = 7, limit: int = 20) -> List[Dict]:
        """Get news for a specific ticker"""
        if not self.ready:
            return []
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            loop = asyncio.get_event_loop()
            news_data = await loop.run_in_executor(
                None,
                lambda: list(self.client.list_ticker_news(
                    ticker=symbol,
                    published_utc_gte=start_date.strftime('%Y-%m-%d'),
                    published_utc_lte=end_date.strftime('%Y-%m-%d'),
                    order='desc',
                    limit=limit
                ))
            )
            
            articles = []
            for article in news_data:
                articles.append({
                    'id': getattr(article, 'id', ''),
                    'title': getattr(article, 'title', ''),
                    'description': getattr(article, 'description', ''),
                    'summary': getattr(article, 'summary', ''),
                    'published_utc': getattr(article, 'published_utc', ''),
                    'article_url': getattr(article, 'article_url', ''),
                    'author': getattr(article, 'author', ''),
                    'tickers': getattr(article, 'tickers', []),
                    'amp_url': getattr(article, 'amp_url', ''),
                    'image_url': getattr(article, 'image_url', ''),
                    'keywords': getattr(article, 'keywords', [])
                })
            
            logger.debug(f"Retrieved {len(articles)} news articles for {symbol}")
            return articles
            
        except Exception as e:
            logger.warning(f"Error fetching news for {symbol}: {e}")
            return []
    
    async def get_market_snapshot(self, symbol: str) -> Optional[Dict]:
        """Get current market snapshot for a ticker"""
        if not self.ready:
            return None
        
        try:
            loop = asyncio.get_event_loop()
            snapshot = await loop.run_in_executor(
                None,
                lambda: self.client.get_snapshot_ticker(ticker=symbol, market_type='stocks')
            )
            
            if snapshot:
                return {
                    'ticker': getattr(snapshot, 'ticker', symbol),
                    'updated': getattr(snapshot, 'updated', ''),
                    'session': {
                        'change': getattr(snapshot.session, 'change', 0) if hasattr(snapshot, 'session') else 0,
                        'change_percent': getattr(snapshot.session, 'change_percent', 0) if hasattr(snapshot, 'session') else 0,
                        'early_trading_change': getattr(snapshot.session, 'early_trading_change', 0) if hasattr(snapshot, 'session') else 0,
                        'early_trading_change_percent': getattr(snapshot.session, 'early_trading_change_percent', 0) if hasattr(snapshot, 'session') else 0,
                        'close': getattr(snapshot.session, 'close', 0) if hasattr(snapshot, 'session') else 0,
                        'high': getattr(snapshot.session, 'high', 0) if hasattr(snapshot, 'session') else 0,
                        'low': getattr(snapshot.session, 'low', 0) if hasattr(snapshot, 'session') else 0,
                        'open': getattr(snapshot.session, 'open', 0) if hasattr(snapshot, 'session') else 0,
                        'previous_close': getattr(snapshot.session, 'previous_close', 0) if hasattr(snapshot, 'session') else 0
                    },
                    'last_quote': {
                        'timeframe': getattr(snapshot.last_quote, 'timeframe', '') if hasattr(snapshot, 'last_quote') else '',
                        'timestamp': getattr(snapshot.last_quote, 'timestamp', 0) if hasattr(snapshot, 'last_quote') else 0,
                        'bid': getattr(snapshot.last_quote, 'bid', 0) if hasattr(snapshot, 'last_quote') else 0,
                        'bid_size': getattr(snapshot.last_quote, 'bid_size', 0) if hasattr(snapshot, 'last_quote') else 0,
                        'ask': getattr(snapshot.last_quote, 'ask', 0) if hasattr(snapshot, 'last_quote') else 0,
                        'ask_size': getattr(snapshot.last_quote, 'ask_size', 0) if hasattr(snapshot, 'last_quote') else 0,
                        'exchange': getattr(snapshot.last_quote, 'exchange', 0) if hasattr(snapshot, 'last_quote') else 0
                    },
                    'last_trade': {
                        'timeframe': getattr(snapshot.last_trade, 'timeframe', '') if hasattr(snapshot, 'last_trade') else '',
                        'timestamp': getattr(snapshot.last_trade, 'timestamp', 0) if hasattr(snapshot, 'last_trade') else 0,
                        'price': getattr(snapshot.last_trade, 'price', 0) if hasattr(snapshot, 'last_trade') else 0,
                        'size': getattr(snapshot.last_trade, 'size', 0) if hasattr(snapshot, 'last_trade') else 0,
                        'exchange': getattr(snapshot.last_trade, 'exchange', 0) if hasattr(snapshot, 'last_trade') else 0
                    }
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error fetching market snapshot for {symbol}: {e}")
            return None
    
    async def get_daily_bars(self, symbol: str, days_back: int = 30) -> List[Dict]:
        """Get daily OHLCV bars for a ticker"""
        if not self.ready:
            return []
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            loop = asyncio.get_event_loop()
            bars = await loop.run_in_executor(
                None,
                lambda: list(self.client.list_aggs(
                    ticker=symbol,
                    multiplier=1,
                    timespan='day',
                    from_=start_date.strftime('%Y-%m-%d'),
                    to=end_date.strftime('%Y-%m-%d'),
                    adjusted=True,
                    sort='asc'
                ))
            )
            
            bar_data = []
            for bar in bars:
                bar_data.append({
                    'timestamp': getattr(bar, 'timestamp', 0),
                    'open': getattr(bar, 'open', 0),
                    'high': getattr(bar, 'high', 0),
                    'low': getattr(bar, 'low', 0),
                    'close': getattr(bar, 'close', 0),
                    'volume': getattr(bar, 'volume', 0),
                    'vwap': getattr(bar, 'vwap', 0),
                    'transactions': getattr(bar, 'transactions', 0)
                })
            
            logger.debug(f"Retrieved {len(bar_data)} daily bars for {symbol}")
            return bar_data
            
        except Exception as e:
            logger.warning(f"Error fetching daily bars for {symbol}: {e}")
            return []
    
    async def get_options_contracts(self, symbol: str) -> List[Dict]:
        """Get options contracts for a ticker (if available in your Polygon plan)"""
        if not self.ready:
            return []
        
        try:
            # Note: Options data requires higher-tier Polygon subscription
            # This is a placeholder for future implementation
            logger.info(f"Options contract lookup for {symbol} - requires Polygon options data subscription")
            return []
            
        except Exception as e:
            logger.warning(f"Error fetching options contracts for {symbol}: {e}")
            return []
    
    async def check_api_status(self) -> Dict:
        """Check Polygon API status and usage"""
        if not self.ready:
            return {'status': 'not_ready', 'message': 'Polygon client not initialized'}
        
        try:
            # Simple API call to check status
            loop = asyncio.get_event_loop()
            snapshot = await loop.run_in_executor(
                None,
                lambda: self.client.get_snapshot_ticker(ticker='AAPL', market_type='stocks')
            )
            
            if snapshot:
                return {
                    'status': 'ready',
                    'message': 'Polygon API is accessible',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Failed to fetch test data',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'API error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
