"""
Options Scanner - Multi-source options chain analysis
"""

import asyncio
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from loguru import logger

try:
    import yahooquery as yq
    YAHOOQUERY_AVAILABLE = True
except ImportError:
    YAHOOQUERY_AVAILABLE = False
    logger.warning("yahooquery not available, falling back to yfinance")

try:
    from polygon import RESTClient
    POLYGON_CLIENT_AVAILABLE = True
except ImportError:
    POLYGON_CLIENT_AVAILABLE = False


class OptionsScanner:
    """Scans options chains for the best plays"""
    
    def __init__(self, polygon_api_key: Optional[str] = None):
        self.min_days = 10
        self.max_days = 30
        self.min_volume = 50
        self.min_oi = 100
        self.otm_min = 0.02  # 2% OTM minimum
        self.otm_max = 0.15  # 15% OTM maximum
        
        # Initialize Polygon client for future options data
        self.polygon_client = None
        if POLYGON_CLIENT_AVAILABLE and polygon_api_key and polygon_api_key != "your_polygon_api_key_here":
            try:
                self.polygon_client = RESTClient(api_key=polygon_api_key)
                logger.info("Polygon client initialized for options scanner")
            except Exception as e:
                logger.warning(f"Failed to initialize Polygon client for options: {e}")
    
    async def scan_symbol(self, symbol: str) -> List[Dict]:
        """Scan options for a single symbol"""
        try:
            # Get current stock price
            stock_price = await self._get_stock_price(symbol)
            if not stock_price:
                return []
            
            # Get options chain (currently Yahoo-based, Polygon integration planned)
            options_data = await self._get_options_chain(symbol)
            if not options_data:
                return []
            
            # Filter and score options
            plays = []
            for option in options_data:
                if self._is_valid_option(option, stock_price):
                    play = self._create_play(option, symbol, stock_price)
                    plays.append(play)
            
            return plays
            
        except Exception as e:
            logger.warning(f"Error scanning {symbol}: {e}")
            return []
    
    async def _get_stock_price(self, symbol: str) -> Optional[float]:
        """Get current stock price"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('currentPrice') or info.get('regularMarketPrice')
        except:
            return None
    
    async def _get_options_chain(self, symbol: str) -> List[Dict]:
        """Get options chain data"""
        try:
            if YAHOOQUERY_AVAILABLE:
                return await self._get_options_yahooquery(symbol)
            else:
                return await self._get_options_yfinance(symbol)
        except:
            return []
    
    async def _get_options_yahooquery(self, symbol: str) -> List[Dict]:
        """Get options using yahooquery (preferred)"""
        try:
            ticker = yq.Ticker(symbol)
            options = ticker.option_chain
            
            if not isinstance(options, pd.DataFrame) or options.empty:
                return []
            
            return options.to_dict('records')
        except:
            return []
    
    async def _get_options_yfinance(self, symbol: str) -> List[Dict]:
        """Get options using yfinance (fallback)"""
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            
            if not expirations:
                return []
            
            all_options = []
            for exp in expirations[:5]:  # Limit to first 5 expirations
                try:
                    chain = ticker.option_chain(exp)
                    
                    # Add calls
                    for _, row in chain.calls.iterrows():
                        option = row.to_dict()
                        option['expiry'] = exp
                        option['type'] = 'CALL'
                        all_options.append(option)
                    
                    # Add puts
                    for _, row in chain.puts.iterrows():
                        option = row.to_dict()
                        option['expiry'] = exp
                        option['type'] = 'PUT'
                        all_options.append(option)
                        
                except Exception as e:
                    logger.debug(f"Error getting {exp} chain for {symbol}: {e}")
                    continue
            
            return all_options
            
        except Exception as e:
            logger.debug(f"Error getting options for {symbol}: {e}")
            return []
    
    def _is_valid_option(self, option: Dict, stock_price: float) -> bool:
        """Check if option meets our criteria"""
        try:
            # Check expiry date
            expiry_str = option.get('expiry', '')
            if isinstance(expiry_str, str):
                expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d')
            else:
                expiry_date = expiry_str
            
            days_to_expiry = (expiry_date - datetime.now()).days
            if not (self.min_days <= days_to_expiry <= self.max_days):
                return False
            
            # Check volume and open interest
            volume = option.get('volume', 0) or 0
            open_interest = option.get('openInterest', 0) or 0
            
            if volume < self.min_volume or open_interest < self.min_oi:
                return False
            
            # Check if option is appropriately OTM
            strike = option.get('strike', 0)
            option_type = option.get('type', '').upper()
            
            if option_type == 'CALL':
                otm_percent = (strike - stock_price) / stock_price
            elif option_type == 'PUT':
                otm_percent = (stock_price - strike) / stock_price
            else:
                return False
            
            return self.otm_min <= otm_percent <= self.otm_max
            
        except Exception as e:
            logger.debug(f"Error validating option: {e}")
            return False
    
    def _create_play(self, option: Dict, symbol: str, stock_price: float) -> Dict:
        """Create a standardized play dictionary"""
        return {
            'symbol': symbol,
            'type': option.get('type', '').upper(),
            'strike': option.get('strike', 0),
            'expiry': option.get('expiry', ''),
            'bid': option.get('bid', 0) or 0,
            'ask': option.get('ask', 0) or 0,
            'last': option.get('lastPrice', 0) or 0,
            'volume': option.get('volume', 0) or 0,
            'open_interest': option.get('openInterest', 0) or 0,
            'implied_volatility': option.get('impliedVolatility', 0) or 0,
            'stock_price': stock_price,
            'timestamp': datetime.now()
        }
