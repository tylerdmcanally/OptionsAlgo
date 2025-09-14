"""
Universe Management - S&P 500 + S&P 400 from Wikipedia
"""

import asyncio
import pandas as pd
import requests
from typing import List, Set
from loguru import logger


class UniverseManager:
    """Manages the stock universe (S&P 500 + S&P 400)"""
    
    def __init__(self):
        self.sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        self.sp400_url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    
    async def get_universe(self) -> List[str]:
        """Get combined S&P 500 + S&P 400 symbols"""
        try:
            # Get both lists concurrently
            sp500_task = asyncio.create_task(self._get_sp500_symbols())
            sp400_task = asyncio.create_task(self._get_sp400_symbols())
            
            sp500_symbols, sp400_symbols = await asyncio.gather(sp500_task, sp400_task)
            
            # Combine and deduplicate
            all_symbols: Set[str] = set(sp500_symbols) | set(sp400_symbols)
            
            # Filter out problematic symbols
            filtered_symbols = [s for s in all_symbols if self._is_valid_symbol(s)]
            
            logger.info(f"Loaded {len(filtered_symbols)} symbols ({len(sp500_symbols)} S&P 500 + {len(sp400_symbols)} S&P 400)")
            
            return sorted(filtered_symbols)
            
        except Exception as e:
            logger.error(f"Failed to load universe: {e}")
            # Fallback to a smaller test universe
            return ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META", "AMZN", "NFLX"]
    
    async def _get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 symbols from Wikipedia"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, pd.read_html, self.sp500_url)
            
            # The first table contains the companies
            sp500_df = df[0]
            symbols = sp500_df['Symbol'].tolist()
            
            return [self._clean_symbol(s) for s in symbols if pd.notna(s)]
            
        except Exception as e:
            logger.warning(f"Failed to load S&P 500: {e}")
            return []
    
    async def _get_sp400_symbols(self) -> List[str]:
        """Get S&P 400 symbols from Wikipedia"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, pd.read_html, self.sp400_url)
            
            # The first table contains the companies
            sp400_df = df[0]
            symbols = sp400_df['Symbol'].tolist()
            
            return [self._clean_symbol(s) for s in symbols if pd.notna(s)]
            
        except Exception as e:
            logger.warning(f"Failed to load S&P 400: {e}")
            return []
    
    def _clean_symbol(self, symbol: str) -> str:
        """Clean and normalize symbol"""
        return str(symbol).strip().upper().replace('.', '-')
    
    def _is_valid_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid for options trading"""
        if not symbol or len(symbol) > 5:
            return False
        
        # Skip certain types of symbols
        invalid_patterns = ['/', '=', '^', '.', 'BRK.B']
        return not any(pattern in symbol for pattern in invalid_patterns)
