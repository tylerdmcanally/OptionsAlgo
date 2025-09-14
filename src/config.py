"""
Configuration management for Options AI Scanner
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration settings for the options scanner"""
    
    def __init__(self):
        # API Keys
        self.polygon_api_key: Optional[str] = os.getenv('POLYGON_API_KEY')
        
        # Scanner Settings
        self.min_days_to_expiry: int = 10
        self.max_days_to_expiry: int = 30
        self.min_volume: int = 50
        self.min_open_interest: int = 100
        self.otm_range: tuple = (0.02, 0.15)  # 2% to 15% OTM
        
        # Sentiment Settings
        self.sentiment_lookback_days: int = 7
        self.max_news_articles: int = 20
        
        # Scoring Weights
        self.weights = {
            'volume': 0.25,
            'open_interest': 0.20,
            'sentiment': 0.20,
            'volatility': 0.15,
            'moneyness': 0.10,
            'time_to_expiry': 0.10
        }
        
        # Cache Settings
        self.cache_dir: Path = Path(__file__).parent.parent / 'cache'
        self.cache_expiry_minutes: int = 30
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration settings"""
        if not self.polygon_api_key:
            raise ValueError("POLYGON_API_KEY is required in .env file")
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total_weight}")
