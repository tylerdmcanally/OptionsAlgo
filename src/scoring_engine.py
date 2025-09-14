"""
Scoring Engine for Options Plays
"""

import math
from datetime import datetime
from typing import Dict
from loguru import logger


class ScoringEngine:
    """Calculates composite scores for options plays"""
    
    def __init__(self):
        self.weights = {
            'volume': 0.25,
            'open_interest': 0.20,
            'sentiment': 0.20,
            'volatility': 0.15,
            'moneyness': 0.10,
            'time_to_expiry': 0.10
        }
    
    def calculate_score(self, play: Dict) -> float:
        """Calculate composite score for an options play (0-100)"""
        try:
            scores = {}
            
            # Volume score (0-100)
            scores['volume'] = self._score_volume(play.get('volume', 0))
            
            # Open interest score (0-100)
            scores['open_interest'] = self._score_open_interest(play.get('open_interest', 0))
            
            # Sentiment score (0-100)
            scores['sentiment'] = self._score_sentiment(play.get('sentiment_score', 0))
            
            # Volatility score (0-100)
            scores['volatility'] = self._score_volatility(play.get('implied_volatility', 0))
            
            # Moneyness score (0-100)
            scores['moneyness'] = self._score_moneyness(play)
            
            # Time to expiry score (0-100)
            scores['time_to_expiry'] = self._score_time_to_expiry(play.get('expiry', ''))
            
            # Calculate weighted composite score
            composite_score = sum(
                scores[component] * self.weights[component]
                for component in scores
            )
            
            return round(composite_score, 2)
            
        except Exception as e:
            logger.debug(f"Error calculating score: {e}")
            return 0.0
    
    def _score_volume(self, volume: int) -> float:
        """Score based on option volume (0-100)"""
        if volume <= 0:
            return 0.0
        
        # Logarithmic scale: 50 volume = 50 points, 500 volume = 75 points, 5000+ = 100 points
        score = 30 + (20 * math.log10(max(1, volume)))
        return min(100.0, max(0.0, score))
    
    def _score_open_interest(self, open_interest: int) -> float:
        """Score based on open interest (0-100)"""
        if open_interest <= 0:
            return 0.0
        
        # Logarithmic scale: 100 OI = 50 points, 1000 OI = 75 points, 10000+ = 100 points
        score = 25 + (25 * math.log10(max(1, open_interest)))
        return min(100.0, max(0.0, score))
    
    def _score_sentiment(self, sentiment: float) -> float:
        """Score based on sentiment (-1 to 1 → 0-100)"""
        # Convert -1 to 1 range to 0-100 scale
        # Neutral (0) = 50 points, very positive (1) = 100 points, very negative (-1) = 0 points
        score = 50 + (sentiment * 50)
        return min(100.0, max(0.0, score))
    
    def _score_volatility(self, implied_vol: float) -> float:
        """Score based on implied volatility (0-100)"""
        if implied_vol <= 0:
            return 50.0  # Neutral if no IV data
        
        # Sweet spot for IV is around 20-40%
        # Too low (<10%) or too high (>100%) gets lower scores
        if 0.15 <= implied_vol <= 0.50:  # 15-50% IV is ideal
            score = 100.0
        elif 0.10 <= implied_vol < 0.15 or 0.50 < implied_vol <= 0.80:  # Decent range
            score = 75.0
        elif 0.05 <= implied_vol < 0.10 or 0.80 < implied_vol <= 1.50:  # Acceptable
            score = 50.0
        else:  # Too extreme
            score = 25.0
        
        return score
    
    def _score_moneyness(self, play: Dict) -> float:
        """Score based on how close option is to ideal OTM range (0-100)"""
        try:
            strike = play.get('strike', 0)
            stock_price = play.get('stock_price', 0)
            option_type = play.get('type', '').upper()
            
            if not strike or not stock_price:
                return 50.0  # Neutral if missing data
            
            # Calculate OTM percentage
            if option_type == 'CALL':
                otm_percent = (strike - stock_price) / stock_price
            elif option_type == 'PUT':
                otm_percent = (stock_price - strike) / stock_price
            else:
                return 50.0
            
            # Ideal range is 3-8% OTM
            if 0.03 <= otm_percent <= 0.08:
                score = 100.0
            elif 0.02 <= otm_percent < 0.03 or 0.08 < otm_percent <= 0.12:
                score = 80.0
            elif 0.01 <= otm_percent < 0.02 or 0.12 < otm_percent <= 0.15:
                score = 60.0
            else:
                score = 30.0
            
            return score
            
        except Exception as e:
            logger.debug(f"Error scoring moneyness: {e}")
            return 50.0
    
    def _score_time_to_expiry(self, expiry: str) -> float:
        """Score based on time to expiry (0-100)"""
        try:
            if isinstance(expiry, str):
                expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
            else:
                expiry_date = expiry
            
            days_to_expiry = (expiry_date - datetime.now()).days
            
            # Ideal range is 14-21 days
            if 14 <= days_to_expiry <= 21:
                score = 100.0
            elif 10 <= days_to_expiry < 14 or 21 < days_to_expiry <= 30:
                score = 80.0
            elif 7 <= days_to_expiry < 10 or 30 < days_to_expiry <= 45:
                score = 60.0
            else:
                score = 30.0
            
            return score
            
        except Exception as e:
            logger.debug(f"Error scoring time to expiry: {e}")
            return 50.0
