"""
Sentiment Analysis using FinBERT and news sources
"""

import asyncio
import aiohttp
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from loguru import logger

# Import our Polygon manager
from polygon_manager import PolygonAPIManager

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    logger.warning("FinBERT dependencies not available, using simple sentiment")


class SentimentAnalyzer:
    """Analyzes sentiment from news sources using FinBERT"""
    
    def __init__(self, polygon_api_key: str):
        self.polygon_api_key = polygon_api_key
        self.lookback_days = 7
        self.max_articles = 20
        
        # Initialize Polygon manager
        self.polygon_manager = PolygonAPIManager(polygon_api_key)
        
        # Initialize FinBERT if available
        if FINBERT_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                self.finbert_ready = True
                logger.info("FinBERT model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load FinBERT: {e}")
                self.finbert_ready = False
        else:
            self.finbert_ready = False
    
    async def get_sentiment(self, symbol: str) -> float:
        """Get sentiment score for a symbol (-1 to 1)"""
        try:
            # Get news articles
            articles = await self._get_news_articles(symbol)
            
            if not articles:
                return 0.0  # Neutral if no news
            
            # Analyze sentiment
            if self.finbert_ready:
                sentiment_score = await self._analyze_with_finbert(articles)
            else:
                sentiment_score = self._simple_sentiment(articles)
            
            return sentiment_score
            
        except Exception as e:
            logger.warning(f"Error getting sentiment for {symbol}: {e}")
            return 0.0
    
    async def _get_news_articles(self, symbol: str) -> List[Dict]:
        """Get news articles from Polygon and Yahoo"""
        articles = []
        
        # Try Polygon first
        polygon_articles = await self._get_polygon_news(symbol)
        articles.extend(polygon_articles)
        
        # If not enough articles, try Yahoo
        if len(articles) < 5:
            yahoo_articles = await self._get_yahoo_news(symbol)
            articles.extend(yahoo_articles)
        
        return articles[:self.max_articles]
    
    async def _get_polygon_news(self, symbol: str) -> List[Dict]:
        """Get news from Polygon News API using the enhanced manager"""
        try:
            # Use the Polygon manager for news
            articles = await self.polygon_manager.get_ticker_news(
                symbol=symbol,
                days_back=self.lookback_days,
                limit=self.max_articles
            )
            
            # Convert to expected format if needed
            converted_articles = []
            for article in articles:
                converted_articles.append({
                    'title': article.get('title', ''),
                    'summary': article.get('description', '') or article.get('summary', ''),
                    'published_utc': article.get('published_utc', ''),
                    'url': article.get('article_url', ''),
                    'author': article.get('author', ''),
                    'tickers': article.get('tickers', []),
                    'keywords': article.get('keywords', [])
                })
            
            logger.debug(f"Retrieved {len(converted_articles)} articles from Polygon manager for {symbol}")
            return converted_articles
            
        except Exception as e:
            logger.debug(f"Error getting Polygon news for {symbol}: {e}")
            return []
    
    async def _get_polygon_news_direct(self, symbol: str) -> List[Dict]:
        """Fallback: Get news from Polygon News API via direct HTTP"""
        try:
            if not self.polygon_api_key or self.polygon_api_key == "your_polygon_api_key_here":
                return []
                
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)
            
            url = "https://api.polygon.io/v2/reference/news"
            params = {
                'ticker': symbol,
                'published_utc.gte': start_date.strftime('%Y-%m-%d'),
                'published_utc.lte': end_date.strftime('%Y-%m-%d'),
                'order': 'desc',
                'limit': self.max_articles,
                'apikey': self.polygon_api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get('results', [])
                        logger.debug(f"Retrieved {len(results)} articles from Polygon direct API for {symbol}")
                        return results
                    else:
                        logger.warning(f"Polygon API returned status {response.status} for {symbol}")
            
            return []
            
        except Exception as e:
            logger.debug(f"Error getting Polygon news: {e}")
            return []
    
    async def _get_yahoo_news(self, symbol: str) -> List[Dict]:
        """Get news from Yahoo Finance (fallback)"""
        try:
            # This is a simplified implementation
            # In practice, you'd scrape Yahoo Finance news or use their unofficial API
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            articles = []
            for item in news[:self.max_articles]:
                articles.append({
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'published_utc': item.get('providerPublishTime', '')
                })
            
            return articles
            
        except Exception as e:
            logger.debug(f"Error getting Yahoo news: {e}")
            return []
    
    async def _analyze_with_finbert(self, articles: List[Dict]) -> float:
        """Analyze sentiment using FinBERT"""
        try:
            sentiments = []
            
            for article in articles:
                text = f"{article.get('title', '')} {article.get('summary', '')}"
                text = text.strip()[:512]  # Limit text length
                
                if not text:
                    continue
                
                # Tokenize and analyze
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # FinBERT classes: negative, neutral, positive
                negative, neutral, positive = predictions[0].tolist()
                
                # Convert to -1 to 1 scale
                sentiment = positive - negative
                sentiments.append(sentiment)
            
            if sentiments:
                return sum(sentiments) / len(sentiments)
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Error with FinBERT analysis: {e}")
            return 0.0
    
    def _simple_sentiment(self, articles: List[Dict]) -> float:
        """Simple keyword-based sentiment analysis (fallback)"""
        try:
            positive_words = [
                'bullish', 'positive', 'growth', 'profit', 'strong', 'buy', 'upgrade',
                'beat', 'exceed', 'outperform', 'rally', 'surge', 'gain', 'rise'
            ]
            
            negative_words = [
                'bearish', 'negative', 'loss', 'weak', 'sell', 'downgrade',
                'miss', 'underperform', 'decline', 'fall', 'drop', 'crash'
            ]
            
            sentiment_scores = []
            
            for article in articles:
                text = f"{article.get('title', '')} {article.get('summary', '')}".lower()
                
                positive_count = sum(1 for word in positive_words if word in text)
                negative_count = sum(1 for word in negative_words if word in text)
                
                if positive_count + negative_count > 0:
                    score = (positive_count - negative_count) / (positive_count + negative_count)
                    sentiment_scores.append(score)
            
            if sentiment_scores:
                return sum(sentiment_scores) / len(sentiment_scores)
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Error with simple sentiment: {e}")
            return 0.0
