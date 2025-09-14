#!/usr/bin/env python3
"""
Options AI Scanner - Main Entry Point
Production-ready scanner for strongest options plays
"""

import asyncio
import sys
from pathlib import Path
from loguru import logger
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config import Config
from universe import UniverseManager
from options_scanner import OptionsScanner
from sentiment_analyzer import SentimentAnalyzer
from scoring_engine import ScoringEngine


async def main():
    """Main scanner execution"""
    logger.info("🚀 Starting Options AI Scanner")
    start_time = datetime.now()
    
    try:
        # Initialize components
        config = Config()
        universe_manager = UniverseManager()
        sentiment_analyzer = SentimentAnalyzer(config.polygon_api_key)
        options_scanner = OptionsScanner(config.polygon_api_key)
        scoring_engine = ScoringEngine()
        
        # Get universe (S&P 500 + S&P 400)
        logger.info("📊 Loading stock universe...")
        symbols = await universe_manager.get_universe()
        logger.info(f"📈 Loaded {len(symbols)} symbols")
        
        # Scan options for each symbol
        logger.info("🔍 Scanning options chains...")
        all_plays = []
        
        for i, symbol in enumerate(symbols[:50]):  # Limit for testing
            try:
                logger.info(f"Scanning {symbol} ({i+1}/{min(len(symbols), 50)})")
                
                # Get sentiment
                sentiment_score = await sentiment_analyzer.get_sentiment(symbol)
                
                # Scan options
                options_plays = await options_scanner.scan_symbol(symbol)
                
                # Score each play
                for play in options_plays:
                    play['sentiment_score'] = sentiment_score
                    play['final_score'] = scoring_engine.calculate_score(play)
                    all_plays.append(play)
                    
            except Exception as e:
                logger.warning(f"Error scanning {symbol}: {e}")
                continue
        
        # Sort by score and display top plays
        top_plays = sorted(all_plays, key=lambda x: x['final_score'], reverse=True)[:20]
        
        logger.info("🏆 TOP OPTIONS PLAYS:")
        logger.info("=" * 80)
        
        for i, play in enumerate(top_plays, 1):
            logger.info(
                f"{i:2d}. {play['symbol']} {play['type']} "
                f"${play['strike']:.2f} {play['expiry']} | "
                f"Score: {play['final_score']:.2f} | "
                f"Bid/Ask: ${play['bid']:.2f}/${play['ask']:.2f} | "
                f"Last: ${play['last']:.2f}"
            )
        
        elapsed = datetime.now() - start_time
        logger.info(f"✅ Scan completed in {elapsed.total_seconds():.1f}s")
        
    except Exception as e:
        logger.error(f"❌ Scanner failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO"
    )
    
    # Run scanner
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
