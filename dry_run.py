#!/usr/bin/env python3
"""
Dry Run of Options AI Scanner with Recent Data
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from config import Config
from sentiment_analyzer import SentimentAnalyzer
from options_scanner import OptionsScanner
from scoring_engine import ScoringEngine
from polygon_manager import PolygonAPIManager


# Test symbols with good options liquidity
TEST_SYMBOLS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft  
    "TSLA",  # Tesla
    "NVDA",  # NVIDIA
    "GOOGL", # Google
    "META",  # Meta
    "AMZN",  # Amazon
    "SPY",   # S&P 500 ETF
    "QQQ",   # NASDAQ ETF
    "IWM"    # Russell 2000 ETF
]


async def dry_run_scanner():
    """Run the options scanner with real recent data"""
    logger.info("🚀 Starting Options AI Scanner - DRY RUN")
    logger.info("📅 Using most recent available data")
    start_time = datetime.now()
    
    try:
        # Initialize components
        logger.info("🔧 Initializing components...")
        config = Config()
        sentiment_analyzer = SentimentAnalyzer(config.polygon_api_key)
        options_scanner = OptionsScanner(config.polygon_api_key)
        scoring_engine = ScoringEngine()
        polygon_manager = PolygonAPIManager(config.polygon_api_key)
        
        # Check API status
        logger.info("📡 Checking API connectivity...")
        api_status = await polygon_manager.check_api_status()
        logger.info(f"Polygon API Status: {api_status['status']}")
        
        logger.info(f"📊 Scanning {len(TEST_SYMBOLS)} liquid symbols for options plays...")
        logger.info("=" * 80)
        
        all_plays = []
        
        for i, symbol in enumerate(TEST_SYMBOLS, 1):
            try:
                logger.info(f"🔍 [{i}/{len(TEST_SYMBOLS)}] Analyzing {symbol}...")
                
                # Get recent news sentiment
                logger.info(f"  📰 Fetching news sentiment...")
                sentiment_score = await sentiment_analyzer.get_sentiment(symbol)
                logger.info(f"  📊 Sentiment: {sentiment_score:.3f} {'📈' if sentiment_score > 0.1 else '📉' if sentiment_score < -0.1 else '➡️'}")
                
                # Get market data
                logger.info(f"  💹 Getting market snapshot...")
                market_data = await polygon_manager.get_market_snapshot(symbol)
                
                current_price = None
                change_percent = None
                if market_data:
                    current_price = market_data.get('last_trade', {}).get('price')
                    change_percent = market_data.get('session', {}).get('change_percent')
                    logger.info(f"  💰 Price: ${current_price:.2f} ({change_percent:+.2f}%)" if current_price else "  💰 Price: Not available (requires paid plan)")
                else:
                    logger.info(f"  💰 Market data: Not available (requires paid subscription)")
                
                # Get historical bars for technical analysis
                logger.info(f"  📈 Fetching recent price history...")
                bars = await polygon_manager.get_daily_bars(symbol, days_back=5)
                if bars:
                    latest_bar = bars[-1]
                    logger.info(f"  📊 Latest: O:{latest_bar['open']:.2f} H:{latest_bar['high']:.2f} L:{latest_bar['low']:.2f} C:{latest_bar['close']:.2f}")
                    current_price = current_price or latest_bar['close']  # Use close if no real-time price
                
                # Scan options (this will use Yahoo Finance)
                logger.info(f"  ⚙️  Scanning options chains...")
                options_plays = await options_scanner.scan_symbol(symbol)
                logger.info(f"  ✅ Found {len(options_plays)} qualifying options")
                
                # Score each play
                for play in options_plays:
                    play['sentiment_score'] = sentiment_score
                    play['market_data'] = {
                        'current_price': current_price,
                        'change_percent': change_percent,
                        'bars_available': len(bars) if bars else 0
                    }
                    play['final_score'] = scoring_engine.calculate_score(play)
                    all_plays.append(play)
                
                logger.info(f"  🎯 Added {len(options_plays)} scored plays")
                logger.info("")
                
            except Exception as e:
                logger.warning(f"❌ Error analyzing {symbol}: {e}")
                continue
        
        # Sort by score and display results
        logger.info("🏆 ANALYZING RESULTS...")
        logger.info("=" * 80)
        
        if not all_plays:
            logger.warning("❌ No options plays found. This could be due to:")
            logger.warning("   • Market hours (options data may be limited)")
            logger.warning("   • Yahoo Finance API limitations")
            logger.warning("   • Strict filtering criteria")
            return
        
        # Sort by final score
        top_plays = sorted(all_plays, key=lambda x: x.get('final_score', 0), reverse=True)[:15]
        
        logger.info(f"📈 TOP {len(top_plays)} OPTIONS PLAYS (Sorted by Score):")
        logger.info("=" * 100)
        logger.info(f"{'#':<3} {'Symbol':<6} {'Type':<4} {'Strike':<8} {'Expiry':<12} {'Score':<6} {'Bid/Ask':<12} {'Vol':<6} {'OI':<6} {'Sentiment':<9}")
        logger.info("-" * 100)
        
        for i, play in enumerate(top_plays, 1):
            sentiment_icon = "📈" if play['sentiment_score'] > 0.1 else "📉" if play['sentiment_score'] < -0.1 else "➡️"
            
            logger.info(
                f"{i:<3} {play['symbol']:<6} {play['type']:<4} "
                f"${play['strike']:<7.2f} {str(play['expiry']):<12} "
                f"{play['final_score']:<6.1f} "
                f"${play['bid']:.2f}/${play['ask']:.2f}    "
                f"{play['volume']:<6} {play['open_interest']:<6} "
                f"{play['sentiment_score']:+.3f} {sentiment_icon}"
            )
        
        # Summary statistics
        logger.info("=" * 100)
        logger.info("📊 SUMMARY STATISTICS:")
        
        avg_score = sum(p['final_score'] for p in top_plays) / len(top_plays)
        avg_sentiment = sum(p['sentiment_score'] for p in top_plays) / len(top_plays)
        
        call_plays = [p for p in top_plays if p['type'] == 'CALL']
        put_plays = [p for p in top_plays if p['type'] == 'PUT']
        
        logger.info(f"  • Total Plays Analyzed: {len(all_plays)}")
        logger.info(f"  • Average Score: {avg_score:.2f}")
        logger.info(f"  • Average Sentiment: {avg_sentiment:+.3f}")
        logger.info(f"  • Call Plays: {len(call_plays)} | Put Plays: {len(put_plays)}")
        logger.info(f"  • Symbols with Positive Sentiment: {len([p for p in top_plays if p['sentiment_score'] > 0])}")
        
        # Top performer
        if top_plays:
            best_play = top_plays[0]
            logger.info(f"  • 🥇 Best Play: {best_play['symbol']} {best_play['type']} ${best_play['strike']:.2f} (Score: {best_play['final_score']:.1f})")
        
        elapsed = datetime.now() - start_time
        logger.info(f"⏱️  Scan completed in {elapsed.total_seconds():.1f} seconds")
        logger.info("✅ DRY RUN COMPLETE!")
        
    except Exception as e:
        logger.error(f"❌ Dry run failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Configure detailed logging for dry run
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO"
    )
    
    logger.info("🧪 OPTIONS AI SCANNER - DRY RUN MODE")
    logger.info("📋 This will test the scanner with real market data")
    logger.info("")
    
    # Run the dry run
    asyncio.run(dry_run_scanner())
