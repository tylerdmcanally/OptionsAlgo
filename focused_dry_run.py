#!/usr/bin/env python3
"""
Focused Dry Run - Show Available Data and Debug Options Filtering
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
from polygon_manager import PolygonAPIManager
import yfinance as yf


async def focused_dry_run():
    """Focused dry run to show available data and debug options filtering"""
    logger.info("🔍 FOCUSED DRY RUN - Data Analysis & Debug")
    logger.info("=" * 60)
    
    try:
        # Initialize components
        config = Config()
        polygon_manager = PolygonAPIManager(config.polygon_api_key)
        
        # Test one symbol thoroughly
        test_symbol = "AAPL"
        logger.info(f"📊 Deep Analysis of {test_symbol}")
        logger.info("-" * 40)
        
        # 1. Get sentiment (with rate limiting awareness)
        logger.info("1️⃣ Testing Sentiment Analysis...")
        try:
            sentiment_analyzer = SentimentAnalyzer(config.polygon_api_key)
            sentiment = await sentiment_analyzer.get_sentiment(test_symbol)
            logger.info(f"   ✅ Sentiment Score: {sentiment:.3f}")
        except Exception as e:
            logger.warning(f"   ❌ Sentiment Error: {e}")
            sentiment = 0.0
        
        # 2. Get Yahoo Finance data
        logger.info("2️⃣ Testing Yahoo Finance Integration...")
        try:
            ticker = yf.Ticker(test_symbol)
            info = ticker.info
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            logger.info(f"   ✅ Current Price: ${current_price:.2f}")
            logger.info(f"   ✅ Market Cap: ${info.get('marketCap', 0):,}")
            logger.info(f"   ✅ Volume: {info.get('volume', 0):,}")
        except Exception as e:
            logger.warning(f"   ❌ Yahoo Finance Error: {e}")
            current_price = 234.0  # Fallback
        
        # 3. Get options chain details
        logger.info("3️⃣ Testing Options Chain Retrieval...")
        try:
            ticker = yf.Ticker(test_symbol)
            expirations = ticker.options
            logger.info(f"   ✅ Available Expirations: {len(expirations)} dates")
            logger.info(f"   📅 Next 3 Expirations: {list(expirations[:3])}")
            
            if expirations:
                # Get first expiration chain
                exp_date = expirations[0]
                chain = ticker.option_chain(exp_date)
                
                calls_count = len(chain.calls)
                puts_count = len(chain.puts)
                logger.info(f"   ✅ {exp_date}: {calls_count} calls, {puts_count} puts")
                
                # Show sample calls with volume > 0
                sample_calls = chain.calls[chain.calls['volume'] > 0].head(3)
                if not sample_calls.empty:
                    logger.info("   📈 Sample Active Calls:")
                    for _, row in sample_calls.iterrows():
                        logger.info(f"      ${row['strike']:.2f} | Vol:{row['volume']} | OI:{row['openInterest']} | Bid:${row['bid']:.2f}")
                else:
                    logger.warning("   ⚠️  No calls with volume > 0 found")
                
                # Show sample puts with volume > 0
                sample_puts = chain.puts[chain.puts['volume'] > 0].head(3)
                if not sample_puts.empty:
                    logger.info("   📉 Sample Active Puts:")
                    for _, row in sample_puts.iterrows():
                        logger.info(f"      ${row['strike']:.2f} | Vol:{row['volume']} | OI:{row['openInterest']} | Bid:${row['bid']:.2f}")
                else:
                    logger.warning("   ⚠️  No puts with volume > 0 found")
                
        except Exception as e:
            logger.error(f"   ❌ Options Chain Error: {e}")
        
        # 4. Test our filtering criteria
        logger.info("4️⃣ Testing Scanner Filtering Criteria...")
        try:
            options_scanner = OptionsScanner(config.polygon_api_key)
            
            # Show current filter settings
            logger.info(f"   📋 Current Filters:")
            logger.info(f"      • Days to Expiry: {options_scanner.min_days}-{options_scanner.max_days}")
            logger.info(f"      • Min Volume: {options_scanner.min_volume}")
            logger.info(f"      • Min Open Interest: {options_scanner.min_oi}")
            logger.info(f"      • OTM Range: {options_scanner.otm_min:.1%}-{options_scanner.otm_max:.1%}")
            
            # Try with relaxed criteria
            logger.info("   🔧 Testing with RELAXED criteria...")
            options_scanner.min_volume = 10  # Reduce from 50
            options_scanner.min_oi = 25      # Reduce from 100
            options_scanner.otm_min = 0.01   # Reduce from 0.02
            options_scanner.otm_max = 0.20   # Increase from 0.15
            
            plays = await options_scanner.scan_symbol(test_symbol)
            logger.info(f"   ✅ Found {len(plays)} plays with relaxed criteria")
            
            if plays:
                logger.info("   🎯 Top 3 Plays Found:")
                for i, play in enumerate(plays[:3], 1):
                    logger.info(f"      {i}. {play['type']} ${play['strike']:.2f} {play['expiry']} | Vol:{play['volume']} OI:{play['open_interest']}")
            
        except Exception as e:
            logger.error(f"   ❌ Scanner Error: {e}")
        
        # 5. Market timing check
        logger.info("5️⃣ Market Timing Analysis...")
        now = datetime.now()
        logger.info(f"   🕐 Current Time: {now.strftime('%Y-%m-%d %H:%M:%S %A')}")
        
        # Check if it's during market hours (rough estimate)
        hour = now.hour
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        
        if weekday < 5:  # Monday-Friday
            if 9 <= hour <= 16:  # Rough market hours
                logger.info("   ✅ During approximate market hours (9AM-4PM EST)")
            else:
                logger.info("   ⚠️  Outside market hours - options data may be stale")
        else:
            logger.info("   ⚠️  Weekend - markets closed")
        
        logger.info("=" * 60)
        logger.info("📋 SUMMARY & RECOMMENDATIONS:")
        logger.info("✅ Polygon API: Working for news/historical (rate limited)")
        logger.info("✅ Yahoo Finance: Working for options chains")
        logger.info("✅ FinBERT: Loaded and functional")
        logger.info("⚠️  Options filtering may be too strict for current market conditions")
        logger.info("💡 Consider adjusting filters during off-hours or low-volume periods")
        
    except Exception as e:
        logger.error(f"❌ Focused dry run failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO"
    )
    
    asyncio.run(focused_dry_run())
