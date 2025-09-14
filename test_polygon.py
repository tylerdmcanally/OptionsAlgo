#!/usr/bin/env python3
"""
Test script for Polygon API integration
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from polygon_manager import PolygonAPIManager
from config import Config


async def test_polygon_integration():
    """Test the Polygon API integration"""
    print("🧪 Testing Polygon API Integration")
    print("=" * 50)
    
    try:
        # Load config
        config = Config()
        print(f"📋 Config loaded, API key configured: {bool(config.polygon_api_key and config.polygon_api_key != 'your_polygon_api_key_here')}")
        
        # Initialize Polygon manager
        polygon_manager = PolygonAPIManager(config.polygon_api_key)
        print(f"🔌 Polygon manager ready: {polygon_manager.ready}")
        
        # Test API status
        print("\n📡 Testing API connectivity...")
        status = await polygon_manager.check_api_status()
        print(f"Status: {status}")
        
        if polygon_manager.ready:
            # Test news fetching
            print("\n📰 Testing news fetching for AAPL...")
            news = await polygon_manager.get_ticker_news('AAPL', days_back=3, limit=5)
            print(f"Retrieved {len(news)} news articles")
            
            if news:
                print("\nSample article:")
                sample = news[0]
                print(f"  Title: {sample.get('title', 'N/A')[:100]}...")
                print(f"  Published: {sample.get('published_utc', 'N/A')}")
                print(f"  Tickers: {sample.get('tickers', [])}")
            
            # Test market snapshot
            print("\n📊 Testing market snapshot for AAPL...")
            snapshot = await polygon_manager.get_market_snapshot('AAPL')
            if snapshot:
                print(f"Current price: ${snapshot.get('last_trade', {}).get('price', 'N/A')}")
                print(f"Session change: {snapshot.get('session', {}).get('change_percent', 'N/A')}%")
            else:
                print("No snapshot data retrieved")
                
            # Test daily bars
            print("\n📈 Testing daily bars for AAPL (last 5 days)...")
            bars = await polygon_manager.get_daily_bars('AAPL', days_back=5)
            print(f"Retrieved {len(bars)} daily bars")
            
            if bars:
                latest = bars[-1]
                print(f"Latest bar: O:{latest.get('open', 'N/A')} H:{latest.get('high', 'N/A')} L:{latest.get('low', 'N/A')} C:{latest.get('close', 'N/A')}")
        else:
            print("⚠️  Polygon API not ready - check your API key in .env file")
        
        print("\n✅ Polygon API test completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_polygon_integration())
