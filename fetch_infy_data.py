#!/usr/bin/env python3
"""
Script to re-fetch INFY.NS data since the cache file is corrupted
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.apis import UnifiedDataFetcher

def main():
    print("Re-fetching INFY.NS data...")

    try:
        # Force fresh fetch by deleting cache file first
        import os
        cache_file = "data/cache/INFY.NS_1y.json"
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print("Removed corrupted cache file")

        fetcher = UnifiedDataFetcher("INFY.NS")
        data = fetcher.get_historical_data(period="1y", interval="1d")

        print(f"Successfully fetched {len(data)} data points for INFY.NS")
        print(f"Date range: {data[0]['date']} to {data[-1]['date']}")
        print(f"Latest close price: ₹{data[-1]['close']}")

    except Exception as e:
        print(f"Failed to fetch INFY.NS data: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)