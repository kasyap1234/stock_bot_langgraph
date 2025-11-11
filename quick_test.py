#!/usr/bin/env python3
"""
Quick sanity check for stock swing trade recommender
Tests basic imports and structure
"""

import sys
from pathlib import Path

print("=" * 80)
print("QUICK SANITY CHECK")
print("=" * 80)
print()

# Test 1: Check Python version
print("1. Checking Python version...")
version = sys.version_info
if version >= (3, 10):
    print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
else:
    print(f"   ✗ Python {version.major}.{version.minor}.{version.micro} - Need 3.10+")
    sys.exit(1)

# Test 2: Check critical files exist
print("\n2. Checking critical files...")
critical_files = [
    "main.py",
    "config/config.py",
    "agents/data_fetcher.py",
    "agents/technical_analysis.py",
    "recommendation/final_recommendation.py",
    ".env",
]

all_exist = True
for file_path in critical_files:
    if Path(file_path).exists():
        print(f"   ✓ {file_path}")
    else:
        print(f"   ✗ {file_path} - MISSING")
        all_exist = False

if not all_exist:
    print("\n   Some critical files are missing!")
    sys.exit(1)

# Test 3: Try importing config
print("\n3. Checking configuration...")
try:
    from config import config
    print(f"   ✓ Config loaded")
    print(f"   - Default stocks: {len(config.DEFAULT_STOCKS)}")
    print(f"   - NIFTY 50 stocks: {len(config.NIFTY_50_STOCKS)}")
except Exception as e:
    print(f"   ✗ Config error: {e}")
    sys.exit(1)

# Test 4: Check .env file has content
print("\n4. Checking environment configuration...")
try:
    # Read .env file directly without python-dotenv
    env_path = Path(".env")
    if env_path.exists():
        content = env_path.read_text()
        keys = ["GROQ_API_KEY", "NEWS_API_KEY", "FRED_API_KEY"]
        configured = []
        for key in keys:
            for line in content.split('\n'):
                if line.startswith(key) and '=' in line:
                    value = line.split('=', 1)[1].strip().strip('"\'')
                    if value and value != "" and "your_" not in value.lower():
                        configured.append(key)
                    break

        if configured:
            print(f"   ✓ {len(configured)}/{len(keys)} API keys configured")
            for key in configured:
                print(f"     - {key}")
        else:
            print(f"   ⚠ No API keys configured (system will work with limited functionality)")
    else:
        print(f"   ⚠ .env file not found")
except Exception as e:
    print(f"   ⚠ Could not check API keys: {e}")

# Test 5: Check if data models can be imported
print("\n5. Checking data models...")
try:
    # Just check if the file exists and has State definition
    models_path = Path("data/models.py")
    if models_path.exists():
        content = models_path.read_text()
        if "State" in content and "TypedDict" in content:
            print(f"   ✓ Data models file exists with State definition")
        else:
            print(f"   ⚠ Data models file exists but State definition unclear")
    else:
        print(f"   ✗ data/models.py not found")
        sys.exit(1)
except Exception as e:
    print(f"   ⚠ Data models check warning: {e}")

# Summary
print()
print("=" * 80)
print("✓ BASIC SANITY CHECK PASSED")
print("=" * 80)
print()
print("Next steps:")
print("  1. Install all dependencies: uv sync")
print("  2. Configure API keys in .env file")
print("  3. Run validation: uv run python validate_setup.py")
print("  4. Run analysis: uv run main.py --ticker RELIANCE.NS")
print()
