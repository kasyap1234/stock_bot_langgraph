#!/bin/bash

# Stock Swing Trade Recommender - Quick Analysis Script
# Usage: ./run_analysis.sh TICKER [OPTIONS]
# Example: ./run_analysis.sh RELIANCE.NS
#          ./run_analysis.sh RELIANCE.NS --backtest

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "=========================================="
    echo "  Stock Swing Trade Recommender"
    echo "=========================================="
    echo -e "${NC}"
}

# Print usage
print_usage() {
    echo "Usage: $0 TICKER [OPTIONS]"
    echo ""
    echo "Examples:"
    echo "  $0 RELIANCE.NS              # Analyze single stock"
    echo "  $0 RELIANCE.NS --backtest   # With backtesting"
    echo "  $0 --nifty50                # Analyze all NIFTY 50 stocks"
    echo "  $0 --multi TCS.NS,INFY.NS   # Analyze multiple stocks"
    echo ""
    echo "Popular NSE Stocks:"
    echo "  RELIANCE.NS  - Reliance Industries"
    echo "  TCS.NS       - Tata Consultancy Services"
    echo "  INFY.NS      - Infosys"
    echo "  HDFCBANK.NS  - HDFC Bank"
    echo "  ICICIBANK.NS - ICICI Bank"
    echo "  SBIN.NS      - State Bank of India"
    echo ""
    exit 1
}

# Check if uv is installed
check_uv() {
    if ! command -v uv &> /dev/null; then
        echo -e "${RED}Error: 'uv' is not installed${NC}"
        echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
}

# Main script
main() {
    print_banner

    # Check if no arguments
    if [ $# -eq 0 ]; then
        print_usage
    fi

    # Check if uv is installed
    check_uv

    # Parse arguments
    TICKER=""
    EXTRA_ARGS=""

    case "$1" in
        --nifty50)
            echo -e "${GREEN}Analyzing all NIFTY 50 stocks...${NC}"
            EXTRA_ARGS="--nifty50"
            shift
            ;;
        --multi)
            if [ -z "$2" ]; then
                echo -e "${RED}Error: --multi requires ticker list${NC}"
                print_usage
            fi
            echo -e "${GREEN}Analyzing multiple stocks: $2${NC}"
            EXTRA_ARGS="--tickers $2"
            shift 2
            ;;
        -h|--help)
            print_usage
            ;;
        *)
            TICKER="$1"
            shift
            echo -e "${GREEN}Analyzing $TICKER...${NC}"
            EXTRA_ARGS="--ticker $TICKER"
            ;;
    esac

    # Add any additional arguments
    while [ $# -gt 0 ]; do
        EXTRA_ARGS="$EXTRA_ARGS $1"
        shift
    done

    echo ""
    echo -e "${YELLOW}Starting analysis...${NC}"
    echo ""

    # Run the analysis
    uv run main.py $EXTRA_ARGS

    echo ""
    echo -e "${GREEN}Analysis complete!${NC}"
}

# Run main function
main "$@"
