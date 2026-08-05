"""Bull Agent MCP Tools - Opportunity analysis tools."""

import numpy as np
from mcp.server.fastmcp import FastMCP

from mcp_tools.market_data import MarketDataGenerator

# Initialize MCP server
mcp = FastMCP("bull-agent-tools")

# Create global market data generator
market_generator = MarketDataGenerator()


@mcp.tool()
async def find_breakout_patterns(symbol: str) -> str:
    """Identify bullish breakout patterns and technical setups."""
    prices = market_generator.generate_price_series(symbol, days=30)
    current_price = prices[-1]["close"]

    breakout_score = np.random.uniform(55, 85)
    momentum = "STRONG" if breakout_score > 70 else "MODERATE"
    separator = "=" * 40

    return f"""
BREAKOUT PATTERN ANALYSIS FOR {symbol}
{separator}
Current Price: ${current_price}
Breakout Score: {breakout_score:.1f}/100
Momentum: {momentum}

Bullish Patterns:
[HIGH] Resistance Breakout
   Price breaking above key resistance
   Target: ${round(current_price * 1.08, 2)} (+8%)

[MED] Ascending Triangle
   Higher lows with resistance test
   Target: ${round(current_price * 1.10, 2)} (+10%)
"""


@mcp.tool()
async def momentum_screener(symbol: str) -> str:
    """Screen for stocks with strong upward momentum."""
    prices = market_generator.generate_price_series(symbol, days=30)
    closes = [p["close"] for p in prices]
    rsi = market_generator.calculate_rsi(closes)

    momentum_score = np.random.uniform(60, 90)
    rating = "VERY STRONG" if momentum_score > 80 else "STRONG"
    separator = "=" * 40

    return f"""
MOMENTUM ANALYSIS FOR {symbol}
{separator}
Momentum Score: {momentum_score:.1f}/100
Rating: {rating}
Trend: BULLISH

Momentum Factors:
- Healthy RSI at {rsi:.1f} - room to run
- MACD bullish crossover confirmed
- Volume surge - institutions accumulating
- Uptrend pattern intact
"""


@mcp.tool()
async def entry_signal_detector(symbol: str) -> str:
    """Detect optimal entry points for long positions."""
    prices = market_generator.generate_price_series(symbol, days=30)
    current_price = prices[-1]["close"]

    entry_quality = np.random.uniform(60, 90)
    sizing = "75-100%" if entry_quality > 80 else "50-75%"
    separator = "=" * 40

    return f"""
ENTRY SIGNAL ANALYSIS FOR {symbol}
{separator}
Current Price: ${current_price}
Entry Quality: {entry_quality:.1f}/100

Entry Signals:
[HIGH] Pullback to Support
   Quality entry at ${round(current_price * 0.98, 2)}
   Stop Loss: ${round(current_price * 0.95, 2)}
   Risk/Reward: 1:3

Position Sizing:
   Suggested: {sizing} of planned position
"""
