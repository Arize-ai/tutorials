"""Bear Agent MCP Tools - Risk analysis tools."""

import numpy as np
from mcp.server.fastmcp import FastMCP

from mcp_tools.market_data import MarketDataGenerator

# Initialize MCP server
mcp = FastMCP("bear-agent-tools")

# Create global market data generator
market_generator = MarketDataGenerator()

RSI_OVERBOUGHT_THRESHOLD = 70
RISK_HIGH_THRESHOLD = 60


@mcp.tool()
async def risk_scanner(symbol: str) -> str:
    """Scan for potential downside risks and warning signals."""
    prices = market_generator.generate_price_series(symbol, days=30)
    current_price = prices[-1]["close"]
    closes = [p["close"] for p in prices]
    rsi = market_generator.calculate_rsi(closes)

    risk_score = np.random.uniform(40, 75)

    risks = []
    if rsi > RSI_OVERBOUGHT_THRESHOLD:
        risks.append(
            {
                "risk": "Overbought Conditions",
                "severity": "HIGH",
                "description": f"RSI at {rsi:.1f} indicates potential pullback",
                "impact": "-5% to -10%",
            }
        )

    if len(risks) == 0:
        risks.append(
            {
                "risk": "Valuation Concerns",
                "severity": "MEDIUM",
                "description": "P/E ratio elevated vs historical average",
                "impact": "-10% to -15%",
            }
        )

    risk_level = "HIGH" if risk_score > RISK_HIGH_THRESHOLD else "MEDIUM"
    separator = "=" * 40

    result = f"""
RISK ANALYSIS FOR {symbol}
{separator}
Current Price: ${current_price}
Risk Score: {risk_score:.1f}/100
Risk Level: {risk_level}

Identified Risks:
"""

    for risk in risks:
        result += f"\n[{risk['severity']}] {risk['risk']}"
        result += f"\n   {risk['description']}"
        result += f"\n   Potential Impact: {risk['impact']}\n"

    return result


@mcp.tool()
async def divergence_detector(symbol: str) -> str:
    """Detect bearish divergences and technical weakness."""
    prices = market_generator.generate_price_series(symbol, days=30)
    closes = [p["close"] for p in prices]
    rsi = market_generator.calculate_rsi(closes)

    divergence_score = np.random.uniform(30, 70)
    separator = "=" * 40

    return f"""
DIVERGENCE ANALYSIS FOR {symbol}
{separator}
Divergence Score: {divergence_score:.1f}/100
RSI: {rsi:.1f}

Detected Divergences:
- RSI Bearish Divergence
   Price making highs but RSI not confirming
   Confidence: 75%

- Volume Divergence
   Declining volume on advances
   Confidence: 70%
"""


@mcp.tool()
async def exit_signal_monitor(symbol: str) -> str:
    """Monitor for distribution patterns and exit signals."""
    prices = market_generator.generate_price_series(symbol, days=30)
    current_price = prices[-1]["close"]

    stop_aggressive = round(current_price * 0.95, 2)
    stop_moderate = round(current_price * 0.93, 2)
    separator = "=" * 40

    return f"""
EXIT SIGNAL MONITOR FOR {symbol}
{separator}
Current Price: ${current_price}

Exit Signals:
[MED] Distribution Pattern
   Heavy selling on up days
   Action: Reduce position size

Stop Loss Recommendations:
   Aggressive: ${stop_aggressive} (-5%)
   Moderate: ${stop_moderate} (-7%)
"""
