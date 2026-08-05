"""Market Data Generator - Creates synthetic market data for testing."""

import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict


class MarketDataGenerator:
    """Generate realistic synthetic market data."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

        # Base prices for common symbols
        self.base_prices = {
            "NVDA": 850.0,
            "AAPL": 185.0,
            "GOOGL": 155.0,
            "MSFT": 420.0,
            "TSLA": 245.0,
        }

    def generate_price_series(self, symbol: str, days: int = 30) -> List[Dict]:
        """Generate realistic OHLCV price series."""
        base_price = self.base_prices.get(symbol, 100.0)

        prices = [base_price]
        for _ in range(days - 1):
            drift = random.uniform(-0.005, 0.01)
            shock = random.gauss(0, 0.02)
            new_price = prices[-1] * (1 + drift + shock)
            prices.append(max(new_price, 1.0))

        # Generate OHLCV data
        ohlcv_data = []
        start_date = datetime.now() - timedelta(days=days)

        for i, close in enumerate(prices):
            date = start_date + timedelta(days=i)
            intraday_range = close * random.uniform(0.01, 0.03)
            open_price = close + random.uniform(-intraday_range/2, intraday_range/2)
            high = max(open_price, close) + random.uniform(0, intraday_range)
            low = min(open_price, close) - random.uniform(0, intraday_range)
            volume = int(random.uniform(50_000_000, 150_000_000))

            ohlcv_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": volume
            })

        return ohlcv_data

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices[-period-1:])
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)

        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.01

        rs = avg_gain / avg_loss if avg_loss != 0 else 100
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_macd(self, prices: List[float]) -> tuple:
        """Calculate MACD and signal line."""
        if len(prices) < 26:
            return (0.0, 0.0)

        # Simplified MACD calculation
        fast_ema = np.mean(prices[-12:])
        slow_ema = np.mean(prices[-26:])
        macd = fast_ema - slow_ema
        signal = macd * 0.9  # Simplified signal

        return (macd, signal)
