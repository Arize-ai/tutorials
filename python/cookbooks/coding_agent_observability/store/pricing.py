"""Order pricing with discount-code support."""

from __future__ import annotations

from dataclasses import dataclass

PERCENT_OFF: dict[str, float] = {"SAVE10": 0.10, "SAVE20": 0.20}


@dataclass(frozen=True)
class Order:
    subtotal: float
    shipping: float
    code: str | None = None


def order_total(order: Order) -> float:
    """Return the order total: the discounted subtotal plus shipping."""
    rate = PERCENT_OFF.get(order.code or "", 0.0)
    return round(order.subtotal * (1 - rate) + order.shipping, 2)
