"""In-memory data for Trailhead Outfitters, a fictional outdoor gear store.

Order dates are relative to today so questions like "the order I placed
yesterday" always have a matching order.
"""

from datetime import date, timedelta

TODAY = date.today()


def _days_ago(days: int) -> date:
    return TODAY - timedelta(days=days)


CUSTOMERS = {
    "maria@example.com": "Maria Lopez",
    "dev@example.com": "Dev Patel",
    "sam@example.com": "Sam Okafor",
    "lee@example.com": "Lee Chen",
    "ana@example.com": "Ana Silva",
    "tom@example.com": "Tom Becker",
}

ORDERS = [
    {"order_id": "TH-1001", "email": "maria@example.com", "placed": _days_ago(1), "items": ["Summit 2 Tent"], "status": "processing"},
    {"order_id": "TH-1002", "email": "maria@example.com", "placed": _days_ago(9), "items": ["Trail Runner Socks (3-pack)"], "status": "delivered"},
    {"order_id": "TH-1003", "email": "dev@example.com", "placed": _days_ago(3), "items": ["Ridge 30L Daypack", "Alpine Water Filter"], "status": "shipped"},
    {"order_id": "TH-1004", "email": "sam@example.com", "placed": _days_ago(7), "items": ["Beacon GPS Watch"], "status": "delivered"},
    {"order_id": "TH-1005", "email": "lee@example.com", "placed": _days_ago(2), "items": ["Glow 400 Headlamp"], "status": "shipped"},
    {"order_id": "TH-1006", "email": "ana@example.com", "placed": _days_ago(14), "items": ["Cascade Rain Jacket"], "status": "delivered"},
    {"order_id": "TH-1007", "email": "tom@example.com", "placed": _days_ago(0), "items": ["Basecamp Stove", "Fuel Canister"], "status": "processing"},
    {"order_id": "TH-1008", "email": "dev@example.com", "placed": _days_ago(21), "items": ["Summit 2 Tent"], "status": "delivered"},
]

PRODUCTS = {
    "Summit 2 Tent": {"price": 249.00, "stock": 4},
    "Ridge 30L Daypack": {"price": 89.00, "stock": 0},
    "Alpine Water Filter": {"price": 39.00, "stock": 27},
    "Beacon GPS Watch": {"price": 329.00, "stock": 2},
    "Glow 400 Headlamp": {"price": 45.00, "stock": 58},
    "Cascade Rain Jacket": {"price": 159.00, "stock": 11},
    "Basecamp Stove": {"price": 69.00, "stock": 0},
    "Trail Runner Socks (3-pack)": {"price": 24.00, "stock": 140},
}

POLICIES = {
    "returns": (
        "Unused items can be returned within 21 days of delivery for a full refund. "
        "Sale and clearance items are final sale and cannot be returned. "
        "Electronics (GPS watches, headlamps) have a 10% restocking fee."
    ),
    "return-shipping": (
        "Return shipping is free for Trailhead Club members. "
        "Everyone else pays a flat $7.95 for a prepaid return label."
    ),
    "shipping": (
        "Standard shipping takes 5-7 business days and is free on orders over $75. "
        "Express shipping takes 2 business days and costs $19."
    ),
    "price-matching": "Trailhead Outfitters does not offer price matching.",
    "warranty": (
        "Tents, packs, and jackets have a 2-year warranty against manufacturing defects. "
        "Electronics have a 1-year warranty."
    ),
}
