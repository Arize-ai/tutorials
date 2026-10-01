"""Tools the support agent can call."""

import json
from datetime import datetime

from store import ORDERS, POLICIES, PRODUCTS


def lookup_order(customer_email: str, order_date: str) -> str:
    placed = datetime.strptime(order_date, "%Y-%m-%d").date()
    matches = [
        order
        for order in ORDERS
        if order["email"] == customer_email.lower() and order["placed"] == placed
    ]
    if not matches:
        return f"No orders found for {customer_email} on {order_date}."
    return json.dumps(matches, default=str)


def check_stock(product_name: str) -> str:
    for name, product in PRODUCTS.items():
        if product_name.lower() in name.lower():
            return json.dumps({"product": name, **product})
    return f"No product matching {product_name!r}."


def get_policy(topic: str) -> str:
    return POLICIES.get(topic, f"No policy found for {topic!r}. Topics: {', '.join(POLICIES)}")


TOOL_SPECS = {
    "lookup_order": {
        "function": lookup_order,
        "description": "Look up a customer's order by their email address and the date they placed it.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_email": {"type": "string", "description": "The customer's email address"},
                "order_date": {"type": "string", "description": "The date the customer placed the order"},
            },
            "required": ["customer_email", "order_date"],
        },
    },
    "check_stock": {
        "function": check_stock,
        "description": "Check the price and stock level of a product.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_name": {"type": "string", "description": "The product name, or part of it"},
            },
            "required": ["product_name"],
        },
    },
    "get_policy": {
        "function": get_policy,
        "description": "Get the text of a store policy.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "enum": list(POLICIES), "description": "The policy topic"},
            },
            "required": ["topic"],
        },
    },
}

# The tools offered to the model
ENABLED_TOOLS = ["lookup_order", "check_stock"]


def execute_tool(name: str, arguments: dict) -> str:
    return TOOL_SPECS[name]["function"](**arguments)
