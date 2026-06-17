from store.pricing import Order, order_total


def test_no_code_adds_shipping() -> None:
    assert order_total(Order(subtotal=100.0, shipping=5.0)) == 105.0


def test_percent_off_applies_to_subtotal_only() -> None:
    assert order_total(Order(subtotal=100.0, shipping=5.0, code="SAVE10")) == 95.0
