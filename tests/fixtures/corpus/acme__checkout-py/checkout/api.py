from __future__ import annotations

from checkout.service import CheckoutService, OrderLine


def create_order_handler(service: CheckoutService, payload: dict[str, list[dict[str, object]]]) -> dict[str, str]:
    """POST /orders — validates the payload and delegates to CheckoutService.place_order."""
    lines = [OrderLine(sku=str(i["sku"]), qty=int(str(i["qty"])), unit_price=str(i["price"])) for i in payload["lines"]]
    if not lines:
        raise ValueError("order must contain at least one line")
    order_id = service.place_order(lines)
    return {"order_id": order_id, "status": "placed"}
