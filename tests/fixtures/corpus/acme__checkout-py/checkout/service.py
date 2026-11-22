"""Order placement orchestration."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from acme_common.money import Money, parse_money

from checkout.billing_client import BillingClient
from checkout.inventory import Inventory, OutOfStockError


@dataclass
class OrderLine:
    sku: str
    qty: int
    unit_price: str


class CheckoutService:
    """Reserves inventory, then charges billing; releases stock if the charge fails."""

    def __init__(self, billing: BillingClient, inventory: Inventory) -> None:
        self.billing = billing
        self.inventory = inventory

    def place_order(self, lines: list[OrderLine]) -> str:
        order_id = f"ord_{uuid.uuid4().hex[:12]}"
        reserved: list[OrderLine] = []
        try:
            for line in lines:
                self.inventory.reserve(line.sku, line.qty)
                reserved.append(line)
            total = self.order_total(lines)
            self.billing.charge(order_id, total, idempotency_key=order_id)
        except (OutOfStockError, RuntimeError):
            for line in reserved:
                self.inventory.release(line.sku, line.qty)
            raise
        return order_id

    @staticmethod
    def order_total(lines: list[OrderLine]) -> Money:
        total = parse_money(f"0 {parse_money(lines[0].unit_price).currency}")
        for line in lines:
            unit = parse_money(line.unit_price)
            total = total.add(Money(unit.amount * line.qty, unit.currency))
        return total
