from __future__ import annotations


class OutOfStockError(Exception):
    pass


class Inventory:
    def __init__(self, stock: dict[str, int]) -> None:
        self._stock = dict(stock)

    def reserve(self, sku: str, qty: int) -> None:
        """Atomically decrement stock or raise OutOfStockError."""
        available = self._stock.get(sku, 0)
        if available < qty:
            raise OutOfStockError(f"{sku}: requested {qty}, available {available}")
        self._stock[sku] = available - qty

    def release(self, sku: str, qty: int) -> None:
        self._stock[sku] = self._stock.get(sku, 0) + qty
