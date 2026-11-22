"""Money value object shared by all Python services."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal

MINOR_UNITS = {"USD": 2, "EUR": 2, "JPY": 0, "VND": 0}


@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str

    def to_minor_units(self) -> int:
        """Convert to integer minor units using banker's rounding."""
        exp = MINOR_UNITS.get(self.currency, 2)
        quantized = self.amount.quantize(Decimal(1).scaleb(-exp), rounding=ROUND_HALF_EVEN)
        return int(quantized.scaleb(exp))

    def add(self, other: Money) -> Money:
        if other.currency != self.currency:
            raise ValueError("currency mismatch")
        return Money(self.amount + other.amount, self.currency)


def parse_money(raw: str) -> Money:
    """Parse strings like '12.50 USD'."""
    value, currency = raw.split()
    return Money(Decimal(value), currency.upper())
