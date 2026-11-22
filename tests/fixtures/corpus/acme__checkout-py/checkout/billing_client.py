"""gRPC client for the Go billing service."""

from __future__ import annotations

import grpc
from acme_common.money import Money

from checkout.gen import billing_pb2, billing_pb2_grpc


class BillingClient:
    """Thin wrapper around BillingServiceStub with deadline and idempotency handling."""

    def __init__(self, target: str, timeout_s: float = 2.0) -> None:
        channel = grpc.insecure_channel(target)
        self._stub = billing_pb2_grpc.BillingServiceStub(channel)
        self._timeout_s = timeout_s

    def charge(self, order_id: str, total: Money, idempotency_key: str) -> str:
        """Charge the order total; returns the billing charge id."""
        request = billing_pb2.ChargeRequest(
            order_id=order_id,
            amount_minor=total.to_minor_units(),
            currency=total.currency,
            idempotency_key=idempotency_key,
        )
        response = self._stub.Charge(request, timeout=self._timeout_s)
        return response.charge_id

    def refund(self, charge_id: str, amount: Money) -> str:
        response = self._stub.Refund(
            billing_pb2.RefundRequest(charge_id=charge_id, amount_minor=amount.to_minor_units()),
            timeout=self._timeout_s,
        )
        return response.refund_id
