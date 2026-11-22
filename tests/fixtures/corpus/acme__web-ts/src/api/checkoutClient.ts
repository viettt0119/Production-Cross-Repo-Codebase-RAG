export interface OrderLineInput {
  sku: string;
  qty: number;
  price: string;
}

export class CheckoutApiError extends Error {}

/** Calls checkout-py POST /orders and returns the created order id. */
export async function submitOrder(baseUrl: string, lines: OrderLineInput[]): Promise<string> {
  const res = await fetch(`${baseUrl}/orders`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ lines }),
  });
  if (!res.ok) {
    throw new CheckoutApiError(`checkout failed: ${res.status}`);
  }
  const body = (await res.json()) as { order_id: string };
  return body.order_id;
}
