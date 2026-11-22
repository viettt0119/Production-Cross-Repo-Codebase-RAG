export const TAX_RATE = 0.08;

/** Applies a percentage discount code; unknown codes are ignored. */
export function applyDiscount(subtotal: number, code?: string): number {
  const table: Record<string, number> = { WELCOME10: 0.1, VIP20: 0.2 };
  const rate = code ? table[code] ?? 0 : 0;
  return Math.round(subtotal * (1 - rate) * 100) / 100;
}

export function withTax(amount: number): number {
  return Math.round(amount * (1 + TAX_RATE) * 100) / 100;
}
