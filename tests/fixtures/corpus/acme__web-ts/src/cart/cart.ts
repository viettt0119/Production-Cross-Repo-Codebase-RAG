import { applyDiscount, withTax } from "./pricing";
import { submitOrder, OrderLineInput } from "../api/checkoutClient";

export class Cart {
  private lines: OrderLineInput[] = [];

  add(line: OrderLineInput): void {
    const existing = this.lines.find((l) => l.sku === line.sku);
    if (existing) {
      existing.qty += line.qty;
    } else {
      this.lines.push({ ...line });
    }
  }

  /** Subtotal after discount, including tax. */
  total(discountCode?: string): number {
    const subtotal = this.lines.reduce((acc, l) => acc + Number(l.price) * l.qty, 0);
    return withTax(applyDiscount(subtotal, discountCode));
  }

  async checkout(baseUrl: string): Promise<string> {
    return submitOrder(baseUrl, this.lines);
  }
}
