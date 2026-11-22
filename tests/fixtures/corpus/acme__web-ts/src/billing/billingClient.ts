import { BillingServiceClient } from "../gen/billing_grpc_pb";

export class RefundService {
  private billing = new BillingServiceClient("billing:7070");

  /** Issues a refund through the Go billing service over gRPC-web. */
  async refundCharge(chargeId: string, amountMinor: number): Promise<string> {
    const res = await this.billing.refund({ chargeId, amountMinor });
    return res.refundId;
  }
}
