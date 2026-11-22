package server

import (
	"context"
	"errors"

	pb "github.com/acme/billing-go/gen/billing/v1"
	"github.com/acme/billing-go/internal/ledger"
)

// ErrDuplicateCharge is returned when an idempotency key was already used.
var ErrDuplicateCharge = errors.New("duplicate charge")

// BillingServer implements the billing.v1.BillingService gRPC API.
type BillingServer struct {
	pb.UnimplementedBillingServiceServer
	ledger *ledger.Ledger
	seen   map[string]string
}

// NewBillingServer wires a server to the ledger.
func NewBillingServer(l *ledger.Ledger) *BillingServer {
	return &BillingServer{ledger: l, seen: map[string]string{}}
}

// Charge captures funds and records a ledger entry. Repeated idempotency keys
// return the original charge id instead of charging twice.
func (s *BillingServer) Charge(ctx context.Context, req *pb.ChargeRequest) (*pb.ChargeResponse, error) {
	if id, ok := s.seen[req.IdempotencyKey]; ok {
		return &pb.ChargeResponse{ChargeId: id, Status: "duplicate"}, nil
	}
	if req.AmountMinor <= 0 {
		return nil, errors.New("amount must be positive")
	}
	entry := s.ledger.Record(req.OrderId, req.AmountMinor, req.Currency)
	s.seen[req.IdempotencyKey] = entry.ID
	return &pb.ChargeResponse{ChargeId: entry.ID, Status: "captured"}, nil
}

// Refund reverses a captured charge by writing a negative ledger entry.
func (s *BillingServer) Refund(ctx context.Context, req *pb.RefundRequest) (*pb.RefundResponse, error) {
	entry, err := s.ledger.Reverse(req.ChargeId, req.AmountMinor)
	if err != nil {
		return nil, err
	}
	return &pb.RefundResponse{RefundId: entry.ID}, nil
}
