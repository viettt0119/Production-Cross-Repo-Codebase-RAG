package main

import (
	"log"
	"net"

	"google.golang.org/grpc"

	pb "github.com/acme/billing-go/gen/billing/v1"
	"github.com/acme/billing-go/internal/ledger"
	"github.com/acme/billing-go/internal/server"
)

func main() {
	lis, err := net.Listen("tcp", ":7070")
	if err != nil {
		log.Fatalf("listen: %v", err)
	}
	s := grpc.NewServer()
	pb.RegisterBillingServiceServer(s, server.NewBillingServer(&ledger.Ledger{}))
	if err := s.Serve(lis); err != nil {
		log.Fatalf("serve: %v", err)
	}
}
