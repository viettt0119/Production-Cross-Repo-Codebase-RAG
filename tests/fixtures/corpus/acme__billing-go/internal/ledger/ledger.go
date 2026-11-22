package ledger

import (
	"errors"
	"fmt"
	"sync"
)

// Entry is an immutable double-entry ledger line.
type Entry struct {
	ID          string
	OrderID     string
	AmountMinor int64
	Currency    string
}

// Ledger is an append-only, in-memory ledger guarded by a mutex.
type Ledger struct {
	mu      sync.Mutex
	entries []Entry
}

// Record appends a positive entry for an order and returns it.
func (l *Ledger) Record(orderID string, amountMinor int64, currency string) Entry {
	l.mu.Lock()
	defer l.mu.Unlock()
	e := Entry{ID: nextID(len(l.entries)), OrderID: orderID, AmountMinor: amountMinor, Currency: currency}
	l.entries = append(l.entries, e)
	return e
}

// Reverse appends a negative entry referencing an existing charge.
func (l *Ledger) Reverse(chargeID string, amountMinor int64) (Entry, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	for _, e := range l.entries {
		if e.ID == chargeID {
			if amountMinor > e.AmountMinor {
				return Entry{}, errors.New("refund exceeds charge")
			}
			r := Entry{ID: nextID(len(l.entries)), OrderID: e.OrderID, AmountMinor: -amountMinor, Currency: e.Currency}
			l.entries = append(l.entries, r)
			return r, nil
		}
	}
	return Entry{}, fmt.Errorf("charge %s not found", chargeID)
}

func nextID(n int) string {
	return fmt.Sprintf("le_%06d", n+1)
}
