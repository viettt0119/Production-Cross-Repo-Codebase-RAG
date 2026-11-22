pub mod fx;

use std::collections::HashMap;

/// A persisted ledger line in minor units.
#[derive(Debug, Clone)]
pub struct LedgerEntry {
    pub id: String,
    pub amount_minor: i64,
    pub currency: String,
}

/// Storage abstraction for ledger entries.
pub trait Store {
    fn append(&mut self, entry: LedgerEntry) -> Result<(), String>;
    fn balance(&self, currency: &str) -> i64;
}

#[derive(Default)]
pub struct MemoryStore {
    entries: Vec<LedgerEntry>,
    by_id: HashMap<String, usize>,
}

impl Store for MemoryStore {
    fn append(&mut self, entry: LedgerEntry) -> Result<(), String> {
        if self.by_id.contains_key(&entry.id) {
            return Err(format!("duplicate entry {}", entry.id));
        }
        self.by_id.insert(entry.id.clone(), self.entries.len());
        self.entries.push(entry);
        Ok(())
    }

    fn balance(&self, currency: &str) -> i64 {
        self.entries.iter().filter(|e| e.currency == currency).map(|e| e.amount_minor).sum()
    }
}

/// Balance across currencies, converted to a reporting currency.
pub fn reporting_balance(store: &MemoryStore, currencies: &[&str], target: &str) -> i64 {
    currencies
        .iter()
        .map(|c| fx::convert_minor(store.balance(c), c, target))
        .sum()
}
