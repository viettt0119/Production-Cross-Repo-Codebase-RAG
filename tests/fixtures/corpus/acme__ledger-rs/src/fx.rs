/// Static FX table (rate to USD, scaled by 1e6).
const RATES_TO_USD_MICRO: [(&str, i64); 3] = [("USD", 1_000_000), ("EUR", 1_080_000), ("VND", 40)];

fn rate(currency: &str) -> i64 {
    RATES_TO_USD_MICRO
        .iter()
        .find(|(c, _)| *c == currency)
        .map(|(_, r)| *r)
        .unwrap_or(1_000_000)
}

/// Converts an amount in minor units between currencies via USD.
pub fn convert_minor(amount_minor: i64, from: &str, to: &str) -> i64 {
    if from == to {
        return amount_minor;
    }
    amount_minor * rate(from) / rate(to)
}
