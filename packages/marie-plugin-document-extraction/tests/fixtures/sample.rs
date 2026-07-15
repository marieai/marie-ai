use std::collections::HashMap;

pub struct InvoiceParser {
    currency: String,
}

impl InvoiceParser {
    pub fn parse(&self, path: &str) -> HashMap<String, f64> {
        let _ = path;
        HashMap::new()
    }
}

pub fn total_amount(amounts: &[f64]) -> f64 {
    amounts.iter().sum()
}
