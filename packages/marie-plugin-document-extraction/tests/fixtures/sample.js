import { readFileSync } from "node:fs";

export class InvoiceParser {
  constructor(currency = "USD") {
    this.currency = currency;
  }

  parse(path) {
    return JSON.parse(readFileSync(path, "utf-8"));
  }
}

/** Sum the amount field across line items. */
function totalAmount(items) {
  return items.reduce((sum, item) => sum + item.amount, 0);
}

export { totalAmount };
