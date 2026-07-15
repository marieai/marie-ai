import { readFileSync } from "node:fs";

export interface LineItem {
  description: string;
  amount: number;
}

export enum InvoiceStatus {
  Draft,
  Sent,
  Paid,
}

export type InvoiceId = string;

export class InvoiceParser {
  constructor(private readonly currency: string = "USD") {}

  parse(path: string): LineItem[] {
    return JSON.parse(readFileSync(path, "utf-8"));
  }
}

export function totalAmount(items: LineItem[]): number {
  return items.reduce((sum, item) => sum + item.amount, 0);
}
