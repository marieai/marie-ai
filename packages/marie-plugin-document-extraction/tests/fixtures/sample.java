package com.example.invoice;

import java.util.List;

public class InvoiceParser {

    public interface LineItem {
        String description();

        double amount();
    }

    private final String currency;

    public InvoiceParser(String currency) {
        this.currency = currency;
    }

    public double totalAmount(List<LineItem> items) {
        return items.stream().mapToDouble(LineItem::amount).sum();
    }
}
