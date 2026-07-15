#include <stdio.h>

struct LineItem {
    double amount;
};

double total_amount(const double *amounts, int count) {
    double total = 0;
    for (int i = 0; i < count; i++) {
        total += amounts[i];
    }
    return total;
}
