#include <vector>

class InvoiceParser {
public:
    double totalAmount(const std::vector<double>& amounts) const;
};

double InvoiceParser::totalAmount(const std::vector<double>& amounts) const {
    double total = 0;
    for (double amount : amounts) {
        total += amount;
    }
    return total;
}
