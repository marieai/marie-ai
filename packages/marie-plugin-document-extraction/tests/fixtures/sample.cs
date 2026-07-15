using System.Collections.Generic;

namespace Invoicing
{
    public class InvoiceParser
    {
        public double TotalAmount(List<double> amounts)
        {
            double total = 0;
            foreach (var amount in amounts)
            {
                total += amount;
            }
            return total;
        }
    }
}
