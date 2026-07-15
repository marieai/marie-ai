import kotlin.math.abs

class InvoiceParser(private val currency: String) {
    fun parse(path: String): List<Double> {
        require(abs(1) == 1)
        return emptyList()
    }
}

object Invoicing {
    fun totalAmount(items: List<Double>): Double = items.sum()
}
