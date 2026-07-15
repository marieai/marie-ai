import Foundation

class InvoiceParser {
    func parse(path: String) -> [Double] {
        return []
    }
}

func totalAmount(_ items: [Double]) -> Double {
    return items.reduce(0, +)
}
