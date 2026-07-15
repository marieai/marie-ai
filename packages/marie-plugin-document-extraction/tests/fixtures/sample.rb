require "json"

module Invoicing
  class InvoiceParser
    def parse(path)
      JSON.parse(File.read(path))
    end
  end

  def self.total_amount(items)
    items.sum { |item| item["amount"] }
  end
end
