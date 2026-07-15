package invoice

import (
	"encoding/json"
	"os"
)

type LineItem struct {
	Description string  `json:"description"`
	Amount      float64 `json:"amount"`
}

type Parser struct {
	Currency string
}

func (p *Parser) Parse(path string) ([]LineItem, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var items []LineItem
	err = json.Unmarshal(data, &items)
	return items, err
}

func TotalAmount(items []LineItem) float64 {
	total := 0.0
	for _, item := range items {
		total += item.Amount
	}
	return total
}
