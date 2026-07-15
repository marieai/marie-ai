<?php

namespace Invoicing;

class InvoiceParser
{
    public function parse(string $path): array
    {
        return json_decode(file_get_contents($path), true);
    }
}

function total_amount(array $items): float
{
    return array_sum(array_column($items, 'amount'));
}
