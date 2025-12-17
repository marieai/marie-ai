---
sidebar_position: 1
---

# Invoice processing tutorial

This tutorial walks you through building a complete invoice processing pipeline with Marie-AI. You'll learn to extract key information like invoice numbers, dates, line items, and totals from invoice documents.

## What you'll build

By the end of this tutorial, you'll have a working pipeline that:

1. Accepts invoice images or PDFs
2. Performs OCR to extract text
3. Identifies invoice fields (number, date, vendor, total)
4. Extracts line items with descriptions and amounts
5. Returns structured JSON data

## Prerequisites

- Marie-AI installed (`pip install marie-ai`)
- Python 3.10+
- Basic understanding of [Executors](../guides/executor.md) and [Flows](../guides/flow.md)

## Project structure

Create the following project structure:

```text
invoice-processor/
├── executors/
│   ├── __init__.py
│   ├── ocr_executor.py
│   └── extraction_executor.py
├── models/
│   └── (model files go here)
├── flow.yml
├── client.py
└── requirements.txt
```

## Step 1: Define document types

Create `executors/__init__.py` with the document schemas:

```python
from docarray import BaseDoc, DocList
from typing import Optional, List
from pydantic import Field

class InvoiceInput(BaseDoc):
    """Input document containing invoice image or PDF."""
    asset_key: str = Field(..., description='Path to invoice file')
    pages: List[int] = Field(default=[1], description='Pages to process')

class LineItem(BaseDoc):
    """A single line item from an invoice."""
    description: str = ''
    quantity: float = 0.0
    unit_price: float = 0.0
    amount: float = 0.0
    confidence: float = 0.0

class InvoiceOutput(BaseDoc):
    """Extracted invoice data."""
    # Source info
    asset_key: str = ''

    # Header fields
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None

    # Vendor info
    vendor_name: Optional[str] = None
    vendor_address: Optional[str] = None

    # Customer info
    customer_name: Optional[str] = None
    customer_address: Optional[str] = None

    # Financial data
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None
    currency: str = 'USD'

    # Line items
    line_items: List[LineItem] = Field(default_factory=list)

    # Metadata
    confidence: float = 0.0
    raw_text: str = ''
    processing_time_ms: float = 0.0
```

## Step 2: Create the OCR executor

Create `executors/ocr_executor.py`:

```python
import time
from marie import Executor, requests
from marie.executor.marie_executor import MarieExecutor
from docarray import DocList
from . import InvoiceInput, InvoiceOutput

class OCRExecutor(MarieExecutor):
    """Performs OCR on invoice images."""

    def __init__(
        self,
        model_path: str = None,
        device: str = 'cuda',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.device = device
        self.model_path = model_path
        # Initialize your OCR model here
        # self.ocr_model = load_model(model_path)
        self.logger.info(f'OCR Executor initialized on {device}')

    @requests(on='/ocr')
    async def extract_text(
        self,
        docs: DocList[InvoiceInput],
        parameters: dict = None,
        **kwargs
    ) -> DocList[InvoiceOutput]:
        """Extract text from invoice images."""
        start_time = time.time()
        results = DocList[InvoiceOutput]()

        for doc in docs:
            self.logger.info(f'Processing: {doc.asset_key}')

            # Load the document
            # In production, load from asset_key (S3, local path, etc.)

            # Perform OCR
            # raw_text = self.ocr_model.extract(image)

            # For this example, simulate OCR output
            raw_text = self._simulate_ocr(doc.asset_key)

            results.append(InvoiceOutput(
                asset_key=doc.asset_key,
                raw_text=raw_text,
                processing_time_ms=(time.time() - start_time) * 1000
            ))

        return results

    def _simulate_ocr(self, asset_key: str) -> str:
        """Simulate OCR output for demonstration."""
        return """
        ACME Corporation
        123 Business Ave, Suite 100
        New York, NY 10001

        INVOICE

        Invoice #: INV-2024-0042
        Date: January 15, 2024
        Due Date: February 15, 2024

        Bill To:
        Example Customer Inc.
        456 Client Street
        Boston, MA 02101

        Description                    Qty    Unit Price    Amount
        ─────────────────────────────────────────────────────────────
        Professional Services          10     $150.00       $1,500.00
        Software License               1      $500.00       $500.00
        Support Package (Annual)       1      $1,200.00     $1,200.00

        ─────────────────────────────────────────────────────────────
                                       Subtotal:           $3,200.00
                                       Tax (8%):           $256.00
                                       TOTAL:              $3,456.00

        Payment Terms: Net 30
        Thank you for your business!
        """
```

## Step 3: Create the extraction executor

Create `executors/extraction_executor.py`:

```python
import re
import time
from marie import Executor, requests
from docarray import DocList
from . import InvoiceOutput, LineItem

class ExtractionExecutor(Executor):
    """Extracts structured data from OCR text."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Define extraction patterns
        self.patterns = {
            'invoice_number': r'Invoice\s*#?\s*:?\s*([A-Z0-9\-]+)',
            'invoice_date': r'Date\s*:?\s*(\w+\s+\d{1,2},?\s+\d{4})',
            'due_date': r'Due\s*Date\s*:?\s*(\w+\s+\d{1,2},?\s+\d{4})',
            'subtotal': r'Subtotal\s*:?\s*\$?([\d,]+\.?\d*)',
            'tax': r'Tax[^:]*:?\s*\$?([\d,]+\.?\d*)',
            'total': r'TOTAL\s*:?\s*\$?([\d,]+\.?\d*)',
        }
        self.logger.info('Extraction Executor initialized')

    @requests(on='/extract')
    async def extract_fields(
        self,
        docs: DocList[InvoiceOutput],
        parameters: dict = None,
        **kwargs
    ) -> DocList[InvoiceOutput]:
        """Extract structured fields from OCR text."""
        start_time = time.time()

        for doc in docs:
            text = doc.raw_text

            # Extract header fields
            doc.invoice_number = self._extract_pattern(text, 'invoice_number')
            doc.invoice_date = self._extract_pattern(text, 'invoice_date')
            doc.due_date = self._extract_pattern(text, 'due_date')

            # Extract financial data
            doc.subtotal = self._extract_amount(text, 'subtotal')
            doc.tax = self._extract_amount(text, 'tax')
            doc.total = self._extract_amount(text, 'total')

            # Extract vendor info
            doc.vendor_name = self._extract_vendor_name(text)
            doc.vendor_address = self._extract_address(text, 'vendor')

            # Extract customer info
            doc.customer_name = self._extract_customer_name(text)
            doc.customer_address = self._extract_address(text, 'customer')

            # Extract line items
            doc.line_items = self._extract_line_items(text)

            # Calculate confidence
            doc.confidence = self._calculate_confidence(doc)

            # Update processing time
            doc.processing_time_ms += (time.time() - start_time) * 1000

        return docs

    def _extract_pattern(self, text: str, pattern_name: str) -> str:
        """Extract a field using regex pattern."""
        pattern = self.patterns.get(pattern_name)
        if pattern:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_amount(self, text: str, pattern_name: str) -> float:
        """Extract a monetary amount."""
        value = self._extract_pattern(text, pattern_name)
        if value:
            # Remove commas and convert to float
            return float(value.replace(',', ''))
        return None

    def _extract_vendor_name(self, text: str) -> str:
        """Extract vendor name (first non-empty line)."""
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('INVOICE'):
                return line
        return None

    def _extract_customer_name(self, text: str) -> str:
        """Extract customer name after 'Bill To:'."""
        match = re.search(r'Bill\s*To\s*:?\s*\n\s*(.+)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_address(self, text: str, address_type: str) -> str:
        """Extract address based on context."""
        # Simplified address extraction
        address_pattern = r'(\d+[^,\n]+,\s*[A-Za-z\s]+,?\s*[A-Z]{2}\s*\d{5})'
        matches = re.findall(address_pattern, text)
        if matches:
            if address_type == 'vendor' and len(matches) >= 1:
                return matches[0]
            elif address_type == 'customer' and len(matches) >= 2:
                return matches[1]
        return None

    def _extract_line_items(self, text: str) -> list:
        """Extract line items from invoice."""
        line_items = []

        # Pattern for line items: Description, Qty, Unit Price, Amount
        pattern = r'([A-Za-z][A-Za-z\s\(\)]+)\s+(\d+)\s+\$?([\d,]+\.?\d*)\s+\$?([\d,]+\.?\d*)'

        matches = re.findall(pattern, text)
        for match in matches:
            description, qty, unit_price, amount = match
            line_items.append(LineItem(
                description=description.strip(),
                quantity=float(qty),
                unit_price=float(unit_price.replace(',', '')),
                amount=float(amount.replace(',', '')),
                confidence=0.85
            ))

        return line_items

    def _calculate_confidence(self, doc: InvoiceOutput) -> float:
        """Calculate overall extraction confidence."""
        fields = [
            doc.invoice_number,
            doc.invoice_date,
            doc.total,
            doc.vendor_name,
        ]
        extracted = sum(1 for f in fields if f is not None)
        return extracted / len(fields)
```

## Step 4: Create the flow configuration

Create `flow.yml`:

```yaml
jtype: Flow
version: '1'
with:
  protocol: [grpc, http]
  port: [54321, 54322]
  cors: true
executors:
  - name: ocr
    uses: OCRExecutor
    py_modules:
      - executors/ocr_executor.py
    uses_with:
      device: cuda
  - name: extractor
    uses: ExtractionExecutor
    py_modules:
      - executors/extraction_executor.py
    needs: ocr
```

## Step 5: Create the client

Create `client.py`:

```python
import json
from marie import Client
from docarray import DocList
from executors import InvoiceInput, InvoiceOutput

def process_invoice(asset_key: str) -> dict:
    """Process a single invoice and return extracted data."""

    # Connect to the service
    client = Client(host='localhost', port=54321, protocol='grpc')

    # Check if service is ready
    if not client.is_flow_ready():
        raise RuntimeError('Invoice processing service is not available')

    # Create input document
    input_doc = InvoiceInput(asset_key=asset_key, pages=[1])

    # Send for processing through the full pipeline
    # First OCR, then extraction
    results = client.post(
        on='/extract',
        inputs=DocList[InvoiceInput]([input_doc]),
        return_type=DocList[InvoiceOutput]
    )

    if not results:
        raise RuntimeError('No results returned')

    # Convert to dictionary
    result = results[0]
    return {
        'invoice_number': result.invoice_number,
        'invoice_date': result.invoice_date,
        'due_date': result.due_date,
        'vendor': {
            'name': result.vendor_name,
            'address': result.vendor_address,
        },
        'customer': {
            'name': result.customer_name,
            'address': result.customer_address,
        },
        'financials': {
            'subtotal': result.subtotal,
            'tax': result.tax,
            'total': result.total,
            'currency': result.currency,
        },
        'line_items': [
            {
                'description': item.description,
                'quantity': item.quantity,
                'unit_price': item.unit_price,
                'amount': item.amount,
            }
            for item in result.line_items
        ],
        'metadata': {
            'confidence': result.confidence,
            'processing_time_ms': result.processing_time_ms,
        }
    }

def main():
    # Process an invoice
    invoice_path = '/path/to/invoice.pdf'

    try:
        result = process_invoice(invoice_path)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f'Error processing invoice: {e}')

if __name__ == '__main__':
    main()
```

## Step 6: Run the pipeline

1. Start the Flow:

```bash
marie flow --uses flow.yml
```

2. In another terminal, run the client:

```bash
python client.py
```

Expected output:

```json
{
  "invoice_number": "INV-2024-0042",
  "invoice_date": "January 15, 2024",
  "due_date": "February 15, 2024",
  "vendor": {
    "name": "ACME Corporation",
    "address": "123 Business Ave, Suite 100, New York, NY 10001"
  },
  "customer": {
    "name": "Example Customer Inc.",
    "address": "456 Client Street, Boston, MA 02101"
  },
  "financials": {
    "subtotal": 3200.0,
    "tax": 256.0,
    "total": 3456.0,
    "currency": "USD"
  },
  "line_items": [
    {
      "description": "Professional Services",
      "quantity": 10,
      "unit_price": 150.0,
      "amount": 1500.0
    },
    {
      "description": "Software License",
      "quantity": 1,
      "unit_price": 500.0,
      "amount": 500.0
    },
    {
      "description": "Support Package",
      "quantity": 1,
      "unit_price": 1200.0,
      "amount": 1200.0
    }
  ],
  "metadata": {
    "confidence": 1.0,
    "processing_time_ms": 45.2
  }
}
```

## Step 7: Add batch processing

For processing multiple invoices, create `batch_client.py`:

```python
import json
import asyncio
from marie import Client
from docarray import DocList
from executors import InvoiceInput, InvoiceOutput

async def process_invoices_batch(invoice_paths: list) -> list:
    """Process multiple invoices in batch."""

    client = Client(host='localhost', port=54321, protocol='grpc', asyncio=True)

    # Create input documents
    inputs = DocList[InvoiceInput]([
        InvoiceInput(asset_key=path, pages=[1])
        for path in invoice_paths
    ])

    results = []

    # Process in batches with progress callback
    async for response in client.post(
        on='/extract',
        inputs=inputs,
        request_size=10,  # Process 10 at a time
    ):
        for doc in response:
            results.append({
                'asset_key': doc.asset_key,
                'invoice_number': doc.invoice_number,
                'total': doc.total,
                'confidence': doc.confidence,
            })
            print(f'Processed: {doc.asset_key} -> {doc.invoice_number}')

    return results

def main():
    invoice_paths = [
        '/invoices/invoice_001.pdf',
        '/invoices/invoice_002.pdf',
        '/invoices/invoice_003.pdf',
        # Add more paths...
    ]

    results = asyncio.run(process_invoices_batch(invoice_paths))
    print(f'\nProcessed {len(results)} invoices')
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
```

## Enhancing the pipeline

### Add validation

Create a validation executor to verify extracted data:

```python
from marie import Executor, requests
from docarray import DocList
from executors import InvoiceOutput

class ValidationExecutor(Executor):
    """Validates extracted invoice data."""

    @requests(on='/validate')
    async def validate(
        self,
        docs: DocList[InvoiceOutput],
        **kwargs
    ) -> DocList[InvoiceOutput]:
        for doc in docs:
            # Verify line items sum to subtotal
            if doc.line_items and doc.subtotal:
                calculated_subtotal = sum(item.amount for item in doc.line_items)
                if abs(calculated_subtotal - doc.subtotal) > 0.01:
                    doc.confidence *= 0.8  # Reduce confidence

            # Verify tax calculation
            if doc.subtotal and doc.tax and doc.total:
                calculated_total = doc.subtotal + doc.tax
                if abs(calculated_total - doc.total) > 0.01:
                    doc.confidence *= 0.9

        return docs
```

### Add storage

Persist results using the StorageMixin:

```python
from marie.executor.marie_executor import MarieExecutor
from marie.executor.mixin import StorageMixin

class PersistentExtractionExecutor(MarieExecutor, StorageMixin):
    def __init__(self, storage: dict = None, **kwargs):
        super().__init__(**kwargs)
        if storage:
            self.setup_storage(
                storage_enabled=True,
                storage_conf=storage.get('psql', {}),
                asset_tracking_enabled=True
            )
```

## Next steps

- Add [form extraction](./form-extraction.md) capabilities
- Learn about [Client](../guides/client.md) advanced features
- Deploy to [Kubernetes](../getting-started/deployment/kubernetes.md)
