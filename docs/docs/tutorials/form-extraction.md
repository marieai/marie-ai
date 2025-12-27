---
sidebar_position: 2
---

# Form extraction tutorial

This tutorial shows you how to build a form extraction pipeline that identifies form fields, checkboxes, and tables from structured documents like applications, surveys, and registration forms.

## What you'll build

A form extraction system that:

1. Detects form field locations (labels and values)
2. Identifies checkbox states (checked/unchecked)
3. Extracts table data with headers and rows
4. Maps fields to a predefined schema
5. Returns structured data with confidence scores

## Prerequisites

- Marie-AI installed (`pip install marie-ai`)
- Completed the [Quickstart](../getting-started/quickstart.md)
- Familiarity with [Executors](../guides/executor.md)

## Project structure

```text
form-extractor/
├── executors/
│   ├── __init__.py
│   ├── field_detector.py
│   ├── checkbox_detector.py
│   └── schema_mapper.py
├── schemas/
│   ├── application_form.json
│   └── registration_form.json
├── flow.yml
├── client.py
└── requirements.txt
```

## Step 1: Define document types

Create `executors/__init__.py`:

```python
from docarray import BaseDoc, DocList
from typing import Optional, List, Dict, Any
from pydantic import Field
from enum import Enum

class FieldType(str, Enum):
    TEXT = 'text'
    DATE = 'date'
    NUMBER = 'number'
    CHECKBOX = 'checkbox'
    RADIO = 'radio'
    SIGNATURE = 'signature'
    TABLE = 'table'

class BoundingBox(BaseDoc):
    """Bounding box coordinates."""
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    page: int = 1

class FormField(BaseDoc):
    """A detected form field."""
    label: str = ''
    value: Optional[str] = None
    field_type: FieldType = FieldType.TEXT
    bbox: Optional[BoundingBox] = None
    confidence: float = 0.0
    is_required: bool = False

class Checkbox(BaseDoc):
    """A detected checkbox."""
    label: str = ''
    is_checked: bool = False
    bbox: Optional[BoundingBox] = None
    confidence: float = 0.0
    group: Optional[str] = None  # For grouped checkboxes

class TableCell(BaseDoc):
    """A single table cell."""
    row: int = 0
    col: int = 0
    value: str = ''
    is_header: bool = False
    bbox: Optional[BoundingBox] = None

class Table(BaseDoc):
    """A detected table."""
    name: Optional[str] = None
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    cells: List[TableCell] = Field(default_factory=list)
    bbox: Optional[BoundingBox] = None
    confidence: float = 0.0

class FormInput(BaseDoc):
    """Input document for form extraction."""
    asset_key: str = Field(..., description='Path to form image/PDF')
    pages: List[int] = Field(default=[1])
    schema_name: Optional[str] = None  # Optional schema to map to

class FormOutput(BaseDoc):
    """Extracted form data."""
    asset_key: str = ''
    form_type: Optional[str] = None

    # Detected elements
    fields: List[FormField] = Field(default_factory=list)
    checkboxes: List[Checkbox] = Field(default_factory=list)
    tables: List[Table] = Field(default_factory=list)

    # Mapped data (if schema provided)
    mapped_data: Dict[str, Any] = Field(default_factory=dict)

    # Metadata
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    raw_text: str = ''
```

## Step 2: Create the field detector

Create `executors/field_detector.py`:

```python
import re
import time
from marie import Executor, requests
from docarray import DocList
from . import FormInput, FormOutput, FormField, FieldType, BoundingBox

class FieldDetectorExecutor(Executor):
    """Detects and extracts form fields."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Field patterns for common form fields
        self.field_patterns = [
            # Pattern: (label_pattern, field_type, is_required)
            (r'Name\s*:?\s*(.+)', FieldType.TEXT, True),
            (r'Full\s*Name\s*:?\s*(.+)', FieldType.TEXT, True),
            (r'First\s*Name\s*:?\s*(.+)', FieldType.TEXT, True),
            (r'Last\s*Name\s*:?\s*(.+)', FieldType.TEXT, True),
            (r'Date\s*of\s*Birth\s*:?\s*(.+)', FieldType.DATE, True),
            (r'DOB\s*:?\s*(.+)', FieldType.DATE, True),
            (r'Email\s*:?\s*(.+)', FieldType.TEXT, False),
            (r'Phone\s*:?\s*(.+)', FieldType.TEXT, False),
            (r'Address\s*:?\s*(.+)', FieldType.TEXT, False),
            (r'City\s*:?\s*(.+)', FieldType.TEXT, False),
            (r'State\s*:?\s*(.+)', FieldType.TEXT, False),
            (r'Zip\s*(?:Code)?\s*:?\s*(.+)', FieldType.TEXT, False),
            (r'SSN\s*:?\s*(.+)', FieldType.TEXT, True),
            (r'Social\s*Security\s*:?\s*(.+)', FieldType.TEXT, True),
            (r'Date\s*:?\s*(.+)', FieldType.DATE, False),
            (r'Signature\s*:?\s*(.+)', FieldType.SIGNATURE, True),
            (r'Amount\s*:?\s*\$?\s*(.+)', FieldType.NUMBER, False),
            (r'Total\s*:?\s*\$?\s*(.+)', FieldType.NUMBER, False),
        ]
        self.logger.info('Field Detector initialized')

    @requests(on='/detect-fields')
    async def detect_fields(
        self,
        docs: DocList[FormInput],
        parameters: dict = None,
        **kwargs
    ) -> DocList[FormOutput]:
        """Detect form fields from document."""
        start_time = time.time()
        results = DocList[FormOutput]()

        for doc in docs:
            # In production, load and OCR the document
            # raw_text = self.ocr_model.extract(doc.asset_key)

            # Simulated OCR output for demonstration
            raw_text = self._get_sample_form_text()

            # Extract fields
            fields = self._extract_fields(raw_text)

            results.append(FormOutput(
                asset_key=doc.asset_key,
                fields=fields,
                raw_text=raw_text,
                processing_time_ms=(time.time() - start_time) * 1000
            ))

        return results

    def _extract_fields(self, text: str) -> list:
        """Extract form fields from text."""
        fields = []

        for pattern, field_type, is_required in self.field_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)

            for match in matches:
                label = self._extract_label(pattern)
                value = match.group(1).strip() if match.lastindex else None

                # Clean up value
                if value:
                    value = self._clean_value(value, field_type)

                fields.append(FormField(
                    label=label,
                    value=value,
                    field_type=field_type,
                    is_required=is_required,
                    confidence=0.85 if value else 0.5
                ))

        return fields

    def _extract_label(self, pattern: str) -> str:
        """Extract readable label from pattern."""
        # Remove regex special characters
        label = re.sub(r'\\s\*|\?|:|\(.*?\)|\+', '', pattern)
        label = label.replace('\\', '').strip()
        return label.title()

    def _clean_value(self, value: str, field_type: FieldType) -> str:
        """Clean extracted value based on type."""
        value = value.strip()

        # Remove trailing field labels that might have been captured
        value = re.sub(r'\s+(Name|Date|Phone|Email|Address|City|State|Zip).*$', '', value, flags=re.IGNORECASE)

        if field_type == FieldType.NUMBER:
            # Extract just the number
            match = re.search(r'[\d,]+\.?\d*', value)
            if match:
                value = match.group(0)

        return value.strip()

    def _get_sample_form_text(self) -> str:
        """Sample form text for demonstration."""
        return """
        APPLICATION FORM

        Personal Information
        ─────────────────────────────────────────────

        First Name: John                    Last Name: Smith
        Date of Birth: 03/15/1985          SSN: XXX-XX-1234

        Contact Information
        ─────────────────────────────────────────────

        Email: john.smith@email.com
        Phone: (555) 123-4567

        Address: 123 Main Street, Apt 4B
        City: New York                      State: NY
        Zip Code: 10001

        Employment
        ─────────────────────────────────────────────

        Employer: ABC Corporation
        Position: Software Engineer
        Annual Income: $85,000

        ☑ I agree to the terms and conditions
        ☐ Subscribe to newsletter
        ☑ I certify this information is accurate

        Signature: John Smith               Date: 01/15/2024
        """
```

## Step 3: Create the checkbox detector

Create `executors/checkbox_detector.py`:

```python
import re
from marie import Executor, requests
from docarray import DocList
from . import FormOutput, Checkbox

class CheckboxDetectorExecutor(Executor):
    """Detects checkboxes and their states."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Checkbox patterns
        self.checked_patterns = [
            r'☑\s*(.+)',           # Unicode checked box
            r'\[x\]\s*(.+)',       # [x] format
            r'\[X\]\s*(.+)',       # [X] format
            r'✓\s*(.+)',           # Checkmark
            r'✔\s*(.+)',           # Heavy checkmark
        ]
        self.unchecked_patterns = [
            r'☐\s*(.+)',           # Unicode unchecked box
            r'\[\s*\]\s*(.+)',     # [ ] format
            r'○\s*(.+)',           # Circle (radio unchecked)
        ]
        self.logger.info('Checkbox Detector initialized')

    @requests(on='/detect-checkboxes')
    async def detect_checkboxes(
        self,
        docs: DocList[FormOutput],
        parameters: dict = None,
        **kwargs
    ) -> DocList[FormOutput]:
        """Detect checkboxes from form text."""

        for doc in docs:
            checkboxes = []

            # Find checked checkboxes
            for pattern in self.checked_patterns:
                matches = re.finditer(pattern, doc.raw_text, re.MULTILINE)
                for match in matches:
                    label = match.group(1).strip()
                    checkboxes.append(Checkbox(
                        label=label,
                        is_checked=True,
                        confidence=0.9,
                        group=self._detect_group(label)
                    ))

            # Find unchecked checkboxes
            for pattern in self.unchecked_patterns:
                matches = re.finditer(pattern, doc.raw_text, re.MULTILINE)
                for match in matches:
                    label = match.group(1).strip()
                    checkboxes.append(Checkbox(
                        label=label,
                        is_checked=False,
                        confidence=0.9,
                        group=self._detect_group(label)
                    ))

            doc.checkboxes = checkboxes

        return docs

    def _detect_group(self, label: str) -> str:
        """Detect checkbox group from label content."""
        label_lower = label.lower()

        if 'agree' in label_lower or 'terms' in label_lower:
            return 'agreements'
        elif 'subscribe' in label_lower or 'newsletter' in label_lower:
            return 'subscriptions'
        elif 'certify' in label_lower or 'confirm' in label_lower:
            return 'certifications'

        return 'general'
```

## Step 4: Create the schema mapper

Create `executors/schema_mapper.py`:

```python
import json
from pathlib import Path
from marie import Executor, requests
from docarray import DocList
from . import FormOutput, FieldType

class SchemaMapperExecutor(Executor):
    """Maps extracted fields to a predefined schema."""

    def __init__(self, schemas_dir: str = './schemas', **kwargs):
        super().__init__(**kwargs)
        self.schemas_dir = Path(schemas_dir)
        self.schemas = {}
        self._load_schemas()
        self.logger.info(f'Schema Mapper initialized with {len(self.schemas)} schemas')

    def _load_schemas(self):
        """Load schema definitions from files."""
        if self.schemas_dir.exists():
            for schema_file in self.schemas_dir.glob('*.json'):
                try:
                    with open(schema_file) as f:
                        schema = json.load(f)
                        self.schemas[schema_file.stem] = schema
                except Exception as e:
                    self.logger.warning(f'Failed to load schema {schema_file}: {e}')

        # Add default schema
        self.schemas['default'] = self._get_default_schema()

    def _get_default_schema(self) -> dict:
        """Default form schema."""
        return {
            'name': 'default',
            'fields': {
                'first_name': {'aliases': ['first name', 'firstname', 'given name'], 'required': True},
                'last_name': {'aliases': ['last name', 'lastname', 'surname', 'family name'], 'required': True},
                'full_name': {'aliases': ['name', 'full name'], 'required': False},
                'date_of_birth': {'aliases': ['date of birth', 'dob', 'birth date', 'birthdate'], 'required': False},
                'email': {'aliases': ['email', 'e-mail', 'email address'], 'required': False},
                'phone': {'aliases': ['phone', 'telephone', 'phone number', 'tel'], 'required': False},
                'address': {'aliases': ['address', 'street address', 'street'], 'required': False},
                'city': {'aliases': ['city'], 'required': False},
                'state': {'aliases': ['state', 'province'], 'required': False},
                'zip_code': {'aliases': ['zip', 'zip code', 'postal code', 'zipcode'], 'required': False},
                'ssn': {'aliases': ['ssn', 'social security', 'social security number'], 'required': False},
                'signature': {'aliases': ['signature', 'sign'], 'required': False},
                'date': {'aliases': ['date', 'signed date'], 'required': False},
            },
            'checkboxes': {
                'terms_accepted': {'aliases': ['agree', 'terms', 'accept'], 'required': True},
                'newsletter': {'aliases': ['subscribe', 'newsletter', 'updates'], 'required': False},
                'certified': {'aliases': ['certify', 'confirm', 'accurate'], 'required': False},
            }
        }

    @requests(on='/map-schema')
    async def map_to_schema(
        self,
        docs: DocList[FormOutput],
        parameters: dict = None,
        **kwargs
    ) -> DocList[FormOutput]:
        """Map extracted fields to schema."""

        for doc in docs:
            # Determine which schema to use
            schema_name = parameters.get('schema_name') if parameters else None
            schema_name = schema_name or doc.form_type or 'default'

            schema = self.schemas.get(schema_name, self.schemas['default'])

            # Map fields
            mapped_data = {}

            # Map text fields
            for schema_field, config in schema.get('fields', {}).items():
                aliases = config.get('aliases', [])
                value = self._find_field_value(doc.fields, aliases)

                if value is not None:
                    mapped_data[schema_field] = value
                elif config.get('required', False):
                    mapped_data[schema_field] = None  # Mark required but missing

            # Map checkboxes
            for schema_checkbox, config in schema.get('checkboxes', {}).items():
                aliases = config.get('aliases', [])
                is_checked = self._find_checkbox_state(doc.checkboxes, aliases)
                mapped_data[schema_checkbox] = is_checked

            doc.mapped_data = mapped_data
            doc.form_type = schema_name

            # Calculate overall confidence
            doc.confidence = self._calculate_confidence(doc, schema)

        return docs

    def _find_field_value(self, fields: list, aliases: list) -> str:
        """Find field value matching any alias."""
        for field in fields:
            field_label_lower = field.label.lower()
            for alias in aliases:
                if alias.lower() in field_label_lower:
                    return field.value
        return None

    def _find_checkbox_state(self, checkboxes: list, aliases: list) -> bool:
        """Find checkbox state matching any alias."""
        for checkbox in checkboxes:
            checkbox_label_lower = checkbox.label.lower()
            for alias in aliases:
                if alias.lower() in checkbox_label_lower:
                    return checkbox.is_checked
        return False

    def _calculate_confidence(self, doc: FormOutput, schema: dict) -> float:
        """Calculate confidence based on required fields."""
        required_fields = [
            name for name, config in schema.get('fields', {}).items()
            if config.get('required', False)
        ]
        required_checkboxes = [
            name for name, config in schema.get('checkboxes', {}).items()
            if config.get('required', False)
        ]

        total_required = len(required_fields) + len(required_checkboxes)
        if total_required == 0:
            return 1.0

        found_count = 0

        for field_name in required_fields:
            if doc.mapped_data.get(field_name) is not None:
                found_count += 1

        for checkbox_name in required_checkboxes:
            if checkbox_name in doc.mapped_data:
                found_count += 1

        return found_count / total_required
```

## Step 5: Create the flow

Create `flow.yml`:

```yaml
jtype: Flow
version: '1'
with:
  protocol: [grpc, http]
  port: [54321, 54322]
executors:
  - name: field-detector
    uses: FieldDetectorExecutor
    py_modules:
      - executors/field_detector.py
  - name: checkbox-detector
    uses: CheckboxDetectorExecutor
    py_modules:
      - executors/checkbox_detector.py
    needs: field-detector
  - name: schema-mapper
    uses: SchemaMapperExecutor
    py_modules:
      - executors/schema_mapper.py
    uses_with:
      schemas_dir: ./schemas
    needs: checkbox-detector
```

## Step 6: Create the client

Create `client.py`:

```python
import json
from marie import Client
from docarray import DocList
from executors import FormInput, FormOutput

def extract_form(asset_key: str, schema_name: str = None) -> dict:
    """Extract data from a form document."""

    client = Client(host='localhost', port=54321, protocol='grpc')

    if not client.is_flow_ready():
        raise RuntimeError('Form extraction service is not available')

    input_doc = FormInput(asset_key=asset_key)

    # Process through the pipeline
    results = client.post(
        on='/map-schema',
        inputs=DocList[FormInput]([input_doc]),
        parameters={'schema_name': schema_name},
        return_type=DocList[FormOutput]
    )

    if not results:
        raise RuntimeError('No results returned')

    result = results[0]

    return {
        'form_type': result.form_type,
        'mapped_data': result.mapped_data,
        'fields': [
            {
                'label': f.label,
                'value': f.value,
                'type': f.field_type.value,
                'confidence': f.confidence,
            }
            for f in result.fields
        ],
        'checkboxes': [
            {
                'label': c.label,
                'checked': c.is_checked,
                'group': c.group,
            }
            for c in result.checkboxes
        ],
        'confidence': result.confidence,
        'processing_time_ms': result.processing_time_ms,
    }

def main():
    form_path = '/path/to/application_form.pdf'

    result = extract_form(form_path, schema_name='default')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
```

## Step 7: Run the pipeline

Start the Flow:

```bash
marie flow --uses flow.yml
```

Run the client:

```bash
python client.py
```

Expected output:

```json
{
  "form_type": "default",
  "mapped_data": {
    "first_name": "John",
    "last_name": "Smith",
    "date_of_birth": "03/15/1985",
    "email": "john.smith@email.com",
    "phone": "(555) 123-4567",
    "address": "123 Main Street, Apt 4B",
    "city": "New York",
    "state": "NY",
    "zip_code": "10001",
    "ssn": "XXX-XX-1234",
    "signature": "John Smith",
    "date": "01/15/2024",
    "terms_accepted": true,
    "newsletter": false,
    "certified": true
  },
  "fields": [
    {"label": "First Name", "value": "John", "type": "text", "confidence": 0.85},
    {"label": "Last Name", "value": "Smith", "type": "text", "confidence": 0.85}
  ],
  "checkboxes": [
    {"label": "I agree to the terms and conditions", "checked": true, "group": "agreements"},
    {"label": "Subscribe to newsletter", "checked": false, "group": "subscriptions"},
    {"label": "I certify this information is accurate", "checked": true, "group": "certifications"}
  ],
  "confidence": 1.0,
  "processing_time_ms": 32.5
}
```

## Custom schemas

Create custom schemas for specific form types. Save as `schemas/application_form.json`:

```json
{
  "name": "application_form",
  "description": "Job application form schema",
  "fields": {
    "applicant_name": {
      "aliases": ["name", "full name", "applicant name"],
      "required": true
    },
    "position": {
      "aliases": ["position", "job title", "role", "applying for"],
      "required": true
    },
    "employer": {
      "aliases": ["employer", "current employer", "company"],
      "required": false
    },
    "annual_income": {
      "aliases": ["income", "salary", "annual income", "annual salary"],
      "required": false
    },
    "start_date": {
      "aliases": ["start date", "available date", "availability"],
      "required": false
    }
  },
  "checkboxes": {
    "background_check_consent": {
      "aliases": ["background check", "consent to check"],
      "required": true
    },
    "relocate_willing": {
      "aliases": ["relocate", "willing to relocate"],
      "required": false
    }
  }
}
```

Use the custom schema:

```python
result = extract_form('/path/to/job_application.pdf', schema_name='application_form')
```

## Next steps

- See [Invoice processing](./invoice-processing.md) for financial documents
- Learn about [Executors](../guides/executor.md) for custom processing
- Explore [deployment options](../getting-started/deployment/index.md) for production
