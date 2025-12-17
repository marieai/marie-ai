---
sidebar_position: 4
---

# Documentation: Structured Region System for Document Processing

This document provides comprehensive training material for understanding the structured region system used in document processing, covering roles, role hints, and key data structures.

## Overview

The structured region system is designed to parse and organize document content into hierarchical, semantic structures. It transforms raw document annotations into structured data that can be processed by downstream systems for field extraction, table processing, and content analysis.

## Core Concepts

### Roles and Role Hints

Role hints provide the critical bridge between static configuration and dynamic processing logic. They enable:

1. **Selective Processing**: Only relevant sections get processed by appropriate handlers
2. **Field Scoping**: Different annotation selectors apply to different section types  
3. **Business Logic**: Custom processing rules based on document semantics
4. **Extensibility**: Easy addition of new section types without code changes
5. **Maintainability**: Clear separation between layout roles and business logic


#### SectionRole (Layout Roles)
`SectionRole` is an enum that defines the **layout-level positioning** and **processing priority** of sections within a document:

```python
class SectionRole(str, Enum):
    CONTEXT_ABOVE = "context_above"    # Header-like context (e.g., claim info)
    MAIN = "main"                      # Primary content (e.g., service lines table)
    CONTEXT_BELOW = "context_below"    # Footer-like context (e.g., totals)
    SIDEBAR = "sidebar"                # Auxiliary content
    UNKNOWN = "unknown"                # Fallback for unclassified content
```


**Purpose:**
- **Layout Control**: Determines where sections appear in the final document structure
- **Processing Priority**: `MAIN` sections are typically processed first and most thoroughly
- **Page Flow**: Controls how content flows across multi-page documents

#### Role Hints (Semantic Roles)
Role hints are **free-form semantic labels** stored in `section.tags["role_hint"]` that provide business-specific meaning:

**Examples:**
- `"claim_information"` - Contains claim details (numbers, patient info)
- `"service_lines"` - Medical service billing data
- `"remarks"` - Remark codes and descriptions
- `"claim_totals"` - Financial summary information

**Purpose:**
- **Downstream Processing**: Different processors handle sections based on role hints
- **Field Mapping**: Determines which annotation selectors apply to which sections
- **Business Logic**: Enables section-specific processing rules

#### Role Normalization Process
The `normalize_role()` function handles the conversion from configuration strings to proper roles:

```python
def normalize_role(value: str) -> Tuple[SectionRole, Optional[str]]:
    # Known enum values → SectionRole enum
    if value in ["main", "context_above", "context_below", "sidebar"]:
        return SectionRole.MAIN, None  # (example)
    
    # Unknown strings → semantic role hint
    else:
        return SectionRole.UNKNOWN, "custom_business_role"
```


---

## Core Data Structures

### 1. KeyValue
**Purpose**: Represents a single key-value pair extracted from document content.

```python
class KeyValue(BaseModel):
    key: str                           # Field name (e.g., "CLAIM_NUMBER")
    value: str                         # Extracted value (e.g., "Z7P4M2Q")
    value_type: ValueType             # Data type classification
    normalized: Optional[str]         # Cleaned/standardized value
    source: Optional[TextSpan]        # Source location in document
    tags: Dict[str, str]              # Additional metadata
```


**Use Cases:**
- Single field extractions (claim numbers, dates, amounts)
- Header information
- Summary/totals data

### 2. KVList
**Purpose**: Container for multiple related key-value pairs forming a logical group.

```python
class KVList(BlockBase):
    type: Literal["kvlist"] = "kvlist"
    items: List[KeyValue]             # Collection of related KV pairs
```


**Use Cases:**
- Claim information sections (patient name, claim number, provider)
- Summary totals (deductibles, copays, allowed amounts)
- Configuration parameters

### 3. RowRole
**Purpose**: Classifies the semantic function of table rows.

```python
class RowRole(str, Enum):
    HEADER = "header"        # Column headers
    SUBHEADER = "subheader"  # Secondary headers
    BODY = "body"            # Data rows
    SECTION = "section"      # Section dividers
    FOOTER = "footer"        # Table footers
    TOTALS = "totals"        # Summary/calculation rows
    SPACER = "spacer"        # Visual separation
```


**Use Cases:**
- Table parsing and validation
- Data extraction prioritization
- Display formatting

### 4. TableRow
**Purpose**: Represents a single row within a table structure with metadata.

```python
class TableRow(BaseModel):
    role: RowRole                     # Semantic function of the row
    cells: List[CellWithMeta]         # Cell content with metadata
    source_page: Optional[int]        # Originating page
    source_line_ids: Optional[List[int]]  # Line numbers for traceability
    tags: Dict[str, str]              # Additional metadata
```


**Use Cases:**
- Service line data (dates, procedures, amounts)
- Multi-page table reconstruction
- Data validation and error tracking

### 5. TableSeries
**Purpose**: Manages multi-page tables as a cohesive unit.

```python
class TableSeries(BlockBase):
    type: Literal["table_series"] = "table_series"
    series_id: Optional[str]          # Unique identifier
    segments: List[TableBlock]        # Page-specific table segments
    unified_header: Optional[List[str]]   # Consistent column headers
```


**Key Methods:**
- `pages()`: Returns all pages containing table data
- `iter_rows(roles)`: Iterates through rows filtered by role

**Use Cases:**
- Service lines spanning multiple pages
- Large datasets requiring pagination
- Cross-page data correlation

### 6. Section
**Purpose**: Groups related content blocks under a semantic heading.

```python
class Section(BaseModel):
    title: Optional[str]              # Section heading
    role: SectionRole                 # Layout positioning
    blocks: List[Block]               # Content (KVList, TableSeries, etc.)
    span: Optional[PageSpan]          # Multi-page footprint
    tags: Dict[str, str]              # Including role_hint
```


**Use Cases:**
- Organizing document into logical parts
- Applying section-specific processing rules
- Maintaining document hierarchy

### 7. StructuredRegion
**Purpose**: Top-level container representing a complete logical document unit.

```python
class StructuredRegion(BaseModel):
    region_id: Optional[str]          # Business identifier
    span: Optional[PageSpan]          # Complete page coverage
    parts: List[RegionPart]           # Page-specific content
    tags: Dict[str, str]              # Region-level metadata
```


**Key Methods:**
- `sections_flat()`: All sections in page order
- `find_section(name)`: Locate section by title
- `table_series()`: Extract all table data

**Use Cases:**
- Complete claim processing units
- Document-level validation
- Multi-page document assembly

### 8. ValueType
**Purpose**: Classifies the semantic type of extracted values for validation and processing.

```python
class ValueType(str, Enum):
    STRING = "string"     # General text
    NUMBER = "number"     # Numeric values
    MONEY = "money"       # Currency amounts
    DATE = "date"         # Date/time values
    CODE = "code"         # Structured codes (procedure, diagnosis)
    NAME = "name"         # Person/entity names
    ID = "id"             # Identifiers
    UNKNOWN = "unknown"   # Unclassified
```


**Use Cases:**
- Data validation
- Format standardization
- Type-specific processing rules

---

## Builder Functions

### build_structured_region()
**Purpose**: Constructs a complete StructuredRegion from sections and page spans.

**Key Features:**
- **Validation**: Prevents untitled or duplicate sections
- **Page Distribution**: Automatically distributes sections across pages based on spans
- **Span Management**: Handles multi-page content coordination

**Usage Pattern:**
```python
region = build_structured_region(
    region_id="claim_12345",
    region_span=aggregate_span,  # Covers all relevant pages
    sections=[claim_info_section, service_lines_section, totals_section]
)
```


### build_table_series_from_pagespan()
**Purpose**: Creates TableSeries from page-distributed table data.

**Key Features:**
- **Multi-page Coordination**: Links table segments across pages
- **Header Management**: Maintains consistent column headers
- **Row Association**: Preserves row relationships across page breaks

**Usage Pattern:**
```python
series = build_table_series_from_pagespan(
    series_id="service_lines_claim_12345",
    pagespan=table_span,
    table=table_obj,
    all_rows=table_rows,
    header_binding=["Date", "Procedure", "Amount"]
)
```


---

## Configuration Examples

### Layer Configuration with Roles
```yaml
layers:
  layer-main:
    region_parser:
      parsing_method: mrp
      sections:
        - title: CLAIM INFORMATION
          role: claim_information    # → role_hint
          parse: kv
        - title: SERVICE LINES
          role: main                # → SectionRole.MAIN
          parse: table
        - title: CLAIM TOTALS
          role: context_below       # → SectionRole.CONTEXT_BELOW
          parse: kv

  layer-remarks:
    region_parser:
      parsing_method: section_processor
    regions:
      - title: REMARK CODES
        role: remarks              # → role_hint
```


### Processing Flow
1. **Configuration Parse**: Role strings converted via `normalize_role()`
2. **Section Creation**: Sections created with proper SectionRole + role_hint tags
3. **Content Processing**: Different processors handle sections based on role_hints
4. **Field Mapping**: Annotation selectors applied based on semantic roles
5. **Output Generation**: Structured data ready for downstream consumers

---

## Best Practices

### Role Assignment
- **Use standard SectionRole enums** when content fits layout patterns
- **Use descriptive role_hints** for business-specific processing
- **Be consistent** with role naming across similar document types

### Section Organization
- **One primary MAIN section** per region (typically the largest table)
- **Context sections** for supporting information
- **Clear, unique titles** for all sections

### Multi-page Handling
- **Use TableSeries** for tables spanning pages
- **Maintain header consistency** across page breaks
- **Preserve row ordering** and relationships

### Validation
- **Check for required sections** in business logic
- **Validate data types** using ValueType
- **Ensure span coverage** matches actual content

This documentation provides the foundation for understanding how the structured region system organizes document content into processable, semantic structures that support both layout-based and business-specific processing requirements.




# Documentation: Role Hints in the Extract Visitor System

## How Role Hints Are Used in Production

Based on the system architecture, here's how `role_hint` values are utilized in the extraction visitor to enable dynamic, business-specific processing:

---

## Role Hint Processing Flow

### 1. **Section Processing Discovery**
When the extraction visitor processes structured regions, it examines each section's role hint to determine the appropriate processing method:

```python
# In the match_section_extract_visitor.py
role_hint = structured_section.tags.get("role_hint")
if not role_hint:
    # Skip sections without role hints or use default processing
    continue
```


### 2. **Processing Method Selection**
The system uses role hints to route sections to specialized processors:

```python
# Example processing logic
for rule in processing_rules:
    if rule.get("role") == role_hint:
        parse_method = rule.get("parse")  # "kv", "table", "custom"
        
        if parse_method == "table":
            self._process_region_as_table(context, structured_section, rule)
        elif parse_method == "kv":
            self._process_region_as_kv(context, structured_section, rule)
        elif parse_method == "custom":
            self._process_custom_region(context, structured_section, rule)
        
        break
else:
    self.logger.warning(
        f"No processing rule found for role_hint '{role_hint}'"
    )
```


### 3. **Field Mapping Based on Role Hints**
The system maps different annotation selectors to fields based on the section's role hint:

```python
# Field mapping lookup by role hint
scoped_field_mappings = [
    fm for fm in field_mappings 
    if fm.scope == FieldScope.REGION and fm.role == role_hint
]

if not scoped_field_mappings:
    self.logger.warning(
        f"No REGION-scoped field mappings with role '{role_hint}' "
        f"found for section '{structured_section.title}'"
    )
```


---

## Practical Usage Examples

### Example 1: Service Lines Processing

**Configuration:**
```yaml
layers:
  layer-main:
    region_parser:
      sections:
        - title: SERVICE LINES
          role: service_lines    # Becomes role_hint
          parse: table

    regions:
      - title: SERVICE LINES
        role: service_lines
        type: table
        table:
          columns:
            DATE_OF_SERVICE:
              annotation_selectors: ["DATES_OF_SERVICE"]
            PROCEDURE_CODE:
              annotation_selectors: ["PROCEDURE_DESCRIPTION"]
            BILLED_AMOUNT:
              annotation_selectors: ["BILLED_AMOUNT"]
```


**Processing Result:**
1. Section created with `role_hint = "service_lines"`
2. Extract visitor finds rule for `role = "service_lines"`
3. Processes section as table with specific column mappings
4. Applies annotation selectors: `DATES_OF_SERVICE`, `PROCEDURE_DESCRIPTION`, `BILLED_AMOUNT`

### Example 2: Remarks Processing

**Configuration:**
```yaml
layers:
  layer-remarks:
    region_parser:
      parsing_method: section_processor
    regions:
      - title: REMARK CODES
        role: remarks    # Becomes role_hint
```


**Processing Result:**
1. Section created with `role_hint = "remarks"`
2. Region processor (`remarks_processor`) handles sections with `role = "remarks"`
3. Custom logic parses remark codes and descriptions
4. Outputs structured TableSeries with remark data

### Example 3: Claim Information Processing

**Configuration:**
```yaml
layers:
  layer-main:
    region_parser:
      sections:
        - title: CLAIM INFORMATION
          role: claim_information    # Becomes role_hint
          parse: kv

    regions:
      - title: CLAIM INFORMATION
        role: claim_information
        type: kv
        fields:
          CLAIM_NUMBER:
            annotation_selectors: ["CLAIM NUMBER"]
          PATIENT_NAME:
            annotation_selectors: ["PATIENT NAME"]
```


**Processing Result:**
1. Section created with `role_hint = "claim_information"`
2. Extract visitor processes as key-value pairs
3. Maps specific fields using annotation selectors
4. Creates KVList with claim information

---

## Advanced Role Hint Usage

### 1. **Conditional Processing Rules**
```python
def process_section_by_role_hint(self, section, role_hint):
    """Process sections based on their role hint."""
    
    if role_hint == "service_lines":
        # Complex table processing with grouping
        self._process_service_lines_table(section)
        
    elif role_hint == "claim_totals":
        # Financial validation and calculation
        self._process_financial_totals(section)
        
    elif role_hint == "remarks":
        # Code normalization and lookup
        self._process_remark_codes(section)
        
    elif role_hint.startswith("custom_"):
        # Business-specific custom processing
        self._process_custom_business_logic(section, role_hint)
```


### 2. **Role Hint Inheritance**
```python
def inherit_processing_rules(self, parent_role_hint, section):
    """Inherit processing behavior from parent sections."""
    
    if parent_role_hint == "claim_information":
        # All child sections inherit claim processing rules
        section.tags["inherited_role"] = parent_role_hint
        section.tags["processing_mode"] = "claim_focused"
```


### 3. **Multi-Role Processing**
```python
def process_composite_roles(self, section):
    """Handle sections that serve multiple purposes."""
    
    role_hint = section.tags.get("role_hint")
    secondary_roles = section.tags.get("secondary_roles", [])
    
    # Primary processing
    self.process_by_role(section, role_hint)
    
    # Additional processing for secondary roles
    for secondary_role in secondary_roles:
        self.apply_secondary_processing(section, secondary_role)
```


---

## Configuration Patterns

### Pattern 1: Role-Specific Field Scoping
```yaml
# Different fields extracted based on role hint
non_repeating_fields:
  CLAIM_NUMBER:
    scope: REGION
    role: claim_information    # Only applies to sections with this role_hint
    annotation_selectors: ["CLAIM NUMBER"]
    
  REMARK_CODE:
    scope: REGION  
    role: remarks             # Only applies to remark sections
    annotation_selectors: ["REMARKS_CODE"]
```


### Pattern 2: Processing Method Override
```yaml
processing_rules:
  - role: service_lines
    parse: table
    validation: strict        # Custom processing options
    
  - role: remarks  
    parse: custom
    processor: remarks_processor
    
  - role: claim_totals
    parse: kv
    post_process: financial_validation
```


### Pattern 3: Cross-Section Dependencies  
```yaml
dependencies:
  - source_role: claim_information
    target_role: service_lines
    relationship: validates_against
    
  - source_role: service_lines
    target_role: claim_totals
    relationship: sums_to
```


---

## Best Practices for Role Hint Usage

### 1. **Consistent Naming**
- Use descriptive, business-meaningful names
- Follow snake_case convention: `claim_information`, `service_lines`, `claim_totals`
- Avoid generic names like `section1`, `data`, `content`

### 2. **Hierarchical Organization**  
- Use prefixes for related sections: `claim_info`, `claim_totals`, `claim_notes`
- Group similar processing types: `table_service_lines`, `table_procedures`

### 3. **Role Hint Documentation**
```python
# Document expected role hints in your processor
EXPECTED_ROLE_HINTS = {
    "service_lines": "Primary billing data table with dates, procedures, amounts",
    "claim_information": "Key-value pairs with claim identifiers and patient info", 
    "claim_totals": "Financial summary with totals, deductibles, payments",
    "remarks": "Remark codes and descriptions for billing clarifications"
}
```


### 4. **Validation and Error Handling**
```python
def validate_role_hint_support(self, role_hint):
    """Ensure role hint is supported by current configuration."""
    
    if role_hint not in self.supported_role_hints:
        raise ValueError(
            f"Unsupported role_hint '{role_hint}'. "
            f"Supported hints: {list(self.supported_role_hints.keys())}"
        )
```


---

## Summary

Role hints provide the critical bridge between static configuration and dynamic processing logic. They enable:

1. **Selective Processing**: Only relevant sections get processed by appropriate handlers
2. **Field Scoping**: Different annotation selectors apply to different section types  
3. **Business Logic**: Custom processing rules based on document semantics
4. **Extensibility**: Easy addition of new section types without code changes
5. **Maintainability**: Clear separation between layout roles and business logic

The role hint system transforms the structured region framework from a generic document parser into a flexible, business-aware extraction engine that can adapt to different document types and processing requirements through configuration alone.