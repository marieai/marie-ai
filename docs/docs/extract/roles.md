# Role Hints

Role hints bridge static configuration with dynamic business logic, allowing the extraction system to apply different processing rules to sections based on their semantic purpose.

---

## Role Hint Processing Flow

### 1. **Section Discovery and Routing**
The extraction visitor examines each section's `role_hint` to determine appropriate processing:

- Extracts `role_hint` from `section.tags["role_hint"]`
- Matches against configured processing rules
- Routes to specialized processors based on business logic

### 2. **Processing Method Selection**
Different sections receive different treatment based on their role hints:

- **Tables**: Complex parsing, grouping, validation
- **Key-Value**: Field extraction and mapping
- **Custom**: Business-specific logic

### 3. **Field Mapping by Role**
The system applies different annotation selectors based on semantic context:

- Service line sections → procedure codes, amounts, dates
- Claim information sections → patient data, claim numbers
- Remark sections → code descriptions, references

---

## Configuration Patterns

### Pattern 1: Role-Based Processing
```yaml
layers:
  layer-main:
    region_parser:
      sections:
        - title: SERVICE LINES
          role: service_lines    # → role_hint
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


### Pattern 2: Multi-Layer Processing
```yaml
layers:
  layer-main:
    # Standard markdown regions
    region_parser:
      parsing_method: mrp
      sections:
        - title: CLAIM INFORMATION
          role: claim_information
          parse: kv

  layer-remarks:
    # Custom processor regions
    region_parser:
      parsing_method: section_processor
      processors:
        remarks_processor:
          role: remarks
          priority: 1
    regions:
      - title: REMARK CODES
        role: remarks
```


### Pattern 3: Field Scoping by Role
```yaml
non_repeating_fields:
  CLAIM_NUMBER:
    scope: REGION
    role: claim_information    # Only applies to this role_hint
    annotation_selectors: ["CLAIM NUMBER"]
    
  REMARK_CODE:
    scope: REGION
    role: remarks             # Only applies to remark sections
    annotation_selectors: ["REMARKS_CODE"]
```


---

## Business Use Cases

### Healthcare Claims Processing
- **`claim_information`**: Patient demographics, provider details
- **`service_lines`**: Medical procedures, dates, billing amounts  
- **`claim_totals`**: Financial summaries, deductibles, payments
- **`remarks`**: Billing codes, denial reasons, adjustments

### Financial Document Processing
- **`account_summary`**: Account numbers, balances, dates
- **`transactions`**: Transaction details, amounts, categories
- **`fees_charges`**: Service fees, interest charges
- **`payment_history`**: Payment dates, amounts, methods

### Legal Document Processing
- **`case_details`**: Case numbers, parties, dates
- **`evidence_list`**: Exhibit numbers, descriptions
- **`rulings`**: Court decisions, orders
- **`references`**: Citations, precedents

---

## Best Practices

### Naming Conventions
- Use descriptive, business-meaningful names
- Follow `snake_case`: `claim_information`, `service_lines`
- Group related sections with prefixes: `claim_info`, `claim_totals`

### Configuration Organization
- **Layer separation**: Group similar processing types
- **Role consistency**: Use same role names across similar document types
- **Clear hierarchy**: Primary content as `main`, supporting content as context roles

### Validation and Error Handling
- Validate role hints against supported values
- Provide meaningful error messages for unsupported roles
- Log processing decisions for debugging

---

## Summary

Role hints transform the structured region system from a generic parser into a business-aware extraction engine:

**Key Benefits:**
- **Selective Processing**: Only relevant sections get processed
- **Field Scoping**: Different extractors for different content types
- **Business Logic**: Custom rules based on document semantics  
- **Extensibility**: New document types via configuration
- **Maintainability**: Clear separation of concerns

**Configuration Flexibility:**
- Multiple processing methods per layer
- Role-specific field mappings
- Custom processors for specialized content
- Cross-section dependencies and validation

This enables teams to handle diverse document types and evolving business requirements through configuration changes rather than code modifications.