"""Document Understanding Design Patterns.

Pre-built code patterns for common document analysis tasks,
inspired by Landing.ai's vision-agent approach but tailored for
Visual Document Understanding (VDU).

These patterns help the planning agent select appropriate strategies
based on document characteristics and task requirements.
"""

from typing import Optional

# =============================================================================
# Document Task Categories
# =============================================================================

DOCUMENT_TOOL_CATEGORIES = """
You are given a task: "{task}" from the user. Based on the task description,
extract the most appropriate category from the following list:

Categories:
- "OCR" - Extracting text from an image or document
- "DocQA" - Answering questions about a document's content
- "table_extraction" - Extracting structured table data from documents
- "form_extraction" - Extracting key-value pairs from forms
- "invoice_extraction" - Extracting structured data from invoices
- "classification" - Classifying document type or category
- "ner" - Named entity recognition (extracting names, dates, amounts, etc.)
- "layout_analysis" - Analyzing document structure and regions
- "handwriting" - Extracting handwritten content
- "signature_detection" - Detecting and extracting signatures
- "document_comparison" - Comparing two document versions
- "quality_assessment" - Assessing document image quality
- "multi_page" - Processing multi-page documents
- "general" - General document understanding tasks

Return ONLY the category name, nothing else.
"""

# =============================================================================
# Design Patterns for Document Tasks
# =============================================================================

DOCUMENT_DESIGN_PATTERNS = """
You are helping a document understanding agent solve visual document tasks.
Based on the user's task description and document characteristics, suggest
the most appropriate design pattern.

<category>small_text
**Description**: The document contains small or fine print text that may be
difficult to read at normal resolution.

**When to use**:
- Footnotes, disclaimers, or legal fine print
- Dense tables with small fonts
- Low-resolution scans

**Pattern**:
```python
def extract_small_text(image, regions=None):
    '''Extract small text by upscaling regions of interest.'''
    import cv2
    import numpy as np

    # If regions specified, crop and upscale each region
    if regions:
        results = []
        for region in regions:
            x1, y1, x2, y2 = region
            crop = image[y1:y2, x1:x2]
            # Upscale 2x for better OCR
            upscaled = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            text = ocr(upscaled)
            results.append({"region": region, "text": text})
        return results
    else:
        # Upscale entire image
        upscaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        return ocr(upscaled)
```
</category>

<category>rotated_text
**Description**: The document is skewed, rotated, or not properly aligned.

**When to use**:
- Scanned documents at an angle
- Mobile phone captures
- Documents with mixed orientations

**Pattern**:
```python
def extract_rotated_text(image):
    '''Detect and correct document rotation before OCR.'''
    import cv2
    import numpy as np

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges and lines
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10
    )

    # Calculate dominant angle
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)

    if angles:
        median_angle = np.median(angles)
        # Rotate image to correct
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return ocr(rotated)

    return ocr(image)
```
</category>

<category>table_extraction
**Description**: Extract structured data from tables in documents.

**When to use**:
- Financial statements
- Reports with tabular data
- Invoices with line items
- Forms with grid layouts

**Pattern**:
```python
def extract_table(image, table_region=None):
    '''Extract table structure and cell contents.'''
    # Step 1: Detect table regions if not provided
    if table_region is None:
        tables = detect_tables(image)
        if not tables:
            return {"error": "No tables detected"}
        table_region = tables[0]  # Use first table

    # Step 2: Extract table structure (rows/columns)
    x1, y1, x2, y2 = table_region
    table_crop = image[y1:y2, x1:x2]
    structure = detect_table_structure(table_crop)

    # Step 3: Extract text from each cell
    cells = []
    for row_idx, row in enumerate(structure['rows']):
        row_cells = []
        for col_idx, cell_bbox in enumerate(row):
            cx1, cy1, cx2, cy2 = cell_bbox
            cell_crop = table_crop[cy1:cy2, cx1:cx2]
            cell_text = ocr(cell_crop)
            row_cells.append(
                {"row": row_idx, "col": col_idx, "text": cell_text, "bbox": cell_bbox}
            )
        cells.append(row_cells)

    return {
        "table_bbox": table_region,
        "rows": len(structure['rows']),
        "cols": len(structure['rows'][0]) if structure['rows'] else 0,
        "cells": cells,
    }
```
</category>

<category>form_extraction
**Description**: Extract key-value pairs from structured forms.

**When to use**:
- Application forms
- Registration documents
- Surveys and questionnaires
- Government forms

**Pattern**:
```python
def extract_form_fields(image):
    '''Extract key-value pairs from form documents.'''
    # Step 1: Run OCR to get all text with positions
    ocr_results = ocr(image)

    # Step 2: Detect form field regions (labels + values)
    fields = detect_form_fields(image)

    # Step 3: Match labels with values based on spatial relationships
    key_values = []
    for field in fields:
        label_bbox = field.get('label_bbox')
        value_bbox = field.get('value_bbox')

        # Find text in label region
        label_text = find_text_in_region(ocr_results, label_bbox)

        # Find text in value region
        value_text = find_text_in_region(ocr_results, value_bbox)

        key_values.append(
            {
                "key": label_text.strip().rstrip(':'),
                "value": value_text.strip(),
                "confidence": field.get('confidence', 0.0),
            }
        )

    return {"fields": key_values}
```
</category>

<category>invoice_extraction
**Description**: Extract structured data from invoices.

**When to use**:
- Accounts payable automation
- Invoice processing workflows
- Receipt digitization

**Pattern**:
```python
def extract_invoice(image):
    '''Extract structured invoice data.'''
    # Step 1: Classify to confirm it's an invoice
    doc_type = classify_document(image)
    if doc_type != 'invoice':
        return {"warning": f"Document appears to be {doc_type}, not invoice"}

    # Step 2: Extract header information
    header_fields = [
        "invoice_number",
        "invoice_date",
        "due_date",
        "vendor_name",
        "vendor_address",
        "customer_name",
        "customer_address",
    ]
    header = extract_named_fields(image, header_fields)

    # Step 3: Extract line items table
    line_items = extract_table(image, table_type='line_items')

    # Step 4: Extract totals
    total_fields = ["subtotal", "tax", "total", "amount_due"]
    totals = extract_named_fields(image, total_fields)

    return {
        "header": header,
        "line_items": line_items,
        "totals": totals,
        "confidence": calculate_confidence(header, line_items, totals),
    }
```
</category>

<category>multi_column
**Description**: Handle documents with multi-column layouts.

**When to use**:
- Newspapers and magazines
- Academic papers (2-column format)
- Reports with sidebar content

**Pattern**:
```python
def extract_multi_column(image):
    '''Extract text from multi-column documents in reading order.'''
    # Step 1: Detect layout regions
    regions = detect_layout(image)

    # Step 2: Classify regions (header, column, footer, sidebar)
    classified = []
    for region in regions:
        region_type = classify_region(image, region['bbox'])
        classified.append({**region, 'type': region_type})

    # Step 3: Sort regions by reading order
    # Headers first, then columns left-to-right top-to-bottom, then footers
    headers = [r for r in classified if r['type'] == 'header']
    columns = [r for r in classified if r['type'] == 'column']
    footers = [r for r in classified if r['type'] == 'footer']

    # Sort columns: top-to-bottom within each vertical band
    columns.sort(key=lambda r: (r['bbox'][0] // 100, r['bbox'][1]))

    ordered_regions = headers + columns + footers

    # Step 4: Extract text from each region
    full_text = []
    for region in ordered_regions:
        x1, y1, x2, y2 = region['bbox']
        crop = image[y1:y2, x1:x2]
        text = ocr(crop)
        full_text.append({"type": region['type'], "text": text, "bbox": region['bbox']})

    return {"regions": full_text}
```
</category>

<category>handwriting
**Description**: Extract handwritten content from documents.

**When to use**:
- Handwritten notes or annotations
- Filled forms with handwriting
- Signatures and initials
- Mixed printed/handwritten documents

**Pattern**:
```python
def extract_handwriting(image, detect_regions=True):
    '''Extract handwritten text from documents.'''
    if detect_regions:
        # Step 1: Detect handwritten vs printed regions
        regions = detect_handwriting_regions(image)

        results = []
        for region in regions:
            x1, y1, x2, y2 = region['bbox']
            crop = image[y1:y2, x1:x2]

            if region['type'] == 'handwriting':
                # Use handwriting-specific OCR
                text = ocr_handwriting(crop)
            else:
                # Use standard OCR for printed text
                text = ocr(crop)

            results.append(
                {
                    "type": region['type'],
                    "text": text,
                    "bbox": region['bbox'],
                    "confidence": region.get('confidence', 0.0),
                }
            )

        return {"regions": results}
    else:
        # Process entire image as handwriting
        return {"text": ocr_handwriting(image)}
```
</category>

<category>document_comparison
**Description**: Compare two versions of a document to find differences.

**When to use**:
- Contract revisions
- Document version control
- Change detection in legal documents

**Pattern**:
```python
def compare_documents(image1, image2):
    '''Compare two document versions and highlight differences.'''
    import difflib

    # Step 1: Extract text from both documents
    text1 = ocr(image1)
    text2 = ocr(image2)

    # Step 2: Compute text differences
    differ = difflib.unified_diff(
        text1.splitlines(keepends=True),
        text2.splitlines(keepends=True),
        fromfile='version1',
        tofile='version2',
    )
    text_diff = list(differ)

    # Step 3: Compute visual differences
    visual_diff = compute_visual_diff(image1, image2)

    # Step 4: Categorize changes
    changes = {
        "additions": [
            l for l in text_diff if l.startswith('+') and not l.startswith('+++')
        ],
        "deletions": [
            l for l in text_diff if l.startswith('-') and not l.startswith('---')
        ],
        "visual_changes": visual_diff['changed_regions'],
    }

    return {
        "text_diff": ''.join(text_diff),
        "changes": changes,
        "similarity_score": difflib.SequenceMatcher(None, text1, text2).ratio(),
    }
```
</category>

<category>quality_assessment
**Description**: Assess and potentially improve document image quality.

**When to use**:
- Before processing low-quality scans
- When OCR confidence is low
- Preprocessing pipeline decisions

**Pattern**:
```python
def assess_quality(image):
    '''Assess document image quality and suggest improvements.'''
    import cv2
    import numpy as np

    metrics = {}
    suggestions = []

    # Check resolution
    h, w = image.shape[:2]
    dpi_estimate = min(w, h) / 8.5  # Assuming letter-size
    metrics['estimated_dpi'] = dpi_estimate
    if dpi_estimate < 150:
        suggestions.append("Low resolution - consider rescanning at 300 DPI")

    # Check brightness/contrast
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    metrics['mean_brightness'] = np.mean(gray)
    metrics['std_contrast'] = np.std(gray)

    if metrics['mean_brightness'] < 100:
        suggestions.append("Image too dark - apply brightness correction")
    elif metrics['mean_brightness'] > 200:
        suggestions.append("Image too bright - may be overexposed")

    if metrics['std_contrast'] < 30:
        suggestions.append("Low contrast - apply contrast enhancement")

    # Check for blur
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    metrics['sharpness'] = laplacian_var
    if laplacian_var < 100:
        suggestions.append("Image appears blurry - consider sharpening")

    # Check for skew
    skew_angle = detect_skew_angle(gray)
    metrics['skew_angle'] = skew_angle
    if abs(skew_angle) > 2:
        suggestions.append(
            f"Document skewed by {skew_angle:.1f} degrees - apply deskew"
        )

    # Overall quality score
    quality_score = 100
    quality_score -= max(0, (150 - dpi_estimate) / 3)
    quality_score -= max(0, (100 - metrics['sharpness']) / 10)
    quality_score -= abs(skew_angle) * 2
    metrics['quality_score'] = max(0, min(100, quality_score))

    return {
        "metrics": metrics,
        "suggestions": suggestions,
        "quality_score": metrics['quality_score'],
    }
```
</category>

<category>multi_page
**Description**: Process multi-page documents maintaining context.

**When to use**:
- PDF documents with multiple pages
- Multi-page contracts
- Reports and manuals

**Pattern**:
```python
def process_multi_page(images, task='extract_all'):
    '''Process multi-page documents with page-level and document-level analysis.'''
    page_results = []

    # Step 1: Process each page
    for page_num, image in enumerate(images):
        page_data = {
            "page": page_num + 1,
            "text": ocr(image),
            "layout": detect_layout(image),
            "tables": detect_tables(image),
        }
        page_results.append(page_data)

    # Step 2: Document-level analysis
    all_text = "\\n\\n".join([p['text'] for p in page_results])

    # Detect document structure (TOC, sections, etc.)
    structure = detect_document_structure(page_results)

    # Cross-page table continuation
    merged_tables = merge_continued_tables(page_results)

    return {
        "pages": page_results,
        "full_text": all_text,
        "structure": structure,
        "tables": merged_tables,
        "page_count": len(images),
    }
```
</category>

<category>nested_regions
**Description**: Handle documents with hierarchical or nested structures.

**When to use**:
- Documents with nested tables
- Complex forms with sections
- Organizational charts

**Pattern**:
```python
def extract_nested_regions(image, max_depth=3):
    '''Extract hierarchically nested regions from documents.'''

    def process_region(img, depth=0):
        if depth >= max_depth:
            return {"text": ocr(img), "children": []}

        # Detect sub-regions
        sub_regions = detect_regions(img)

        if not sub_regions or len(sub_regions) <= 1:
            return {"text": ocr(img), "children": []}

        children = []
        for region in sub_regions:
            x1, y1, x2, y2 = region['bbox']
            crop = img[y1:y2, x1:x2]
            child_result = process_region(crop, depth + 1)
            child_result['bbox'] = region['bbox']
            child_result['type'] = region.get('type', 'unknown')
            children.append(child_result)

        return {"text": None, "children": children}  # Text is in children

    result = process_region(image)
    return {"hierarchy": result}
```
</category>
"""

# =============================================================================
# Tool Suggestions by Category
# =============================================================================

TOOL_SUGGESTIONS = {
    "OCR": ["ocr", "ocr_handwriting"],
    "DocQA": ["ocr", "vqa", "extract_text_regions"],
    "table_extraction": ["detect_tables", "extract_table_structure", "ocr"],
    "form_extraction": ["detect_form_fields", "ocr", "extract_key_value"],
    "invoice_extraction": [
        "classify_document",
        "detect_tables",
        "ocr",
        "extract_named_fields",
    ],
    "classification": ["classify_document"],
    "ner": ["ocr", "extract_entities"],
    "layout_analysis": ["detect_layout", "detect_regions"],
    "handwriting": ["detect_handwriting_regions", "ocr_handwriting", "ocr"],
    "signature_detection": ["detect_signatures", "verify_signature"],
    "document_comparison": ["ocr", "compute_visual_diff"],
    "quality_assessment": ["assess_image_quality", "detect_skew"],
    "multi_page": ["ocr", "detect_layout", "detect_tables", "merge_pages"],
    "general": ["ocr", "vqa", "classify_document"],
}

# =============================================================================
# Helper Functions
# =============================================================================


def get_pattern_for_category(category: str) -> Optional[str]:
    """Get the design pattern for a specific category.

    Args:
        category: The document task category.

    Returns:
        The design pattern string or None if not found.
    """
    import re

    # Extract pattern for the given category
    pattern = rf'<category>{category}\n(.*?)</category>'
    match = re.search(pattern, DOCUMENT_DESIGN_PATTERNS, re.DOTALL)

    if match:
        return match.group(1).strip()
    return None


def categorize_document_task(task: str) -> str:
    """Categorize a document task based on its description.

    This is a simple heuristic-based categorization. For production use,
    this should be replaced with an LLM call using DOCUMENT_TOOL_CATEGORIES.

    Args:
        task: The task description from the user.

    Returns:
        The category string.
    """
    task_lower = task.lower()

    # Keyword-based categorization
    if any(kw in task_lower for kw in ['table', 'grid', 'row', 'column', 'cell']):
        return 'table_extraction'
    elif any(kw in task_lower for kw in ['form', 'field', 'fill', 'checkbox']):
        return 'form_extraction'
    elif any(kw in task_lower for kw in ['invoice', 'receipt', 'bill', 'payment']):
        return 'invoice_extraction'
    elif any(
        kw in task_lower for kw in ['classify', 'type', 'category', 'kind of document']
    ):
        return 'classification'
    elif any(kw in task_lower for kw in ['entity', 'name', 'date', 'amount', 'ner']):
        return 'ner'
    elif any(kw in task_lower for kw in ['layout', 'structure', 'region', 'section']):
        return 'layout_analysis'
    elif any(
        kw in task_lower for kw in ['handwrit', 'handwritten', 'cursive', 'script']
    ):
        return 'handwriting'
    elif any(kw in task_lower for kw in ['signature', 'sign', 'autograph']):
        return 'signature_detection'
    elif any(kw in task_lower for kw in ['compare', 'diff', 'difference', 'version']):
        return 'document_comparison'
    elif any(kw in task_lower for kw in ['quality', 'blur', 'skew', 'resolution']):
        return 'quality_assessment'
    elif any(kw in task_lower for kw in ['multi-page', 'pages', 'pdf', 'document']):
        return 'multi_page'
    elif any(
        kw in task_lower for kw in ['question', 'answer', 'what', 'where', 'who', 'how']
    ):
        return 'DocQA'
    elif any(kw in task_lower for kw in ['text', 'ocr', 'extract', 'read']):
        return 'OCR'
    else:
        return 'general'


def get_suggested_tools(category: str) -> list:
    """Get suggested tools for a category.

    Args:
        category: The document task category.

    Returns:
        List of suggested tool names.
    """
    return TOOL_SUGGESTIONS.get(category, TOOL_SUGGESTIONS['general'])


# =============================================================================
# Pattern-Based Prompt Construction
# =============================================================================


def build_pattern_prompt(task: str, include_examples: bool = True) -> str:
    """Build a prompt with relevant design patterns for the task.

    Args:
        task: The user's task description.
        include_examples: Whether to include code examples.

    Returns:
        A prompt string with relevant patterns.
    """
    category = categorize_document_task(task)
    pattern = get_pattern_for_category(category)
    tools = get_suggested_tools(category)

    prompt = f"""Task Category: {category}
Suggested Tools: {', '.join(tools)}

"""

    if pattern and include_examples:
        prompt += f"""Recommended Design Pattern:
{pattern}

"""

    prompt += f"""Your task: {task}

Based on the category and pattern above, create a step-by-step plan to accomplish this task.
Use the suggested tools and follow the pattern structure where applicable.
"""

    return prompt
