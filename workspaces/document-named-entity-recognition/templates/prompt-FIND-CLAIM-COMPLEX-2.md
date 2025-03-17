You are tasked with analyzing **Explanation of Benefits (EOB)** document images to annotate claim sections and produce structured output. The output must align with the provided **Document Context** and extracted **Image Context**, ensuring clarity and reliability.

---

### **Definitions:**
- **Document Context**: A table of text provided as prior knowledge to reinforce the extraction process.
- **Image Context**: Text extracted directly from the document image to provide additional annotations.

---

### **Annotation Requirements:**

1. **Grounding with Document Context**:
   - Use the provided **Document Context** to validate extracted data.
   - If no **Document Context** is available, set `"grounded": false` for all fields.
   - Set `"grounded": true` only if the extracted value matches the **Document Context**.

2. **Annotation Details**:
   For each claim header, provide the following information:
   - **Value**: The extracted value.
   - **Row Number**: The row number where the field was found.
   - **Status**: `"present"` if directly extracted, or `"inferred"` if deduced.
   - **Grounded**: `"true"` or `"false"` based on correlation with the **Document Context**.
   - **Reasoning**: Explanation for the annotation decision.
   - **Confidence**: A score indicating the confidence in the annotation.
   - **Source**:
     - `"text"` if extracted from the **Document Context**.
     - `"image"` if extracted from the **Image Context**.
   - **Image Text**: The extracted text from the image (required if `"source": "image"`).
   - **Provided Text**: The corresponding text from the **Document Context** (required if `"source": "text"`).

3. **Order of Appearance**:
   - Annotate claims in the exact sequence they appear in the document.
   - Use the row number of the **Claim ID** or the first field in a claim to maintain order.

4. **Handling Missing Fields**:
   - If a claim header is missing, annotate as follows:
     - `"value": "XXXXX"`
     - `"row": null`
     - `"status": "inferred"`
     - `"grounded": false`

5. **Document Variations**:
   - Ensure robust handling of structural differences in documents, including OCR errors and formatting variations.

---

### **Goal:**
Extract and annotate **Claim Headers**:
   - **Claim ID**: A unique alphanumeric identifier.
   - **Member ID**: A unique identifier for the member.
   - **Patient Account**: The patient account number.
   - **Patient Name**: The full name of the patient.

---

### **Output Format**:

The structured output must be in JSON format. Example:

```json
{
  "claims": [
    {
      "Claim Headers": {
        "Claim ID": {
          "value": "<Extracted Claim ID>",
          "row": <Row Number>,
          "status": "present",
          "source": "text",
        },
        "Member ID": {
          "value": "XXXXX",
          "row": null,
          "status": "inferred",
          "source": null,
        },
        "Patient Account": {
          "value": "<Extracted Patient Account>",
          "row": <Row Number>,
          "status": "inferred",
          "grounded": false,
          "source": "image",
        }
      }
    }
  ]
}
```

---

### **Steps for Annotation**:

1. **Extraction and Correlation**:
   - Use the **Document Context** first for validation.
   - Cross-reference extracted data from the **Image Context**.
   - If data cannot be matched to the **Document Context**, set `"grounded": false`.

2. **Annotation Confidence**:
   - Assign a confidence score for each annotation based on OCR quality, clarity of data, and grounding.

3. **Maintain Order**:
   - Ensure annotations follow the order of appearance in the document.

4. **Reasoning for Missing or Inferred Fields**:
   - Provide a clear explanation for fields marked as `"inferred"` or `"grounded": false`.


### ** Document Context:**
