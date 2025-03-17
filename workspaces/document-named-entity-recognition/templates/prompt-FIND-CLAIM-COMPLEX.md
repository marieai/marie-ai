
You are tasked with analyzing **Explanation of Benefits (EOB)** document images to annotate claim sections and produce structured TSV output. The extracted annotations must align with the provided **Provided Context** and extracted **Image Context**, ensuring clarity, accuracy, and proper field correlation.

**Think before you annotate!**

String matching and contextual understanding are essential to accurately extract the required information. The **Provided Context** serves as a reference to validate and reinforce the extracted data, while the **Image Context** provides additional annotations to enhance the extraction process.

---

### **Definitions:**
- **Provided Context**: OCR Text extracted as prior knowledge to reinforce the extraction process.
- **Image Context**: The current image to provide additional annotations.
- **Row Number (ROW)**: The row number where the field was found in the document, starting from 0.
- **Group Counter (GROUP)**: A numerical identifier assigned to each claim section to maintain sequential order.
- **Field Name (REAL_KEY)**: The original field name as it appears in the document.
- **Normalized Field Name (NORMALIZED_KEY)**: A standardized version of the field name for consistency across different document variations.
- **Value**: The extracted content of the field.
- **Status**:
  - `"provided"`: Extracted from the **Provided Context**.
  - `"image"`: Extracted directly from the **Image Context**.
  - `"inferred"`: Deduced from context or nearby fields.
- **Reasoning**: A short explanation justifying the status of the extracted value.

---

### **Annotation Requirements:**

1. **Strict Prioritization of Provided Context**:
   - Always prioritize the **Provided Context** for validating and extracting data.
   - Use the **Provided Context** as the primary source to ground extracted values.
   - Fallback to **Image Context** only if the field is explicitly missing in the **Provided Context**.

2. **TSV Output Details**:
   For each claim header, output a structured TSV file with the following columns:

   | Column Name  | Description |
   |-------------|-------------|
   | **ROW** | The row number where the field was found in the document. |
   | **GROUP** | The claim section counter to track multiple claims in sequence. |
   | **REAL_KEY** | The actual field name as it appears in the document (e.g., `Claim ID`, `Member ID`, `Patient Account`, `Patient Name`). |
   | **NORMALIZED_KEY** | The standardized version of the field name (e.g., `CLAIM_ID`, `MEMBER_ID`, `PATIENT_ACCOUNT`, `PATIENT_NAME`). if no mapping is provid normalize as JSON key |
   | **VALUE** | The extracted value from the **Provided Context** (or **Image Context** as a fallback). |
   | **STATUS** | Indicates if the value was `"provided"`, `"image"`, or `"inferred"`. |
   | **REASONING** | Explanation for why the field was extracted from the chosen context. |


    Do not include any additional columns in the TSV output beyond the specified fields.

3. **Order of Appearance**:
   - Annotate claims in the exact sequence they appear in the document.
   - Use the row number of the **Claim ID** or the first field in a claim to maintain order.

4. **Handling Missing Fields**:
   - If a claim header is missing, annotate as follows:
     - **REAL_KEY**: The name of the missing field.
     - **NORMALIZED_KEY**: The standardized version of the field name.
     - **ROW**: `null`.
     - **STATUS**: `"inferred"`.
     - **REASONING**: Explanation for why the field is missing.

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

The structured output must be in `TAB-SEPARATED VALUES (TSV)` format. Example:

```tsv
ROW          GROUP                  REAL_KEY          NORMALIZED_KEY       VALUE      STATUS      REASONING
[ROW NUMBER] [CLAIM GROUP COUNTER]  [FIELD_NAME]      [STANDARD_NAME]     [VALUE]     [STATUS]    [ANNOTATION REASON]
...
```

### ** Provided Context:**
