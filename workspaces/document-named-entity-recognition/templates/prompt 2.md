
You are analyzing a series of **Explanation of Benefits (EOB)** document images to extract and annotate **all service table headers**. The output must be structured, ensuring accuracy and alignment with the **Provided Context** and **Image Context**.

**Think before you annotate!**

String matching and contextual understanding are essential to accurately extract the required information. The **Provided Context** serves as a reference to validate and reinforce the extracted data, while the **Image Context** provides additional annotations to enhance the extraction process.

---

### **Annotation Requirements:**

1. **Extract All Service Table Headers**:
   - Identify and extract **every** table header found in service tables, including but not limited to:
     - `"Service Dates"`, `"PL"`, `"Service Code"`, `"Num. Services"`, `"Submitted Charges"`, `"Negotiated Amount"`, `"Payable Amount"`, `"Deductible"`, `"Co-Pay"`, `"Patient Responsibility"`, `"Insurance Payment"`, `"Remarks"`, etc.
   - **Do not** extract claim-level headers such as `"Claim ID"`, `"Member ID"`, `"Patient Account"`, or `"Patient Name"`.

2. **Include Row Numbers**:
   - Each extracted table header must include the **row number** where it was found.

3. **Maintain Document Order**:
   - Extract headers in the exact **order of appearance** in the document.
   - Preserve the **row sequence** to ensure correct structural representation.

4. **Handle Missing Headers**:
   - If any expected headers are absent, annotate them as `"inferred"`, ensuring all headers are accounted for.

---

### **Output Format**:

The structured output must be formatted as TSV for efficient downstream ETL processing. Example:

#### **TSV Output Example**:

```tsv
ROW    HEADER NAME               STATUS
25     Service Dates             present
25     PL                        present
25     Service Code              present
25     Num. Services             present
```

---

### **Steps for Annotation:**

#### 1. **Identify and Extract All Table Headers**:
   - Locate **every** header in the service table, ensuring full coverage.
   - If a table header is **missing**, infer it if context allows.

#### 2. **Ensure Order of Extraction**:
   - Preserve **document order** when extracting headers.

#### 3. **Handle Missing Headers**:
   - If any expected headers are absent, annotate them as `"inferred"` to ensure full annotation coverage.

---