from dspy.operators import ExtractText, InferField, MatchPattern, TabulateResults
from dspy.pipeline import Pipeline

# Define the DSPy-compatible pipeline for service table headers extraction
pipeline = Pipeline(
    name="Service Table Header Extraction",
    description="Extracts and annotates all service table headers from Explanation of Benefits (EOB) documents.",
)

# Stage 1: Parse and tokenize textual data
pipeline.add_operator(
    ExtractText(
        name="ParseText",
        input_source=["Provided Context", "Image Context"],
        description="Extract text from Provided Context (if available) and fallback to Image Context.",
        output_fields=["raw_text", "row_data"],
    )
)

# Stage 2: Match headers with predefined patterns
pipeline.add_operator(
    MatchPattern(
        name="MatchHeaders",
        patterns=[
            "Service Dates",
            "PL",
            "Service Code",
            "Num. Services",
            "Submitted Charges",
            "Negotiated Amount",
            "Payable Amount",
            "Deductible",
            "Co-Pay",
            "Patient Responsibility",
            "Insurance Payment",
            "Remarks",
        ],
        input_field="raw_text",
        output_field="matched_headers",
        description="Match extracted headers against predefined patterns for service table headers.",
    )
)

# Stage 3: Infer missing headers based on context
pipeline.add_operator(
    InferField(
        name="InferMissingHeaders",
        context_field="row_data",
        input_field="matched_headers",
        output_field="all_headers",
        inference_rules={
            "infer_missing": True,
            "infer_reason": "Based on adjacent headers and structural cues.",
        },
        description="Infer missing headers by analyzing context and row structure.",
    )
)

# Stage 4: Tabulate results
pipeline.add_operator(
    TabulateResults(
        name="TabulateHeaders",
        input_field="all_headers",
        output_format="tsv",
        columns=["ROW", "HEADER NAME", "STATUS", "SOURCE"],
        description="Tabulate results into a structured TSV output for downstream processing.",
    )
)

# Finalize pipeline
pipeline.set_output_specification(
    output_format="tsv",
    output_fields=["ROW", "HEADER NAME", "STATUS", "SOURCE"],
    description="Outputs a TSV file with row numbers, header names, and statuses.",
)

# Print pipeline for review
print(pipeline)
