from typing import List

from pydantic import BaseModel, Field


class ReasoningMixin:
    reasoning: str | None = Field(
        default=None,
        description="Short explanation of the reasoning behind the extraction.",
        # description="Explain the step-by-step thought process behind the provided values. Include key considerations and how they influenced the final decisions."
    )


class Segment(BaseModel, ReasoningMixin):
    line_number: int = Field(..., description="Exact row number from the OCR input.")

    label: str = Field(
        ..., description="The target field label extracted from the OCR input."
    )
    value: str = Field(
        ...,
        description="The text extracted immediately after the first delimiter until the next target field label or end of the line.",
    )

    label_found_at: str = Field(
        ...,
        description='A string formatted as `"Found in row X"`, where X is the current line number that was matched to in the OCR Data.',
    )


class ExtractionResult(BaseModel):
    extractions: List[Segment] = Field(
        ..., description="Array of extracted segments with key-value pairs."
    )


class LineSegment(BaseModel, ReasoningMixin):
    line_number: int = Field(..., description="Exact row number from the OCR input.")
    value: str  # the rest of the line's content
    found_at: str = Field(..., description="Formatted as 'Found in row X'.")


class Table(BaseModel):
    name: str
    header_rows: List[LineSegment]
    rows: List[LineSegment]
    columns: List[str] = Field(..., description="List of column names.")


class TableExtractionResult(BaseModel):
    extractions: List[Table]


# Sample dataset instantiation with reasoning provided for each extraction
sample_dataset = TableExtractionResult(
    extractions=[
        Table(
            name="Header Set 1",
            columns=[
                "Column A",
                "Column B",
                "Column C",
                "Column D",
                "Column E",
                "Column F",
                "Column G",
            ],
            header_rows=[
                LineSegment(
                    line_number=1,
                    value="SERVICE DATES SERVICE CODES NUM SVC SUBMITTED CHARGES",
                    found_at="Found in row 1",
                    reasoning="Extracted primary header row containing service dates and service codes.",
                ),
                LineSegment(
                    line_number=2,
                    value="PL NUM SVC SVCS CHARGES AMOUNT AMOUNT PAYABLE REMARKS INSURANCE RESP AMOUNT",
                    found_at="Found in row 2",
                    reasoning="Extracted secondary header row with details on charges and remarks.",
                ),
            ],
            rows=[
                LineSegment(
                    line_number=3,
                    found_at="Found in row 3",
                    value="03/01/2025 12345 1 100.00 50.00 30.00 20.00",
                    reasoning="Service line extraction for the first service entry.",
                ),
                LineSegment(
                    line_number=4,
                    found_at="Found in row 4",
                    value="03/02/2025 67890 2 200.00 70.00 40.00 30.00",
                    reasoning="Service line extraction for the second service entry.",
                ),
                LineSegment(
                    line_number=5,
                    found_at="Found in row 5",
                    value="03/03/2025 54321 1 50.00 20.00 10.00 5.00",
                    reasoning="Service line extraction for the third service entry.",
                ),
            ],
        ),
        Table(
            name="Header Set 2",
            columns=[
                "Column A",
                "Column B",
                "Column C",
                "Column D",
                "Column E",
                "Column F",
                "Column G",
            ],
            header_rows=[
                LineSegment(
                    line_number=1,
                    found_at="Found in row 1",
                    value="COPAY AMOUNT SEE REMARKS DEDUCTIBLE COINSURANCE PATIENT RESPONSIBILITY",
                    reasoning="Extracted primary header row for copay and responsibility details.",
                ),
                LineSegment(
                    line_number=2,
                    found_at="Found in row 2",
                    value="SERVICE DATE SERVICE CODES SUBMITTED CHARGES",
                    reasoning="Extracted secondary header row with additional service information.",
                ),
            ],
            rows=[
                LineSegment(
                    line_number=6,
                    found_at="Found in row 6",
                    value="03/04/2025 11111 1 150.00 40.00 30.00 20.00",
                    reasoning="Service line extraction with copay and deductible details.",
                ),
                LineSegment(
                    line_number=7,
                    found_at="Found in row 7",
                    value="03/05/2025 22222 2 300.00 80.00 60.00 40.00",
                    reasoning="Service line extraction with updated charge and copay details.",
                ),
                LineSegment(
                    line_number=8,
                    found_at="Found in row 8",
                    value="03/06/2025 33333 1 75.00 25.00 15.00 10.00",
                    reasoning="Service line extraction with copay and coinsurance information.",
                ),
            ],
        ),
    ]
)

print(sample_dataset.model_dump())
print(sample_dataset.model_json_schema())


class GroupedEntry(BaseModel):
    value: str
    line_number: int
    label_found_at: str
    reasoning: str


class GroupedData(BaseModel):
    CLAIM_ID: List[GroupedEntry] = []
    MEMBER_ID: List[GroupedEntry] = []
    PATIENT_ACCOUNT: List[GroupedEntry] = []
    MEMBER: List[GroupedEntry] = []


print(ExtractionResult.model_json_schema())
