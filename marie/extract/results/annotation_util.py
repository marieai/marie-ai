from typing import List, Optional

from marie.extract.structures.annotation import Annotation
from marie.extract.structures.concrete_annotations import TypedAnnotation
from marie.extract.structures.line_with_meta import LineWithMeta


def filter_lines_by_annotation(
    lines: List[LineWithMeta], annotation_name: str
) -> List[LineWithMeta]:
    """
    Filter lines containing annotations that match the specified name.

    This function examines each line's annotations and includes the line in the
    result if any of its annotations match the specified name. The filtering
    process terminates early for each line once a matching annotation is found,
    optimizing performance for large document sets.

    :param lines: Collection of LineWithMeta objects to be filtered
    :param annotation_name: Target annotation name to match against
    :return: List of LineWithMeta objects containing at least one annotation
             with the specified name. Returns an empty list if no input lines
             are provided
    :raises ValueError: If an annotation is encountered that is not a TypedAnnotation
                        instance, indicating an unsupported annotation type

    .. note::
       The function uses early termination optimization - once a matching annotation
       is found in a line, processing moves to the next line immediately.

    .. warning::
       All annotations must be TypedAnnotation instances. Other annotation types
       will raise a ValueError.

    **Example Usage:**

    .. code-block:: python

        # Filter lines containing date annotations
        date_lines = filter_lines_by_annotation(document_lines, "DATE")

        # Filter lines containing claim number annotations
        claim_lines = filter_lines_by_annotation(all_lines, "CLAIM_NUMBER")

        # Process service line annotations
        service_lines = filter_lines_by_annotation(lines, "SERVICE_LINE")


    .. seealso::
       :func:`get_annotation_names` for discovering available annotation types
       :func:`count_annotations_by_type` for statistical analysis of annotations
    """
    if not lines:
        return []

    filtered_lines: List[LineWithMeta] = []

    for line in lines:
        annotations: List[Annotation] = line.annotations
        if not annotations:
            continue

        for annotation in annotations:
            if not isinstance(annotation, TypedAnnotation):
                raise ValueError(
                    f"Unsupported annotation type '{type(annotation).__name__}' "
                    f"encountered in line processing. Expected TypedAnnotation instance. "
                    f"This may indicate a configuration or data structure issue."
                )

            if annotation.name == annotation_name:
                filtered_lines.append(line)
                break  # Early termination once match is found

    return filtered_lines


def get_annotation_names(lines: List[LineWithMeta]) -> List[str]:
    """
    Extract all unique annotation names from the provided lines.

    This function analyzes all annotations across the provided lines and returns
    a comprehensive, sorted list of unique annotation names. This is useful for
    discovering what types of annotations are available in a document set and
    for validation purposes.

    :param lines: Collection of LineWithMeta objects to analyze
    :return: Alphabetically sorted list of unique annotation names found across all lines.
             Returns an empty list if no lines or annotations are provided
    :raises ValueError: If an annotation is not a TypedAnnotation instance

    .. note::
       The returned list is automatically sorted alphabetically for consistent output.

    **Example Usage:**

    .. code-block:: python

        # Discover all annotation types in a document
        annotation_types = get_annotation_names(document_lines)
        print(f"Available annotations: {annotation_types}")
        # Output: ['CLAIM_NUMBER', 'DATE', 'PATIENT_NAME', 'SERVICE_CODE']

        # Validate expected annotations are present
        required_annotations = {'CLAIM_NUMBER', 'DATE'}
        available_annotations = set(get_annotation_names(lines))
        missing = required_annotations - available_annotations
        if missing:
            print(f"Missing required annotations: {missing}")

    **Use Cases:**

    - **Configuration Validation**: Ensure required annotation types are present
    - **Data Discovery**: Understand what annotations are available in new datasets
    - **Testing**: Validate annotation extraction pipeline outputs
    - **Documentation**: Generate lists of supported annotation types

    .. seealso::
       :func:`filter_lines_by_annotation` for filtering by specific annotation names
       :func:`count_annotations_by_type` for quantitative analysis
    """
    if not lines:
        return []

    annotation_names = set()

    for line in lines:
        annotations: List[Annotation] = line.annotations
        if not annotations:
            continue

        for annotation in annotations:
            if not isinstance(annotation, TypedAnnotation):
                raise ValueError(
                    f"Unsupported annotation type '{type(annotation).__name__}' "
                    f"encountered during name extraction. Expected TypedAnnotation instance. "
                    f"This indicates an inconsistency in the annotation data structure."
                )
            annotation_names.add(annotation.name)

    return sorted(annotation_names)


def count_annotations_by_type(lines: List[LineWithMeta]) -> dict[str, int]:
    """
    Count occurrences of each annotation type across all lines.

    This function provides statistical analysis of annotation distribution,
    counting how many times each annotation type appears across the entire
    line collection. This is valuable for understanding document structure,
    validating extraction results, and identifying potential processing issues.

    :param lines: Collection of LineWithMeta objects to analyze
    :return: Dictionary mapping annotation names to their occurrence counts.
             Keys are annotation names (strings) and values are counts (integers).
             Returns an empty dictionary if no lines or annotations are provided
    :raises ValueError: If an annotation is not a TypedAnnotation instance

    .. note::
       The count represents the total number of annotation instances, not the number
       of lines containing each annotation type. A single line may contribute multiple
       counts if it contains multiple instances of the same annotation type.

    **Example Usage:**

    .. code-block:: python

        # Analyze annotation distribution in a document
        counts = count_annotations_by_type(document_lines)
        print(f"Annotation statistics: {counts}")
        # Output: {'CLAIM_NUMBER': 5, 'DATE': 12, 'PATIENT_NAME': 3, 'AMOUNT': 8}

        # Identify most common annotations
        most_common = max(counts.items(), key=lambda x: x[1])
        print(f"Most frequent annotation: {most_common[0]} ({most_common[1]} occurrences)")

        # Validate minimum expected counts
        if counts.get('CLAIM_NUMBER', 0) < 1:
            raise ValueError("No claim numbers found in document")

    **Use Cases:**

    - **Quality Assurance**: Verify expected annotation counts meet business rules
    - **Performance Monitoring**: Track annotation extraction effectiveness over time
    - **Data Profiling**: Understand the composition of document datasets
    - **Debugging**: Identify missing or over-extracted annotation types
    - **Reporting**: Generate statistical summaries for stakeholders

    **Business Applications:**

    - **Claims Processing**: Ensure required claim elements are properly extracted
    - **Invoice Analysis**: Validate that line items, amounts, and dates are captured
    - **Contract Review**: Confirm key terms and conditions are identified

    .. seealso::
       :func:`filter_lines_by_annotation` for extracting specific annotation types
       :func:`get_annotation_names` for discovering available annotation types
    """
    if not lines:
        return {}

    annotation_counts: dict[str, int] = {}

    for line in lines:
        annotations: List[Annotation] = line.annotations
        if not annotations:
            continue

        for annotation in annotations:
            if not isinstance(annotation, TypedAnnotation):
                raise ValueError(
                    f"Unsupported annotation type '{type(annotation).__name__}' "
                    f"encountered during count analysis. Expected TypedAnnotation instance. "
                    f"This suggests a data integrity issue in the annotation pipeline."
                )

            annotation_counts[annotation.name] = (
                annotation_counts.get(annotation.name, 0) + 1
            )

    return annotation_counts


def validate_annotation_completeness(
    lines: List[LineWithMeta],
    required_annotations: List[str],
    minimum_counts: Optional[dict[str, int]] = None,
) -> dict[str, str]:
    """
    Validate that required annotations are present with sufficient counts.

    This function performs comprehensive validation to ensure that all required
    annotation types are present in the document and meet minimum count requirements.
    It's designed for quality assurance in document processing pipelines.

    :param lines: Collection of LineWithMeta objects to validate
    :param required_annotations: List of annotation names that must be present
    :param minimum_counts: Optional dictionary specifying minimum required counts
                          for specific annotation types. If not provided, only
                          presence validation is performed
    :return: Dictionary containing validation results. Empty dict indicates success.
             Keys are error types, values are error descriptions
    :rtype: dict[str, str]

    **Example Usage:**

    .. code-block:: python

        # Basic presence validation
        required = ['CLAIM_NUMBER', 'PATIENT_NAME', 'DATE']
        errors = validate_annotation_completeness(lines, required)
        if errors:
            print(f"Validation failed: {errors}")

        # Validation with minimum counts
        minimums = {'CLAIM_NUMBER': 1, 'SERVICE_LINE': 3, 'DATE': 2}
        errors = validate_annotation_completeness(
            lines, ['CLAIM_NUMBER', 'SERVICE_LINE', 'DATE'], minimums
        )
    """
    errors = {}

    try:
        available_annotations = set(get_annotation_names(lines))
        annotation_counts = count_annotations_by_type(lines)

        # Check for missing required annotations
        missing_annotations = set(required_annotations) - available_annotations
        if missing_annotations:
            errors['missing_annotations'] = (
                f"Required annotations not found: {sorted(missing_annotations)}"
            )

        # Check minimum count requirements
        if minimum_counts:
            insufficient_counts = []
            for annotation_name, min_count in minimum_counts.items():
                actual_count = annotation_counts.get(annotation_name, 0)
                if actual_count < min_count:
                    insufficient_counts.append(
                        f"{annotation_name} (found: {actual_count}, required: {min_count})"
                    )

            if insufficient_counts:
                errors['insufficient_counts'] = (
                    f"Annotations below minimum counts: {insufficient_counts}"
                )

    except Exception as e:
        errors['validation_error'] = f"Validation process failed: {str(e)}"

    return errors
