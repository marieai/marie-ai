import importlib
import re
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional, TypeAlias, Union

from marie.extract.structures import UnstructuredDocument
from marie.logging_core.predefined import default_logger as logger

TransformMapping: TypeAlias = Dict[str, Union[str, float, None]]
TransformReturnType: TypeAlias = Union[TransformMapping, List[TransformMapping]]


def convert_name_format(value: str, field_def: Dict[str, Any]) -> dict[str, None] | str:
    if not value:
        return {"first_name": None, "middle_name": None, "last_name": None}

    string_format = field_def.get(
        'name_format', '{title} {first} {middle} {last} {suffix} ({nickname})'
    )
    full_name = value

    from nameparser import HumanName

    name = HumanName(full_name, string_format=string_format)
    parsed_name = {"first": name.first, "middle": name.middle, "last": name.last}

    return parsed_name


def as_safe_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0
    except TypeError:
        return 0.0


def convert_to_decimal_money(field_value: str) -> float:
    field_value_orig = field_value
    if not field_value or field_value.strip() == "":
        return 0.0  # We should return None here, but for now we are returning 0.0

    # Detect if the value is negative (parentheses or '-')
    is_negative = '(' in field_value and ')' in field_value or '-' in field_value
    if is_negative:
        field_value = field_value.replace('(', '').replace(')', '').replace('-', '')

    # Replace spaces between digits with a decimal point
    field_value = re.sub(r"(\d)\s+(\d)", r"\1.\2", field_value)

    # Clean up: Remove currency symbols, commas, and other invalid characters
    cleaned = field_value.strip().replace("$", "").replace(",", "").replace(" ", "")
    cleaned = re.sub(r"[^\d.]", "", cleaned)  # Remove non-numeric, non-dot characters

    # Ensure there is at most one decimal point
    if cleaned.count('.') > 1:
        cleaned = cleaned.replace('.', '', cleaned.count('.') - 1)

    try:
        quantized = Decimal(cleaned).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)
        result = float(quantized)
    except (ValueError, ArithmeticError):
        # Default return 0.0 in case the input is invalid
        return 0.0

    if is_negative:
        result = -result

    logger.debug(
        f"Converted Value: {result:10.2f}  |  Original : {field_value_orig.strip()}"
    )
    return result


def convert_money_format(
    value: str, field_def: Dict[str, Any], default: float = None
) -> float | None:
    """
    Convert money string to float. If the value is invalid, return None.
    Handles various formats including: $1,234.56, (1,234.56), -1234.56, 1 234,56, etc.
    0.001 -> 0.00
    1234 -> 1234.00
    1234.5 -> 1234.50
    1234.567 -> 1234.57 (rounded)
    :param value: Money string value to convert
    :param field_def: Field definition dictionary
    :param default: Default value to return if conversion fails
    :return:
    """
    if not value:
        if default is not None:
            return default
        return None

    # Remove dollar sign and commas
    # value = value.replace('$', '').replace(',', '')
    try:
        return convert_to_decimal_money(value)
    except ValueError:
        logger.warning(f"Error converting money format for value: {value}")
        return None


def transform_field_value(
    field_def: Dict[str, Any], value: str, document: UnstructuredDocument | None = None
) -> TransformReturnType:
    """Transform field value based on field type."""
    if not value:
        return value

    field_name = field_def.get('name', None)
    field_type = field_def.get('type', 'ALPHA_NUMERIC')

    if field_name is None:
        raise ValueError(
            f"Field name is required in field definition. Field definition: {field_def}"
        )

    field_name = field_def.get('name', None)
    if field_name is None:
        raise ValueError(
            f"Field name is required in field definition. Field definition: {field_def}"
        )

    # Check for a custom transformer function
    if "transform" in field_def:
        custom_transformer_path = field_def["transform"]
        try:
            module_name, func_name = custom_transformer_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            transformer_func = getattr(module, func_name)
            return transformer_func(value, field_def, document=document)
        except (ImportError, AttributeError, ValueError) as e:
            logger.error(
                f"Could not resolve or use transform function '{custom_transformer_path}': {e}"
            )
            return value
        except Exception as e:
            logger.error(
                f"Error during custom transformation of field '{field_name}' with '{custom_transformer_path}': {e}"
            )
            return value

    transformers = {
        'MONEY': lambda v: convert_money_format(v, field_def) if v else None,
        'DATE': lambda v: convert_date_format(v, field_def) if v else None,
        'NAME': lambda v: convert_name_format(v, field_def) if v else None,
        'ALPHA_NUMERIC': lambda v: convert_to_alphanumeric(v) if v else None,
        'NUMERIC': lambda v: int(v) if v and v.isdigit() else float(v) if v else None,
        'ALPHA': lambda v: v.strip() if v else None,
    }

    transformer = transformers.get(field_type)
    if transformer:
        try:
            return transformer(value)
        except Exception as e:
            logger.error(f"Error transforming {field_name} ({field_type}): {e}")
            return value

    return value


def java_to_python_date_format(java_format: str) -> str:
    """
    Convert Java date format to Python strftime format with case-sensitive token mapping.
    """
    token_map = {
        'yyyy': '%Y',
        'yy': '%y',
        'MM': '%m',  # Month
        'dd': '%d',  # Day of month
        'HH': '%H',  # Hour (24)
        'hh': '%I',  # Hour (12)
        'mm': '%M',  # Minute
        'ss': '%S',  # Second
        'a': '%p',  # AM/PM
        'EEE': '%a',  # Weekday short
        'EEEE': '%A',  # Weekday full
        'MMM': '%b',  # Month short
        'MMMM': '%B',  # Month full
        'D': '%j',  # Day of year
        'z': '%Z',  # Time zone
    }

    i = 0
    result = []

    max_token_len = max(len(t) for t in token_map.keys())

    while i < len(java_format):
        matched = False
        for token_len in range(max_token_len, 0, -1):
            if i + token_len <= len(java_format):
                token = java_format[i : i + token_len]
                if token in token_map:
                    result.append(token_map[token])
                    i += token_len
                    matched = True
                    break
        if not matched:
            result.append(java_format[i])
            i += 1

    converted = ''.join(result)
    return converted


DASH_VARIANTS = r"[–—‒−-]"  # en dash, em dash, figure dash, minus sign, hyphen


def _normalize_date_range_input(raw: str) -> str:
    # Collapse all whitespace to single spaces
    s = re.sub(r"\s+", " ", raw or "").strip()
    # Normalize any dash variants to a hyphen, preserving optional spaces around it
    s = re.sub(rf"\s*{DASH_VARIANTS}\s*", " - ", s)
    # Final collapse just in case we produced double spaces
    return re.sub(r"\s{2,}", " ", s).strip()


def convert_date_format(
    value: str, field_def: Dict[str, Any]
) -> Union[str, Dict[str, Union[str, None]], None]:
    """Convert date string from input format to output format (YYYY-MM-DD).
    Supports date ranges via validation_regex and derived fields.
    """
    output_format = '%m/%d/%Y'  # MM/dd/yyyy ' # Example: 04/15/2025, this is the default output format in US
    derived_fields = field_def.get("derived_fields", None)
    validation_regex = field_def.get("validation_regex", None)
    # our formats are coming from java, so we need to convert them to python
    field_format = field_def.get("format", 'MM/dd/yyyy')

    if not value or value == '':
        return {key: None for key in derived_fields.keys()} if derived_fields else None

    value = _normalize_date_range_input(value)
    # Supported input formats in Java format
    # common_formats = [
    #     '%m/%d/%Y',  # MM/DD/YYYY
    #     '%Y-%m-%d',  # YYYY-MM-DD
    #     '%d/%m/%Y',  # DD/MM/YYYY
    #     '%m-%d-%Y',  # MM-DD-YYYY
    #     '%m%d%Y'  # MMDDYYYY
    # ]

    common_formats = [
        'MM/DD/YYYY',
        'YYYY-MM-DD',
        'DD/MM/YYYY',
        'MM-DD-YYYY',
        'MM/DD/YY'
    ]
    if field_format:
        common_formats = [field_format] + common_formats

    def try_parse_date(
        date_str: str, field_format: Optional[str] = None
    ) -> Union[str, None]:
        # Try parsing the date string with the provided format
        if field_format:
            try:
                return datetime.strptime(date_str.strip(), field_format).strftime(
                    output_format
                )
            except ValueError as e:
                print(e)
                pass

        for fmt in common_formats:
            try:
                return datetime.strptime(date_str.strip(), fmt).strftime(output_format)
            except ValueError:
                continue
        # Generalize this
        token_map = {
            "DD": "%d",
            "MM": "%m",
            "YYYY": "%Y",
            "YY": "%y",
        }
        py_output_format = output_format
        for k, v in token_map.items():
            py_output_format = py_output_format.replace(k, v)

        # Try parsing with 4-digit year first, then fallback to 2-digit
        for fmt in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime(py_output_format)
            except ValueError:
                continue
        return None

    # If regex is defined, use it to extract and convert components
    if validation_regex and derived_fields:
        # formats can be different for each split
        format_split = field_def.get('split', '-')
        split_values = value.split(format_split)
        split_formats = field_format.split(format_split)
        derived_items = derived_fields.items()
        match = re.match(validation_regex, value)

        if not match:
            if len(split_values) != len(split_formats):
                logger.warning(f"Split values and formats do not match for '{value}'")
                return {str(key): None for key, _ in derived_items}

            if len(split_values) != len(derived_items):
                logger.warning(
                    f"Split values and derived fields do not match for '{value}'"
                )
                return {str(key): None for key, _ in derived_items}

            logger.warning(f"Value '{value}' does not match regex '{validation_regex}'")
            output = {}
            for i, (split_value, split_format, derived_item) in enumerate(
                zip(split_values, split_formats, derived_items)
            ):
                extracted_date = split_value.strip()
                split_format = java_to_python_date_format(split_format.strip())
                formatted = try_parse_date(extracted_date, split_format)
                if not formatted:
                    logger.warning(
                        f"Failed to parse date '{extracted_date}' with format '{split_format}'"
                    )
                output[derived_item[0]] = formatted
            return output

        output = {}
        for i, (group_name, output_field) in enumerate(derived_items):
            extracted_date = match.group(group_name)
            split_format = java_to_python_date_format(split_formats[i].strip())
            formatted = try_parse_date(extracted_date, field_format=split_format)
            if not formatted:
                logger.warning(
                    f"Failed to parse date '{extracted_date}' with format '{field_format}'"
                )
            output[group_name] = formatted
        return output

    parsed_date = try_parse_date(value, field_format=field_format)
    return parsed_date if parsed_date else value


def convert_to_alphanumeric(
    field_value: str,
) -> str:
    if field_value is None:
        return ""
    if not isinstance(field_value, str):
        field_value = str(field_value)
    return re.sub(r'[^a-zA-Z0-9]', '', field_value)


def trim_name_prefix(field_value: str) -> str:
    if field_value is None:
        return None
    if not isinstance(field_value, str):
        field_value = str(field_value)

    honorifics = [
        "MR",
        "MRS",
        "MS",
        "DR",
        "MISS",
        "SIR",
        "PROF",
        "REV",
        "FR",
        "SR",
        "HON",
        "MR.",
        "MRS.",
        "MS.",
        "DR.",
        "MISS.",
        "SIR.",
        "PROF.",
        "REV.",
        "FR.",
        "SR.",
        "HON.",
    ]
    pattern = r'^(?:' + '|'.join(re.escape(h) for h in honorifics) + r')\s+'

    return re.sub(pattern, '', field_value, flags=re.IGNORECASE)


def trim_name_suffix(field_value: str) -> str:
    """
    Removes common relationship suffixes from names.

    Examples:
        >>> trim_name_suffix("PHILIP F SMITH (SELF)")
        'PHILIP F TOURANGEAU'
        >>> trim_name_suffix("DALTON R PHELPS (SON)")
        'DALTON R PHELPS'

    Args:
        field_value: The name string to process

    Returns:
        The name with relationship suffixes removed
    """
    if field_value is None:
        return None

    if not isinstance(field_value, str):
        field_value = str(field_value)

    # Pattern to match relationship suffixes in parentheses
    # This will match things like (SELF), (SON), (DAUGHTER), etc.
    relationship_pattern = r'\s*\([A-Z\s]+\)\s*$'

    return re.sub(relationship_pattern, '', field_value).strip()


if __name__ == "__main__":
    print(java_to_python_date_format("MMddyyyy"))  # %m%d%Y
    print(java_to_python_date_format("yyyyMMdd"))  # %Y%m%d
    print(java_to_python_date_format("MM/dd/yyyy"))  # %m/%d/%Y
    print(java_to_python_date_format("MM-dd-yy"))  # %m-%d-%y
    print(java_to_python_date_format("MMM dd, yyyy"))  # %b %d, %Y
    print(java_to_python_date_format("dd-MMM-yyyy"))  # %d-%b-%Y
    print(java_to_python_date_format("MM/dd/yyyy-MM/dd/yyyy"))  # %m/%d/%Y-%m/%d/%Y
    print(java_to_python_date_format("MMM dd yyyy-MMM dd yyyy"))  # %b %d %Y-%b %d %Y

    field_def = {
        "validation_regex": r'^(?P<begin>\d{2}/\d{2}/\d{4})\s*-\s*(?P<end>\d{2}/\d{2}/\d{4})$',
        "derived_fields": {
            "begin": "BEGIN_DATE_OF_SERVICE",
            "end": "END_DATE_OF_SERVICE",
        },
    }

    value = "02/14/2024 - 03/01/2024"
    result = convert_date_format(value, field_def)
    print(result)
    # Output: {'BEGIN_DATE_OF_SERVICE': '2024-02-14', 'END_DATE_OF_SERVICE': '2024-03-01'}

    # print(convert_to_decimal_money("0.001"))
