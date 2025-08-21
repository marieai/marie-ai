import re

from marie.extract.structures.cell_with_meta import CellWithMeta
from marie.extract.structures.line_with_meta import LineWithMeta
from marie.extract.structures.structured_region import KeyValue, ValueType


def _text_cell(text: str):
    """Create a text-only CellWithMeta for tests."""
    return CellWithMeta(lines=[LineWithMeta(line=text)])


def split_markdown_sections(md: str):
    parts = re.split(r"(?m)^##\s+", md.strip())
    parts = [p.strip() for p in parts if p.strip()]
    out = {}
    for p in parts:
        title, _, body = p.partition("\n")
        out[title.strip().upper()] = body.strip()
    return out


def parse_bullets_to_kvlist(section_text: str):
    kvs = []
    for m in re.finditer(r"(?m)^\-\s+([^:]+):\s*(.+)$", section_text):
        kvs.append(
            KeyValue(
                key=m.group(1).strip(),
                value=m.group(2).strip(),
                value_type=ValueType.UNKNOWN,
            )
        )
    return kvs


def parse_markdown_table(section_text: str):
    lines = [
        ln.strip() for ln in section_text.splitlines() if ln.strip().startswith("|")
    ]
    if len(lines) < 2:
        return [], []
    headers = [h.strip() for h in lines[0].strip("|").split("|")]
    sep = lines[1]
    if not re.search(r"\|\s*-+", sep):
        data_lines = lines[1:]
    else:
        data_lines = lines[2:]
    rows = []
    for dl in data_lines:
        rows.append([v.strip() for v in dl.strip("|").split("|")])
    return headers, rows


def normalize_row_values(headers, row):
    out = []
    money_like = {
        "BILLED_AMOUNT",
        "DISCOUNT",
        "ALLOWED_AMOUNT",
        "DEDUCTIBLE",
        "COINSURANCE",
        "COPAY",
        "BALANCE_PAYABLE",
    }
    for h, v in zip(headers, row):
        if h.upper() in money_like:
            v = v.strip().replace("$", "")
            v = f"${float(v):.2f}" if v else ""
        out.append(v)
    if len(out) < len(headers):
        out += [""] * (len(headers) - len(out))
    return out
