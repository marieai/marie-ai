import re

import markdown
from bs4 import BeautifulSoup
from markdown.extensions.tables import TableExtension

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
    """
    Parses a Markdown table into headers and rows.

    This function takes a Markdown-encoded text containing a table, parses it, and extracts
    the headers and rows in a structured format. The Markdown text is converted to HTML first,
    and then the function identifies and extracts table data using BeautifulSoup.

    :param section_text: A string containing the Markdown-formatted text to parse.
    :returns: A tuple containing two lists - the first list is the headers of the table,
        and the second list contains the rows of the table.
    """
    html_output = markdown.markdown(section_text, extensions=[TableExtension()])
    soup = BeautifulSoup(html_output, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        return [], []

    table = tables[0]
    table_data = []
    rows = table.find_all("tr")

    for row in rows:
        cols = row.find_all(["th", "td"])
        cols_text = [col.get_text(strip=True) for col in cols]
        table_data.append(cols_text)

    if not table_data:
        return [], []

    headers = table_data[0]
    rows = table_data[1:]

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
