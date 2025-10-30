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
    if not section_text:
        return kvs

    # First: parse markdown list items and extract key/value
    try:
        html = markdown.markdown(section_text)
        soup = BeautifulSoup(html, "html.parser")
        items = soup.find_all("li")

        def _clean_key(s: str) -> str:
            s = (s or "").strip()
            # Drop a single trailing ':' or em dash '—'
            if s.endswith(":") or s.endswith("—"):
                s = s[:-1]
            return s.strip()

        def _clean_value(s: str) -> str:
            return (s or "").strip()

        for li in items:
            key_text = None
            value_text = None

            # Strategy A: bold/strong key if present
            strong = li.find(["strong", "b"])
            if strong:
                key_text = _clean_key(strong.get_text(" ", strip=True))

                # Collect text after the bold key within the same <li>
                segs = []
                saw_key = False
                for child in li.children:
                    if getattr(child, "name", None) in ("strong", "b"):
                        if not saw_key:
                            saw_key = True
                        else:
                            segs.append(child.get_text(" ", strip=True))
                        continue
                    if saw_key:
                        segs.append(
                            child.get_text(" ", strip=True)
                            if hasattr(child, "get_text")
                            else str(child).strip()
                        )

                value_text = _clean_value(" ".join(filter(None, segs)))
                # Trim only allowed separators at the start of the value: ':' or em dash '—'
                value_text = re.sub(r'^\s*[:—]\s*', '', value_text).strip()

            # Strategy B: split the full text by allowed separators only
            if not key_text or value_text is None or value_text == "":
                full_text = li.get_text(" ", strip=True)

                # Allowed separators: colon ':' or em dash '—' (U+2014)
                m = re.match(r"^\s*([^:—]+?)\s*[:—]\s*(.+?)\s*$", full_text)
                if m:
                    key_text = _clean_key(m.group(1))
                    value_text = _clean_value(m.group(2))
                else:
                    # Heuristic without explicit separator (e.g., "KEY $-35.00")
                    m2 = re.match(r"^\s*([A-Z0-9_ ]{2,})\s+(.+?)\s*$", full_text)
                    if m2:
                        key_text = _clean_key(m2.group(1))
                        value_text = _clean_value(m2.group(2))

            if key_text and value_text is not None and key_text.strip():
                kvs.append(
                    KeyValue(
                        key=key_text.strip(),
                        value=value_text.strip(),
                        value_type=ValueType.UNKNOWN,
                    )
                )

        if kvs:
            return kvs
    except Exception:
        # Fall through to regex fallback
        pass

    # Final fallback: scan raw text bullets and split using only ':' or em dash '—'
    for m in re.finditer(r"(?m)^[\-\*\+]\s+(.+)$", section_text):
        line = m.group(1).strip()
        # Optional bold markers around the key, with allowed separators only
        m2 = re.match(r"^(?:\*\*|__)?\s*([^:—]+?)\s*(?:\*\*|__)?\s*[:—]\s*(.+)$", line)
        if m2:
            kvs.append(
                KeyValue(
                    key=m2.group(1).strip(),
                    value=m2.group(2).strip(),
                    value_type=ValueType.UNKNOWN,
                )
            )
            continue
        # No explicit separator heuristic
        m3 = re.match(r"^(?:\*\*|__)?\s*([A-Z0-9_ ]{2,})\s*(?:\*\*|__)?\s+(.+)$", line)
        if m3:
            kvs.append(
                KeyValue(
                    key=m3.group(1).strip(),
                    value=m3.group(2).strip(),
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
        cols_text = []
        for col in cols:
            # Replace <br> and <br/> tags with a space before extracting text. REMARK CODE
            for br in col.find_all("br"):
                br.replace_with(" ")
            text = col.get_text()
            cols_text.append(text)
        # cols_text = [col.get_text(strip=True) for col in cols]
        table_data.append(cols_text)

    if not table_data:
        return [], []

    headers = table_data[0]
    rows = table_data[1:]

    return headers, rows


def normalize_row_values(headers, row):
    """This is a passthrough as the fields are formatted via transformers."""
    out = []
    for h, v in zip(headers, row):
        out.append(v)
    if len(out) < len(headers):
        out += [""] * (len(headers) - len(out))
    return out
