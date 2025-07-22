import re
from difflib import SequenceMatcher


def normalize_text(text: str) -> str:
    """
    Normalize text for matching: remove spaces, colons, hyphens, etc.
    """
    return re.sub(r"[\s:\-–—|]+", "", text).upper()


class FieldValueExtractor:
    def __init__(
        self,
        target_labels: list[str],
        fuzzy_threshold: float = 0.85,
        debug: bool = False,
    ):
        self.target_labels = target_labels
        self.fuzzy_threshold = fuzzy_threshold
        self.debug = debug  # <-- NEW
        self.normalized_label_lookup = self._prepare_label_lookup(target_labels)

    def _prepare_label_lookup(self, labels):
        return {self._normalize(label): label for label in labels}

    def _normalize(self, s: str) -> str:
        return re.sub(r"[\s:]+", "", s.upper())

    def extract(self, line: str, match_text: str) -> str:
        """
        Extract value after matched label, stopping at the next known label.
        Handles extra spacing, delimiters, and OCR typos.
        """
        label_tokens = re.split(r"\s+", match_text.strip())
        pattern = r"[\s:\-–—|]*".join(re.escape(token) for token in label_tokens)
        match = re.search(pattern, line, flags=re.IGNORECASE)
        if not match:
            if self.debug:
                print(f"[DEBUG] No match for label: {match_text} in line: {line}")
            return ""

        after = line[match.end() :].lstrip()

        delimiter_match = re.match(r"^[#:=\-–—|\t\s]+(.+)", after)
        if delimiter_match:
            after = delimiter_match.group(1).strip()

        if not after:
            if self.debug:
                print(f"[DEBUG] No content after label: {match_text}")
            return ""

        after_norm = normalize_text(after)
        cut_idx = len(after)

        for label in self.target_labels:
            label_norm = normalize_text(label)
            pos = after_norm.find(label_norm)
            if pos != -1:
                real_cut_pos = self.find_real_position(after, label_norm)
                if real_cut_pos != -1 and real_cut_pos < cut_idx:
                    cut_idx = real_cut_pos
            else:
                after_tokens = after.split()
                for i in range(len(after_tokens)):
                    window = " ".join(after_tokens[i : i + 2])
                    if not window:
                        continue
                    similarity = SequenceMatcher(
                        None, normalize_text(window), label_norm
                    ).ratio()
                    if similarity >= self.fuzzy_threshold:
                        real_cut_pos = after.find(after_tokens[i])
                        if real_cut_pos != -1 and real_cut_pos < cut_idx:
                            cut_idx = real_cut_pos

        value = after[:cut_idx].strip()
        value = re.sub(r"\s+", " ", value)  # Normalize internal spaces

        if self.debug:
            print(f"[DEBUG] Line: {line}")
            print(f"[DEBUG] Matched Label: {match_text}")
            print(f"[DEBUG] Extracted Value: '{value}'\n")

        return value

    def find_real_position(self, text: str, normalized_label: str) -> int:
        """
        Find real position of a normalized label inside original text.
        Used to map normalized string position back to raw index.
        """
        cleaned = ""
        mapping = {}
        j = 0
        for i, c in enumerate(text):
            if c not in ": \t\n\r-–—|":
                cleaned += c.upper()
                mapping[j] = i
                j += 1

        idx = cleaned.find(normalized_label)
        if idx == -1:
            return -1
        return mapping.get(idx, -1)
