from __future__ import annotations

import random
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import torch
except Exception:
    torch = None

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, set_seed
except Exception:
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    pipeline = None
    set_seed = None

from faker import Faker


@dataclass
class TransformerAugmentOptions:
    n_samples: int = 5
    model_name: str = "google/flan-t5-base"  # any text2text model works
    max_new_tokens: int = 128
    temperature: float = 0.9
    top_p: float = 0.92
    seed: Optional[int] = 42
    device: Optional[str] = None  # "cuda", "cpu", or None to auto-detect
    overwrite: bool = False
    locale: str = "en_US"
    delimiters: Optional[List[str]] = None
    preserve_uppercase_labels: bool = True  # keep ALL-CAPS labels as-is


class TransformerReferenceBlockAugmenter:
    """
    Augments 'multiline_reference_blocks' by prompting a transformer model (e.g., T5/FLAN-T5)
    to replace semantic tokens while preserving label phrases. Falls back to Faker if the model isn't available.
    """

    def __init__(self, options: Optional[TransformerAugmentOptions] = None):
        self.opts = options or TransformerAugmentOptions()
        self.fake = Faker(self.opts.locale)
        if self.opts.seed is not None and set_seed is not None:
            try:
                set_seed(self.opts.seed)
                random.seed(self.opts.seed)
                if torch is not None:
                    torch.manual_seed(self.opts.seed)
            except Exception:
                pass

        self._delimiters = (
            self.opts.delimiters
            if self.opts.delimiters
            else [":", "-", "|", "•", "—", "=", "→"]
        )

        self._nlp = None
        self._maybe_init_pipeline()

    # -------------- Public API --------------

    def augment(self, data: dict) -> dict:
        """
        Input dict must contain 'multiline_reference_blocks'.
        Returns a copy with populated lines under:
          - 'augmented_reference_blocks' (default), or
          - 'multiline_reference_blocks' (if overwrite=True).
        """
        src = data.get("multiline_reference_blocks", {})
        out = deepcopy(data)

        result: Dict[str, List[str]] = {}
        for key, patterns in src.items():
            augmented_lines: List[str] = []
            for p in patterns:
                augmented_lines.extend(
                    self._augment_line_multiple(p, self.opts.n_samples)
                )
            result[key] = augmented_lines

        if self.opts.overwrite:
            out["multiline_reference_blocks"] = result
        else:
            out["augmented_reference_blocks"] = result
        return out

    # -------------- Internals --------------

    def _maybe_init_pipeline(self):
        if pipeline is None or AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
            return  # transformers not installed; rely on Faker fallback

        device = (
            0
            if (
                self.opts.device in ("cuda", None)
                and torch
                and torch.cuda.is_available()
            )
            else -1
        )
        try:
            tok = AutoTokenizer.from_pretrained(self.opts.model_name)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(self.opts.model_name)
            self._nlp = pipeline(
                "text2text-generation",
                model=mdl,
                tokenizer=tok,
                device=device,
            )
        except Exception:
            self._nlp = None  # fall back gracefully

    def _augment_line_multiple(self, pattern_line: str, n: int) -> List[str]:
        if self._nlp is None:
            return [self._faker_render(pattern_line) for _ in range(n)]

        prompts = [self._build_prompt(pattern_line) for _ in range(n)]

        print(prompts)
        outputs = self._nlp(
            prompts,
            max_new_tokens=self.opts.max_new_tokens,
            do_sample=True,
            temperature=self.opts.temperature,
            top_p=self.opts.top_p,
            num_return_sequences=1,
        )
        texts = [
            self._postprocess(
                o[0]["generated_text"] if isinstance(o, list) else o["generated_text"]
            )
            for o in outputs
        ]
        return [self._normalize_spacing(t) for t in texts]

    def _build_prompt(self, pattern_line: str) -> str:
        delimiters = ", ".join(self._delimiters)
        examples = [
            ("CLAIM NUMBER DELIMITER ALPHANUMERIC", "CLAIM NUMBER : 2Z5L8Q7"),
            (
                "NAME DELIMITER NAME PATIENT ACCOUNT DELIMITER ALPHANUMERIC",
                "NAME : John K. Smith PATIENT ACCOUNT - PA-482-Z7K9",
            ),
            (
                "ISSUE DATE DELIMITER DATE PROVIDER TAX ID DELIMITER ALPHANUMERIC",
                "ISSUE DATE : 03/14/2024 PROVIDER TAX ID - 73-9182746",
            ),
        ]
        ex_str = "\n".join([f"Input: {inp}\nOutput: {out}" for inp, out in examples])
        preserve_note = (
            "Preserve any ALL-CAPS label words exactly as they appear."
            if self.opts.preserve_uppercase_labels
            else ""
        )

        return """
            Replace placeholders in the input with realistic values.  
            Keep ALL-CAPS label words exactly as written.
            
            Placeholders and replacement rules:
            - DELIMITER: one of [:, -, |, •, —, =, →] with natural spacing
            - ALPHANUMERIC: random mix of letters and digits
            - NAME or PATIENT NAME: realistic person name
            - PROVIDER NAME: realistic company/provider name
            - PROVIDER TAX ID: US EIN format NN-NNNNNNN
            - HCID: plausible health-card-like ID
            - EMPLOYEE: short employee code
            - CHECK NUM: numeric check number
            - PATIENT ACCOUNT: like PA-123-ABCD
            - DATE: common date format (MM/DD/YYYY or YYYY-MM-DD)
            
            Generate the output on a single line.
            Input: CLAIM NUMBER DELIMITER ALPHANUMERIC PROVIDER TAX ID DELIMITER ALPHANUMERIC HCID DELIMITER ALPHANUMERIC EMPLOYEE DELIMITER NAME PROVIDER NAME DELIMITER
            Output:
            """

    def _build_prompXXt(self, pattern_line: str) -> str:
        delimiters = ", ".join(self._delimiters)
        examples = [
            ("CLAIM NUMBER DELIMITER ALPHANUMERIC", "CLAIM NUMBER : 2Z5L8Q7"),
            (
                "NAME DELIMITER NAME PATIENT ACCOUNT DELIMITER ALPHANUMERIC",
                "NAME : John K. Smith PATIENT ACCOUNT - PA-482-Z7K9",
            ),
            (
                "ISSUE DATE DELIMITER DATE PROVIDER TAX ID DELIMITER ALPHANUMERIC",
                "ISSUE DATE : 03/14/2024 PROVIDER TAX ID - 73-9182746",
            ),
        ]
        ex_str = "\n".join([f"Input: {inp}\nOutput: {out}" for inp, out in examples])
        preserve_note = (
            "Preserve any ALL-CAPS label words exactly as they appear."
            if self.opts.preserve_uppercase_labels
            else ""
        )

        return (
            "You are formatting document reference lines.\n"
            "Goal: Replace semantic placeholders with realistic values and keep label words unchanged.\n"
            f"{preserve_note}\n"
            "Semantic placeholders:\n"
            "- DELIMITER: choose one of the following and add natural spacing: "
            + delimiters
            + "\n"
            "- ALPHANUMERIC: random mix of letters and digits\n"
            "- NAME: realistic person name\n"
            "- PATIENT NAME: realistic person name\n"
            "- PROVIDER NAME: realistic company or provider name\n"
            "- PROVIDER TAX ID: US EIN-like format (NN-NNNNNNN)\n"
            "- HCID: plausible health-card-like ID\n"
            "- EMPLOYEE: short employee code\n"
            "- CHECK NUM: numeric check number\n"
            "- PATIENT ACCOUNT: mixed account number like PA-123-ABCD\n"
            "- DATE: a common date format (e.g., MM/DD/YYYY, YYYY-MM-DD)\n\n"
            "Output must be a single line with labels kept as-is and placeholders replaced.\n\n"
            f"{ex_str}\n\n"
            f"Input: {pattern_line}\n"
            "Output:"
        )

    def _postprocess(self, text: str) -> str:
        line = text.splitlines()[0].strip()
        line = re.sub(r"\s*([:\-\|\=•—→])\s*", r" \1 ", line)
        return self._normalize_spacing(line)

    @staticmethod
    def _normalize_spacing(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    # ---------- Faker fallback helpers ----------

    def _faker_render(self, pattern_line: str) -> str:
        words = self._normalize_spacing(pattern_line).split()
        out: List[str] = []
        i = 0
        while i < len(words):
            token = words[i].upper()
            if token == "DELIMITER":
                out.append(random.choice(self._delimiters))
            elif token == "ALPHANUMERIC":
                out.append(self._alnum())
            elif token == "NAME":
                out.append(self.fake.name())
            elif token == "PATIENT":
                if i + 1 < len(words) and words[i + 1].upper() == "NAME":
                    out.append(self.fake.name())
                    i += 1
                else:
                    out.append(words[i])
            elif token == "PROVIDER":
                if i + 1 < len(words):
                    nxt = words[i + 1].upper()
                    if nxt == "NAME":
                        out.append(self.fake.company())
                        i += 1
                    elif (
                        nxt == "TAX"
                        and i + 2 < len(words)
                        and words[i + 2].upper() == "ID"
                    ):
                        out.append(self._tax_id())
                        i += 2
                    else:
                        out.append(words[i])
                else:
                    out.append(words[i])
            elif token == "HCID":
                out.append(self._hcid())
            elif token == "EMPLOYEE":
                out.append(self._emp_code())
            elif (
                token == "CHECK"
                and i + 1 < len(words)
                and words[i + 1].upper() == "NUM"
            ):
                out.append(self._digits(6, 10))
                i += 1
            elif (
                token == "PATIENT"
                and i + 1 < len(words)
                and words[i + 1].upper() == "ACCOUNT"
            ):
                out.append(self._patient_account())
                i += 1
            elif token == "DATE":
                out.append(self._date_val())
            else:
                out.append(words[i])
            i += 1

        line = " ".join(out)
        line = re.sub(r"\s*([:\-\|\=•—→])\s*", r" \1 ", line)
        return self._normalize_spacing(line)

    @staticmethod
    def _digits(min_len=6, max_len=10):
        import random as _r

        length = _r.randint(min_len, max_len)
        return "".join(_r.choice("0123456789") for _ in range(length))

    @staticmethod
    def _alnum(min_len=6, max_len=12):
        import random as _r

        length = _r.randint(min_len, max_len)
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        return "".join(_r.choice(alphabet) for _ in range(length))

    @classmethod
    def _tax_id(cls):
        return f"{cls._digits(2)}-{cls._digits(7)}"

    @classmethod
    def _hcid(cls):
        return f"{cls._alnum(3,4)}{cls._digits(2)}{cls._alnum(2,3)}{cls._digits(4)}"

    @classmethod
    def _emp_code(cls):
        return f"EMP{cls._digits(1,3)}{cls._alnum(3,4)}"

    @classmethod
    def _patient_account(cls):
        return f"PA-{cls._digits(3)}-{cls._alnum(4,6)}"

    def _date_val(self):
        dt = self.fake.date_between(start_date="-3y", end_date="+10d")
        formats = ["%m/%d/%Y", "%Y-%m-%d", "%b %d, %Y", "%m-%d-%Y"]
        return dt.strftime(random.choice(formats))


# ---------------- Example ----------------
if __name__ == "__main__":
    sample = {
        'multiline_reference_blocks': {
            'BestPattern': [
                'CLAIM NUMBER DELIMITER ALPHANUMERIC PROVIDER TAX ID DELIMITER ALPHANUMERIC HCID DELIMITER ALPHANUMERIC EMPLOYEE DELIMITER NAME PROVIDER NAME DELIMITER',
                'NAME CHECK NUM DELIMITER ALPHANUMERIC PATIENT NAME DELIMITER NAME PATIENT ACCOUNT DELIMITER ALPHANUMERIC ISSUE DATE DELIMITER DATE',
            ]
        },
        'layout_id': '103932',
    }

    opts = TransformerAugmentOptions(
        n_samples=3,
        model_name="google/flan-t5-xl",  # or any text2text model
        seed=123,
        overwrite=False,
        locale="en_US",
    )
    augmenter = TransformerReferenceBlockAugmenter(opts)
    out = augmenter.augment(sample)

    generated = out.get("augmented_reference_blocks", {})
    for k, lines in generated.items():
        print(f"[{k}]")
        for line in lines[:6]:
            print("  ", line)
