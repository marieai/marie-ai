import datetime
import difflib
import os
from functools import lru_cache
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import false

from marie.extract.annotators.base import AnnotatorCapabilities, DocumentAnnotator
from marie.extract.annotators.field_extractor import FieldValueExtractor
from marie.extract.annotators.multi_line_matcher import MultiLinePatternMatcher
from marie.extract.structures.line_with_meta import LineWithMeta
from marie.extract.structures.unstructured_document import UnstructuredDocument
from marie.utils.json import store_json_object


# TODO: Implement FLANT-T5 for OCR correction
class TinyOCRCorrector:
    def __init__(self):
        self.corrections = {'MISSPELED KEY': 'MISSPELLED KEY'}

    def correct(self, text: str) -> str:
        text_upper = text.upper()
        for wrong, correct in self.corrections.items():
            if wrong in text_upper:
                text_upper = text_upper.replace(wrong, correct)
        return text_upper


@lru_cache(maxsize=None)
def _get_transformer(
    model_name: str, for_mlm: bool = False, **kwargs
) -> SentenceTransformer:
    """
    Load (or return cached) SentenceTransformer.
    kwargs can include trust_remote_code, device, etc.
    """
    return SentenceTransformer(model_name, **kwargs)


class FaissHybridAnnotator(DocumentAnnotator):
    def __init__(
        self,
        working_dir: str,
        annotator_conf: dict[str, Any],
        layout_conf: dict[str, Any],
    ):
        super().__init__()
        self.working_dir = working_dir
        self.annotator_conf = annotator_conf
        self.layout_conf = layout_conf

        print(f"Initializing {self.__class__.__name__} with config: {annotator_conf}")

        self.name = annotator_conf.get('name', "faiss-hybrid-annotator")
        self.model_name = annotator_conf.get(
            'model_name', 'sentence-transformers/all-MiniLM-L6-v2'
        )  # JINA provides better embeddings
        self.top_k_candidates = annotator_conf.get('top_k_candidates', 3)
        self.fuzzy_threshold = annotator_conf.get('fuzzy_threshold', 0.8)
        self.embedding_threshold = annotator_conf.get('embedding_threshold', 0.85)
        self.fuzzy_weight = annotator_conf.get('fuzzy_weight', 0.3)
        self.embedding_weight = annotator_conf.get('embedding_weight', 0.7)
        self.min_final_score = annotator_conf.get('min_final_score', 0.7)
        self.min_acceptance_score = annotator_conf.get('min_acceptance_score', 0.7)
        self.critical_fields = annotator_conf.get('critical_fields', [])
        self.critical_field_boost = annotator_conf.get('critical_field_boost', 0.1)
        self.memory_enabled = annotator_conf.get('memory_enabled', True)
        self.memory_fields = annotator_conf.get('memory_fields', [])
        self.deduplicate_fields = annotator_conf.get('deduplicate_fields', False)
        self.deduplication_strategy = annotator_conf.get(
            'deduplication_strategy', "first"
        )
        self.target_labels = annotator_conf.get('target_labels', [])
        self.dynamic_ngram_thresholds = annotator_conf.get(
            'dynamic_ngram_thresholds', {"short": 3, "medium": 7, "long": 15}
        )

        # self.model = SentenceTransformer(self.model_name)
        self.model = _get_transformer(self.model_name, False, trust_remote_code=False)

        self.label_embeddings = self.model.encode(
            self.target_labels, normalize_embeddings=True
        )
        self.index = self.build_faiss_index(self.label_embeddings)
        self.embedding_cache = {}
        self.field_memory = {}
        self.ocr_corrector = TinyOCRCorrector()
        self.value_extractor = FieldValueExtractor(self.target_labels)

        # multilines are disabled by default
        self.multiline_enabled = annotator_conf.get("multiline_enabled", False)
        if self.multiline_enabled:
            ref_blocks = annotator_conf.get("multiline_reference_blocks", {})
            self.multiline_threshold = annotator_conf.get("multiline_threshold", 0.85)
            self.multiline_window = annotator_conf.get("multiline_window", 2)
            # resusing same model causes issues with max_seq_length
            # model_mml = SentenceTransformer(
            #     self.model_name,
            #     trust_remote_code=True
            # )
            # reuse same loader but pass trust_remote_code and override seq_length
            model_mml = _get_transformer(self.model_name, True, trust_remote_code=True)
            model_mml.max_seq_length = 1024  # input sequence length up to 8192

            self.multiline_matcher = MultiLinePatternMatcher(
                model=model_mml,
                threshold=self.multiline_threshold,
                reference_blocks=ref_blocks,
            )

        self.output_dir = os.path.join(working_dir, "agent-output", self.name)
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def capabilities(self) -> list:
        return [AnnotatorCapabilities.EXTRACTOR]

    def build_faiss_index(self, embeddings: np.ndarray):
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index

    def annotate(self, document: UnstructuredDocument, frames: list) -> list[Any]:
        self.validate_document(document)
        lines_by_page = document.lines_by_page
        all_extractions = []

        for page_id, lines_with_meta in lines_by_page.items():
            extraction_result = self._process_lines(lines_with_meta)
            extract = {
                "page_id": page_id,
                "page_number": page_id + 1,
                "extractions": extraction_result,
            }
            all_extractions.append(extract)
            # FIXME: should we store this in the output dir as individual files?

        output_file = os.path.join(self.output_dir, f"annotations.json")
        store_json_object(all_extractions, output_file)

        return all_extractions

    def parse_output(self, raw_output: str):
        raise NotImplementedError()

    def _process_lines(self, document_lines: List[Any]) -> List[Dict[str, Any]]:
        all_extractions = []
        missing_fields = set(self.target_labels)

        # determine allowed rows
        if self.multiline_enabled:
            texts = [lm.line for lm in document_lines]
            blocks = self.multiline_matcher.find_matching_blocks(
                texts, window=self.multiline_window
            )
            allowed = set()
            for b in blocks:
                allowed |= set(range(b["start_line"] - 1, b["end_line"]))
        else:
            allowed = set(range(len(document_lines)))

        for idx, line_with_meta in enumerate(document_lines):
            if idx not in allowed:
                continue
            raw_line = line_with_meta.line
            corrected = self.ocr_corrector.correct(raw_line.strip())
            tokens = corrected.split()
            i = 0
            while i < len(tokens):
                sizes = self.determine_ngram_sizes(" ".join(tokens[i:]))
                matched = False
                for sz in sizes:
                    if i + sz > len(tokens):
                        continue
                    ngram = " ".join(tokens[i : i + sz])
                    emb = self.encode_with_cache([ngram])[0]
                    label, score, strat, cands = self.hybrid_match(
                        ngram, emb, top_k=self.top_k_candidates
                    )
                    if label and score >= self.min_acceptance_score:
                        val = self.value_extractor.extract(corrected, label)
                        self.field_memory[label] = val.strip()
                        all_extractions.append(
                            self._build_field_result(
                                label, val, idx, score, strat, corrected, sz
                            )
                        )
                        missing_fields.discard(label)
                        i += sz
                        matched = True
                        break
                if not matched:
                    i += 1

        if self.deduplicate_fields:
            all_extractions = self.deduplicate_field_entries(
                all_extractions, strategy=self.deduplication_strategy
            )

        return all_extractions

    def hybrid_match(self, text: str, emb: np.ndarray, top_k: int = 3):
        """
        Try Fuzzy match first -> Embedding match -> Fallback to Memory if enabled.
        """
        # Step 1: Try Fuzzy First
        fuzzy_label, fuzzy_score = self.fuzzy_match(text)
        if fuzzy_label and fuzzy_score >= self.fuzzy_threshold:
            return (
                fuzzy_label,
                fuzzy_score,
                "fuzzy-direct",
                [{"label": fuzzy_label, "final_score": fuzzy_score}],
            )

        # Step 2: Then try Embedding
        faiss_scores, faiss_indices = self.index.search(
            np.expand_dims(emb, axis=0), top_k
        )
        candidates = []

        for i in range(top_k):
            idx = int(faiss_indices[0][i])
            emb_score = float(faiss_scores[0][i])
            label = self.target_labels[idx]
            boosted_score = self.boost_if_critical(label, emb_score)

            candidates.append(
                {
                    "label": label,
                    "source": "embedding",
                    "raw_score": boosted_score * self.embedding_weight,
                }
            )

        if candidates:
            raw_scores = np.array([c['raw_score'] for c in candidates])
            exp_scores = np.exp(raw_scores - np.max(raw_scores))
            softmax_scores = exp_scores / exp_scores.sum()

            for i, c in enumerate(candidates):
                c['final_score'] = softmax_scores[i]

            candidates = sorted(
                candidates, key=lambda x: x['final_score'], reverse=True
            )
            best = candidates[0]

            if best['final_score'] >= self.min_final_score:
                return (
                    best['label'],
                    best['final_score'],
                    "embedding-softmax",
                    candidates,
                )

        # Step 3: Finally fallback to Memory
        if self.memory_enabled and text in self.field_memory:
            # Pretend it's a 100% confident memory recall
            return text, 1.0, "memory-fallback", [{"label": text, "final_score": 1.0}]

        return None, 0.0, "none", []

    def boost_if_critical(self, label: str, score: float) -> float:
        return (
            min(score + self.critical_field_boost, 1.0)
            if label in self.critical_fields
            else score
        )

    def fuzzy_match(self, candidate: str) -> (str, float):
        matches = difflib.get_close_matches(
            candidate, self.target_labels, n=1, cutoff=self.fuzzy_threshold
        )
        if matches:
            return (
                matches[0],
                difflib.SequenceMatcher(None, candidate, matches[0]).ratio(),
            )
        return None, 0.0

    def encode_with_cache(self, texts: List[str]) -> np.ndarray:
        new_texts, cached = [], []
        for text in texts:
            if text in self.embedding_cache:
                cached.append(self.embedding_cache[text])
            else:
                new_texts.append(text)

        if new_texts:
            new_embs = self.model.encode(new_texts, normalize_embeddings=True)
            for t, e in zip(new_texts, new_embs):
                self.embedding_cache[t] = e
                cached.append(e)

        return np.vstack(cached)

    def generate_ngrams(self, text: str, n: int) -> List[str]:
        tokens = text.split()
        return [
            ' '.join(tokens[i : i + j])
            for i in range(len(tokens))
            for j in range(1, n + 1)
            if i + j <= len(tokens)
        ]

    def determine_ngram_sizes(self, text: str) -> List[int]:
        tokens = text.split()
        l = len(tokens)
        if l <= self.dynamic_ngram_thresholds['short']:
            return [2]
        elif l <= self.dynamic_ngram_thresholds['medium']:
            return [3, 2]
        elif l <= self.dynamic_ngram_thresholds['long']:
            return [4, 3, 2]
        else:
            return [5, 4, 3, 2]

    def _build_field_result(
        self, label, value, idx, score, strategy, source_line, ngram_size
    ):
        return {
            "line_number": idx + 1,
            "label": label,
            "value": value.strip(),
            "label_found_at": f"Found in row {idx + 1}",
            "reasoning": f"{strategy.capitalize()} match with confidence {round(score, 3)} (n-gram size {ngram_size})",
            "source_text": source_line.strip(),
        }

        if False:
            return {
                "field_name": label,
                "value": value.strip(),
                "confidence": round(score, 3),
                "line_number": idx + 1,
                "value_found_at": f"row {idx + 1}",
                "reasoning": f"{strategy.capitalize()} Top-{self.top_k_candidates} blended match (n-gram size {ngram_size})",
                "source_text": source_line.strip(),
                "match_strategy": strategy,
                "memory_applied": False,
            }

    def deduplicate_field_entries(
        self, fields: List[Dict[str, Any]], strategy: str = "first"
    ) -> List[Dict[str, Any]]:
        seen = {}
        for field in fields:
            label = field['field_name']
            if label not in seen:
                seen[label] = field
            else:
                if strategy == "first":
                    continue
                elif strategy == "highest_confidence":
                    if field['confidence'] > seen[label]['confidence']:
                        seen[label] = field
        return list(seen.values())

    def get_current_timestamp_utc(self):
        return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
