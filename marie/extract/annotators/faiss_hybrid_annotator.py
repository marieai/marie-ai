import datetime
import difflib
import os
import re
from functools import lru_cache
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from marie.extract.annotators.base import AnnotatorCapabilities, DocumentAnnotator
from marie.extract.annotators.field_extractor import FieldValueExtractor
from marie.extract.annotators.multi_line_matcher import MultiLinePatternMatcher
from marie.extract.structures.unstructured_document import UnstructuredDocument
from marie.logging_core.logger import MarieLogger
from marie.models.utils import torch_gc
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
        **kwargs: Any,
    ):
        super().__init__()
        self.logger = MarieLogger(context=self.__class__.__name__)

        self.working_dir = working_dir
        self.annotator_conf = annotator_conf
        self.layout_conf = layout_conf

        self.logger.info(
            f"Initializing {self.__class__.__name__} with config: {annotator_conf}"
        )

        self.name = annotator_conf.get('name', "faiss-hybrid-annotator")
        self.model_name = annotator_conf.get(
            'model_name', 'sentence-transformers/all-MiniLM-L6-v2'
        )  # JINA provides better embeddings

        # Retrieval tasks for Jina v3 dual-encoder usage
        # Use passage/doc task for labels (documents), query task for search strings
        self.query_task = annotator_conf.get('query_task', 'retrieval.query')
        self.doc_task = annotator_conf.get('doc_task', 'retrieval.passage')

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
        self._norm_label_map: dict[str, str] = {}
        for lbl in self.target_labels:
            self._norm_label_map[self._normalize_for_match(lbl)] = lbl
        self._norm_labels = list(self._norm_label_map.keys())

        self._debug = bool(self.annotator_conf.get("debug", True))

        self.dynamic_ngram_thresholds = annotator_conf.get(
            'dynamic_ngram_thresholds', {"short": 3, "medium": 7, "long": 15}
        )

        self.model_name = 'jinaai/jina-embeddings-v3'
        self.model_name = 'Snowflake/snowflake-arctic-embed-l-v2.0'  #
        # self.model_name = 'Snowflake/snowflake-arctic-embed-m-v2.0'  #

        self.model = _get_transformer(
            self.model_name,
            False,
            revision="main",
            trust_remote_code=True,
        )
        self.model.max_seq_length = 1024  # input sequence length up to 8192
        print('self.model_name : ', self.model_name)

        self.query_task = 'text-matching'
        self.doc_task = 'text-matching'

        self.query_task = 'query'
        self.doc_task = 'query'

        # IMPORTANT: encode label embeddings as documents/passages
        self.label_embeddings = self.model.encode(
            self.target_labels,
            normalize_embeddings=True,
            task=self.doc_task,
            prompt_name=self.doc_task,
        )

        self.index = self.build_faiss_index(self.label_embeddings)
        self.embedding_cache = {}
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
            # model_mml = _get_transformer(self.model_name, True,
            #                              revision="main", trust_remote_code=True,
            #                              )
            # model_mml.max_seq_length = 1024  # input sequence length up to 8192

            self.multiline_matcher = MultiLinePatternMatcher(
                model=self.model,
                threshold=self.multiline_threshold,
                reference_blocks=ref_blocks,
                enable_rerank=True,  # disable for now, as it requires more gpu memory
                debug=False,
            )

        self.output_dir = os.path.join(working_dir, "agent-output", self.name)
        os.makedirs(self.output_dir, exist_ok=True)

    def __del__(self):
        try:
            torch_gc()
        except Exception:
            # Avoid raising in destructor
            pass

    @property
    def capabilities(self) -> list:
        return [AnnotatorCapabilities.EXTRACTOR]

    def build_faiss_index(self, embeddings: np.ndarray):
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index

    def _should_log_line(self, text: str) -> bool:
        if not self._debug:
            return False
        t = text.upper()
        # Focus on labels in question to limit noise
        return any(k in t for k in ["CLAIM", "PROVIDER", "TAX", "ID:"])

    def annotate(self, document: UnstructuredDocument, frames: list) -> list[Any]:
        self.validate_document(document)
        lines_by_page = document.lines_by_page
        all_extractions = []

        if self._debug:
            self.logger.debug("annotate(): target_labels=%s", self.target_labels)

        for page_id, lines_with_meta in lines_by_page.items():
            print('***page_id : ', page_id, ' num lines: ', len(lines_with_meta))

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

        # Build and persist a summary
        summary = self._build_summary(all_extractions)
        summary_file = os.path.join(self.output_dir, "annotation_summary.json")
        store_json_object(summary, summary_file)

        # Display the annotation summary in console
        print("=== Annotation Summary ===")
        print(f"- Pages processed: {summary['pages_processed']}")
        print(f"- Total extractions: {summary['total_extractions']}")
        print(f"- Unique labels found: {len(summary['labels_found'])}")
        print(
            f"- Missing labels: {sorted(summary['missing_labels']) if summary['missing_labels'] else 'None'}"
        )
        print(
            f"- Average confidence: {round(summary['avg_confidence'], 3) if summary['avg_confidence'] is not None else 'N/A'}"
        )
        if summary["per_label"]:
            print("- Per-label counts:")
            for label, stats in sorted(summary["per_label"].items()):
                avg_c = (
                    round(stats["avg_confidence"], 3)
                    if stats["avg_confidence"] is not None
                    else "N/A"
                )
                print(f"  • {label}: count={stats['count']}, avg_conf={avg_c}")

        return all_extractions

    def parse_output(self, raw_output: str):
        raise NotImplementedError()

    def concepts(self):
        """
        Return the concepts used for matching.
        """

        concepts = {
            "CLAIM_ID": [
                "CLAIM NUMBER",
                "CLAIM NO",
                "CLAIM #",
                "CLAIM ID",
                "CLM#",
                "CLM NO",
                "CLAIM NUMBER : Z7P4M2Q",
            ],
            "PROVIDER_TAX_ID": [
                "PROVIDER TAX ID",
                "PROVIDER TIN",
                "TAX ID",
                "TIN",
                "FEIN",
                "EIN",
                "PROVIDER TAX ID : 12-3456789",
            ],
            "HCID": ["HCID", "HEALTHCARE ID", "HC-ID", "HC ID", "HC-84Z7-9932"],
            "CHECK_NUMBER": [
                "CHECK NUM",
                "CHECK NUMBER",
                "CHK NO",
                "CHECK #",
                "CHECK NUM : 123456789",
            ],
            "PATIENT_ACCOUNT": [
                "PATIENT ACCOUNT",
                "PATIENT ACCT",
                "ACCOUNT NUMBER",
                "ACCT #",
                "PA-12345-ABC",
            ],
            # Parties
            "EMPLOYEE_NAME": [
                "EMPLOYEE",
                "EMPLOYEE NAME",
                "EMP",
                "EMPLOYEE : John Doe",
            ],
            "PATIENT_NAME": [
                "PATIENT NAME",
                "PT NAME",
                "BENEFICIARY",
                "PATIENT NAME : Jane Smith",
            ],
            "PROVIDER_NAME": [
                "PROVIDER NAME",
                "FACILITY",
                "HOSPITAL",
                "CLINIC",
                "Acme Hospital",
            ],
            # Dates
            "ISSUE_DATE": [
                "ISSUE DATE",
                "ISSUED",
                "STATEMENT DATE",
                "DATE OF ISSUE",
                "ISSUE DATE : 2024-11-05",
            ],
            # Generic header anchors (optional but useful to stabilize)
            "HEADER_ANCHORS": [
                "EXPLANATION OF BENEFITS",
                "EOB",
                "SUMMARY",
                "STATEMENT",
                "REMITTANCE ADVICE",
            ],
        }

        return concepts

    def _process_lines(self, document_lines: List[Any]) -> List[Dict[str, Any]]:
        all_extractions = []
        missing_fields = set(self.target_labels)
        # determine allowed rows
        if self.multiline_enabled:
            print('Multiline enabled, using pattern matcher')

            texts = [lm.line for lm in document_lines]
            blocks = self.multiline_matcher.find_matching_blocks(
                lines=texts,
                window_sizes=(self.multiline_window,),
                per_label=False,
                concept_view=True,
                concepts=self.concepts(),
                drift_monitor=True,
                good_threshold=0.75,
                enforce_good_only=True,
                annotate_good_flag=True,
            )

            allowed = set()
            for b in blocks:
                allowed |= set(range(b["start_line"] - 1, b["end_line"]))
            if self._debug:
                self.logger.info(
                    "Multiline enabled: %d blocks; allowed idx=%s",
                    len(blocks),
                    sorted(list(allowed))[:100],
                )
                for b in blocks:
                    self.logger.info(
                        "Block pattern=%s score=%.3f lines=%s-%s",
                        b.get("pattern"),
                        b.get("score", 0.0),
                        b.get("start_line"),
                        b.get("end_line"),
                    )
                    print(b.get("text"))
                    print(b)
                    print('#' * 80)
        else:
            allowed = set(range(len(document_lines)))
            if self._debug:
                self.logger.debug(
                    "Multiline disabled: all %d lines allowed", len(document_lines)
                )

        for idx, line_with_meta in enumerate(document_lines):
            if idx not in allowed:
                if self._debug and self._should_log_line(line_with_meta.line):
                    self.logger.debug(
                        "Skipping line %d (not allowed): %s",
                        idx + 1,
                        line_with_meta.line,
                    )
                continue

            raw_line = line_with_meta.line
            corrected = self.ocr_corrector.correct(raw_line.strip())

            if self._should_log_line(corrected):
                self.logger.debug("Line %d: %s", idx + 1, corrected)

            tokens = corrected.split()
            i = 0
            while i < len(tokens):
                remaining = " ".join(tokens[i:])
                sizes = self.determine_ngram_sizes(remaining)
                matched = False
                for sz in sizes:
                    if i + sz > len(tokens):
                        continue
                    ngram = " ".join(tokens[i : i + sz])

                    # Log attempted ngram
                    if self._should_log_line(corrected):
                        self.logger.debug(
                            "  try ngram(i=%d, sz=%d): '%s'", i, sz, ngram
                        )

                    emb = self.encode_with_cache([ngram])[0]
                    # Inspect fuzzy first
                    f_label, f_score = self.fuzzy_match(ngram)
                    if self._should_log_line(corrected):
                        self.logger.debug(
                            "    fuzzy -> label=%s score=%.3f (thr=%.3f)",
                            f_label,
                            f_score,
                            self.fuzzy_threshold,
                        )

                    label, score, strat, cands = self.hybrid_match(
                        ngram, emb, top_k=self.top_k_candidates
                    )

                    if self._should_log_line(corrected):
                        if cands:
                            self.logger.debug(
                                "    embed candidates: %s",
                                ", ".join(
                                    [
                                        f"{c['label']}:{c.get('final_score', 0):.3f}"
                                        for c in cands
                                    ]
                                ),
                            )
                        self.logger.debug(
                            "    chosen: label=%s score=%.3f strat=%s (min_final=%.3f min_accept=%.3f)",
                            label,
                            score,
                            strat,
                            self.min_final_score,
                            self.min_acceptance_score,
                        )

                    if label and score >= self.min_acceptance_score:
                        val = self.value_extractor.extract(corrected, label)
                        if self._should_log_line(corrected):
                            self.logger.debug("    extract '%s' => '%s'", label, val)

                        all_extractions.append(
                            self._build_field_result(
                                label, val, idx, score, strat, corrected, sz
                            )
                        )
                        missing_fields.discard(label)
                        i += sz
                        matched = True
                        break
                    else:
                        if self._should_log_line(corrected):
                            reason = []
                            if not label:
                                reason.append("no-label")
                            else:
                                reason.append(f"score<{self.min_acceptance_score:.2f}")
                            self.logger.debug(
                                "    reject ngram '%s' (%s)", ngram, "/".join(reason)
                            )

                if not matched:
                    i += 1

        if self.deduplicate_fields:
            if self._debug:
                self.logger.debug(
                    "Deduplicating %d fields (strategy=%s)",
                    len(all_extractions),
                    self.deduplication_strategy,
                )
            all_extractions = self.deduplicate_field_entries(
                all_extractions, strategy=self.deduplication_strategy
            )

        if self._debug:
            found = [e["label"] for e in all_extractions]
            self.logger.debug("Found labels on page: %s", found)
        return all_extractions

    # --- Normalization helpers (for matching only) ---
    _punct_re = re.compile(
        r"[^\w\s]+"
    )  # remove punctuation but keep spaces and word chars

    def _normalize_for_match(self, text: str) -> str:
        # Uppercase, strip punctuation, collapse whitespace
        t = text.upper()
        t = self._punct_re.sub("", t)  # remove punctuation like ':' '-' etc.
        t = " ".join(t.split())  # collapse spaces
        return t

    def hybrid_match(self, text: str, emb: np.ndarray, top_k: int = 3):
        """
        Try Fuzzy match first -> Embedding match -> Fallback to Memory if enabled.
        """
        # Normalize candidate only for matching; keep original for value extraction
        norm_text = self._normalize_for_match(text)

        # 1) Fuzzy First (on normalized space)
        fuzzy_label, fuzzy_score = self.fuzzy_match(norm_text, already_normalized=True)
        if fuzzy_label and fuzzy_score >= self.fuzzy_threshold:
            return (
                fuzzy_label,
                fuzzy_score,
                "fuzzy-direct",
                [{"label": fuzzy_label, "final_score": fuzzy_score}],
            )

        # 2) Embedding (embedding of normalized text is often more stable in presence of punctuation)
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
            raw_scores = np.array([c["raw_score"] for c in candidates])
            exp_scores = np.exp(raw_scores - np.max(raw_scores))
            softmax_scores = exp_scores / exp_scores.sum()
            for i, c in enumerate(candidates):
                c["final_score"] = softmax_scores[i]
            candidates.sort(key=lambda x: x["final_score"], reverse=True)
            best = candidates[0]
            if best["final_score"] >= self.min_final_score:
                return (
                    best["label"],
                    best["final_score"],
                    "embedding-softmax",
                    candidates,
                )

        return None, 0.0, "none", []

    def boost_if_critical(self, label: str, score: float) -> float:
        return (
            min(score + self.critical_field_boost, 1.0)
            if label in self.critical_fields
            else score
        )

    def fuzzy_match(
        self, candidate: str, already_normalized: bool = False
    ) -> (str, float):
        """
        Fuzzy-match candidate against normalized labels,
        then map back to the original label.
        """
        norm_candidate = (
            candidate if already_normalized else self._normalize_for_match(candidate)
        )
        if not norm_candidate:
            return None, 0.0

        # Use normalized label list for robust matching
        matches = difflib.get_close_matches(
            norm_candidate, self._norm_labels, n=1, cutoff=self.fuzzy_threshold
        )
        if not matches:
            return None, 0.0

        matched_norm = matches[0]
        # Compute score using normalized strings
        score = difflib.SequenceMatcher(None, norm_candidate, matched_norm).ratio()
        original_label = self._norm_label_map.get(matched_norm)
        return (original_label, score) if original_label else (None, 0.0)

    def encode_with_cache(self, texts: List[str]) -> np.ndarray:
        """
        Encode normalized texts for more stable embedding matches.
        Uses retrieval.query task for queries and a task-aware cache.
        """
        new_texts, cached = [], []
        for text in texts:
            norm = self._normalize_for_match(text)
            cache_key = f"{self.query_task}:{norm}"
            if cache_key in self.embedding_cache:
                cached.append(self.embedding_cache[cache_key])
            else:
                new_texts.append(norm)

        if new_texts:
            new_embs = self.model.encode(
                new_texts,
                normalize_embeddings=True,
                task=self.query_task,
                prompt_name=self.query_task,
            )
            for t, e in zip(new_texts, new_embs):
                cache_key = f"{self.query_task}:{t}"
                self.embedding_cache[cache_key] = e
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
            "confidence": round(score, 3),
            "match_strategy": strategy,
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

    def _build_summary(self, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build an annotation summary based on page-level extractions.
        """
        total = 0
        labels_found = set()
        missing_labels = set(self.target_labels)
        per_label: Dict[str, Dict[str, Any]] = {}
        confidences = []

        for page in pages:
            extractions = page.get("extractions", [])
            total += len(extractions)
            for item in extractions:
                label = item.get("label")
                conf = item.get("confidence")
                if label:
                    labels_found.add(label)
                    if label not in per_label:
                        per_label[label] = {"count": 0, "confidences": []}
                    per_label[label]["count"] += 1
                    if isinstance(conf, (int, float)):
                        per_label[label]["confidences"].append(conf)
                        confidences.append(conf)

        missing_labels -= labels_found

        # Aggregate per-label stats
        per_label_stats = {}
        for label, info in per_label.items():
            if info["confidences"]:
                avg_c = float(sum(info["confidences"]) / len(info["confidences"]))
            else:
                avg_c = None
            per_label_stats[label] = {
                "count": info["count"],
                "avg_confidence": avg_c,
            }

        avg_conf = float(sum(confidences) / len(confidences)) if confidences else None

        return {
            "pages_processed": len(pages),
            "total_extractions": total,
            "labels_found": sorted(labels_found),
            "missing_labels": sorted(missing_labels),
            "avg_confidence": avg_conf,
            "per_label": per_label_stats,
            "generated_at": self.get_current_timestamp_utc(),
        }
