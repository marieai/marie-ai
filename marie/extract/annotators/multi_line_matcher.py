from typing import Dict, List, Tuple, Union

import numpy as np
from sentence_transformers import SentenceTransformer


def suppress_overlapping_blocks(matches: List[Dict]) -> List[Dict]:
    sorted_matches = sorted(matches, key=lambda x: x['score'], reverse=True)
    final_matches = []
    occupied_lines = set()

    for match in sorted_matches:
        match_lines = set(range(match['start_line'], match['end_line'] + 1))
        if occupied_lines.isdisjoint(match_lines):
            final_matches.append(match)
            occupied_lines.update(match_lines)
        else:
            print(
                f"[Dropped Overlapping Block] score={match['score']}, lines={match['start_line']}-{match['end_line']}, pattern={match['pattern']}"
            )

    return final_matches


class MultiLinePatternMatcher:
    def __init__(
        self,
        model: SentenceTransformer,
        threshold: float = 0.85,
        reference_blocks: Dict[str, Union[str, List[str]]] = None,
    ):
        self.model = model
        self.threshold = threshold
        self.reference_blocks: Dict[str, str] = {}
        self.reference_block_embeddings: Dict[str, np.ndarray] = {}

        if reference_blocks:
            for label, block in reference_blocks.items():

                # accept single string or list of strings
                if isinstance(block, list):
                    text = "\n".join(block)
                else:
                    text = block
                self.add_reference_block(label, text)

    def add_reference_block(self, label: str, text: str):
        if label in self.reference_blocks:
            print(f"[Warning] Overwriting existing reference block for label: {label}")
        if not text:
            raise ValueError(f"Text for reference block '{label}' cannot be empty.")

        self.reference_blocks[label] = text
        self.reference_block_embeddings[label] = self.model.encode(
            text, normalize_embeddings=True
        )

    def sliding_line_blocks(
        self, lines: List[str], window: int = 2
    ) -> List[Tuple[str, Tuple[int, int]]]:
        return [
            ("\n".join(lines[i : i + window]), (i, i + window - 1))
            for i in range(len(lines) - window + 1)
        ]

    def match_line_block_to_pattern(self, block_text: str) -> Tuple[str, float]:
        emb = self.model.encode(block_text, normalize_embeddings=True)
        best_label = None
        best_score = 0.0

        for pattern_name, ref_emb in self.reference_block_embeddings.items():
            score = float(np.dot(emb, ref_emb))
            if score > best_score:
                best_score = score
                best_label = pattern_name

        return (
            (best_label, best_score)
            if best_score >= self.threshold
            else (None, best_score)
        )

    def find_matching_blocks(self, lines: List[str], window: int = 2) -> List[Dict]:
        results = []
        for block_text, (start_idx, end_idx) in self.sliding_line_blocks(lines, window):
            # print(f"[Matching Block] start_line={start_idx + 1}, end_line={end_idx + 1}, text='{block_text}'")
            pattern, score = self.match_line_block_to_pattern(block_text)
            # print(f"[Matching Result] score={score}")
            if pattern:
                results.append(
                    {
                        "pattern": pattern,
                        "score": round(score, 3),
                        "start_line": start_idx + 1,
                        "end_line": end_idx + 1,
                        "text": block_text,
                    }
                )
        return suppress_overlapping_blocks(results)
