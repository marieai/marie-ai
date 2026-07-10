"""BM25-based skill search index.

This module provides fast skill discovery using BM25 ranking via the bm25s library.
It replaces the simple keyword matching in SkillMetadata.matches_query() with
proper information retrieval ranking.

BM25s achieves 500x speedup over rank-bm25 via sparse matrix computation.
See: https://github.com/xhluca/bm25s
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

from marie.logging_core.logger import MarieLogger

if TYPE_CHECKING:
    from marie.agent.skills.models import Skill

logger = MarieLogger(__name__)

# Lazy import bm25s to avoid import errors if not installed
_bm25s = None


def _get_bm25s():
    """Lazy import bm25s module."""
    global _bm25s
    if _bm25s is None:
        try:
            import bm25s

            _bm25s = bm25s
        except ImportError:
            logger.warning(
                "bm25s not installed. Install with: uv add 'bm25s[core]>=0.2.0'"
            )
            _bm25s = False
    return _bm25s if _bm25s else None


class SkillSearchIndex:
    """BM25-based skill search index.

    Provides fast skill discovery using BM25 ranking. Skills are indexed
    by their name, description, and tags.

    Example:
        >>> index = SkillSearchIndex()
        >>> index.build_index(skills)
        >>> results = index.search("extract invoice data", top_k=3)
        >>> for skill, score in results:
        ...     print(f"{skill.name}: {score:.2f}")
    """

    def __init__(self):
        """Initialize the search index."""
        self._skills: List[Skill] = []
        self._retriever = None
        self._indexed = False

    @property
    def is_available(self) -> bool:
        """Check if BM25 search is available (bm25s installed)."""
        return _get_bm25s() is not None

    @property
    def num_skills(self) -> int:
        """Number of indexed skills."""
        return len(self._skills)

    def build_index(self, skills: List[Skill]) -> None:
        """Build BM25 index from skill metadata.

        Args:
            skills: List of skills to index
        """
        bm25s = _get_bm25s()
        if bm25s is None:
            logger.warning("BM25 search unavailable, falling back to linear search")
            self._skills = skills
            self._indexed = False
            return

        self._skills = list(skills)

        if not self._skills:
            self._retriever = None
            self._indexed = False
            return

        # Build corpus from skill metadata
        corpus = []
        for skill in self._skills:
            # Combine name, description, and tags for indexing
            text = (
                f"{skill.metadata.name} "
                f"{skill.metadata.description} "
                f"{' '.join(skill.metadata.tags)}"
            )
            corpus.append(text)

        try:
            # Tokenize and index using bm25s
            corpus_tokens = bm25s.tokenize(corpus, lower=True)
            self._retriever = bm25s.BM25()
            self._retriever.index(corpus_tokens)
            self._indexed = True
            logger.debug(f"Built BM25 index for {len(self._skills)} skills")
        except Exception as e:
            logger.warning(f"Failed to build BM25 index: {e}")
            self._indexed = False

    def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[Skill, float]]:
        """Search skills by query, return top-k with scores.

        Args:
            query: Search query string
            top_k: Maximum number of results to return
            threshold: Minimum score threshold (0.0 to include all)

        Returns:
            List of (skill, score) tuples, sorted by score descending
        """
        if not self._skills:
            return []

        # If BM25 not available or not indexed, fall back to linear search
        if not self._indexed or self._retriever is None:
            return self._fallback_search(query, top_k, threshold)

        bm25s = _get_bm25s()
        if bm25s is None:
            return self._fallback_search(query, top_k, threshold)

        try:
            # Tokenize query
            query_tokens = bm25s.tokenize([query], lower=True)

            # Retrieve top-k results
            # Note: top_k should not exceed number of skills
            k = min(top_k, len(self._skills))
            results, scores = self._retriever.retrieve(query_tokens, k=k)

            # results and scores are 2D arrays (batch_size=1, k)
            indices = results[0]
            score_values = scores[0]

            # Filter by threshold and pair with skills
            filtered = [
                (self._skills[idx], float(score))
                for idx, score in zip(indices, score_values)
                if score >= threshold
            ]

            return filtered

        except Exception as e:
            logger.warning(f"BM25 search failed: {e}, falling back to linear search")
            return self._fallback_search(query, top_k, threshold)

    def _fallback_search(
        self,
        query: str,
        top_k: int,
        threshold: float,
    ) -> List[Tuple[Skill, float]]:
        """Fallback to SkillMetadata.matches_query() when BM25 unavailable.

        Args:
            query: Search query string
            top_k: Maximum number of results
            threshold: Minimum score threshold

        Returns:
            List of (skill, score) tuples
        """
        results = []
        for skill in self._skills:
            score = skill.metadata.matches_query(query)
            if score >= threshold:
                results.append((skill, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def clear(self) -> None:
        """Clear the index."""
        self._skills = []
        self._retriever = None
        self._indexed = False
