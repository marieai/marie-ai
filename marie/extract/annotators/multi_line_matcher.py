import re
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from marie.extract.annotators.reranker import Reranker

# Interpretable Text Embeddings and Text Similarity Explanation: A Primer
# REFERENCE : https://arxiv.org/html/2502.14862v1


@lru_cache(maxsize=None)
def _get_reranker_cached(rerank_model: str, device: Optional[str]) -> Reranker:
    """Cached, shared reranker to avoid repeated GPU allocations"""
    return Reranker(rerank_model, device=device)


def suppress_overlapping_blocks(blocks: List[Dict]) -> List[Dict]:
    blocks = sorted(blocks, key=lambda b: (-b["score"], b["start_line"], b["end_line"]))
    chosen, occupied = [], set()
    for b in blocks:
        rng = set(range(b["start_line"], b["end_line"] + 1))
        if occupied.isdisjoint(rng):
            chosen.append(b)
            occupied.update(rng)
    chosen.sort(key=lambda b: (b["start_line"], b["end_line"]))
    return chosen


_alnum = re.compile(r"[A-Za-z0-9]+")


def _alnum_tokens(s: str) -> List[str]:
    return _alnum.findall((s or "").lower())


def _char_ngrams(s: str, n: int = 3) -> set:
    s = re.sub(r"\s+", " ", (s or "").lower())
    return {s[i : i + n] for i in range(len(s) - n + 1)} if len(s) >= n else set()


def _overlap_score(a: str, b: str) -> float:
    """lexical similarity: token Jaccard + char-3gram Jaccard."""
    ta, tb = set(_alnum_tokens(a)), set(_alnum_tokens(b))
    jt = (len(ta & tb) / max(1, len(ta | tb))) if (ta or tb) else 0.0
    ga, gb = _char_ngrams(a), _char_ngrams(b)
    jc = (len(ga & gb) / max(1, len(ga | gb))) if (ga and gb) else 0.0
    return 0.6 * jt + 0.4 * jc


def _robust_calibrate(
    arr: np.ndarray, lo_q=5.0, hi_q=95.0, flat_eps=1e-3
) -> Tuple[np.ndarray, float]:
    """
    Percentile min–max to [0,1] with a 'flat-spread' guard.
    Returns (norm, conf) where:
      - norm ∈ [0,1] are calibrated scores
      - conf ∈ [0,1] shrinks when the channel's spread is tiny
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr, 0.0
    lo = float(np.percentile(arr, lo_q))
    hi = float(np.percentile(arr, hi_q))
    rng = hi - lo
    if rng < flat_eps:
        # all values look the same -> treat as uninformative
        return np.full_like(arr, 0.5, dtype=np.float32), 0.0
    arr_clip = np.clip(arr, lo, hi)
    norm = (arr_clip - lo) / (rng + 1e-8)
    # confidence grows with spread; 0.2 is a practical spread scale
    conf = float(min(1.0, rng / 0.2))
    return norm.astype(np.float32), conf


class MultiLinePatternMatcher:
    _collapse_ws = re.compile(r"\s+")

    def __init__(
        self,
        model: SentenceTransformer,
        threshold: float = 0.6,
        reference_blocks: Dict[str, Union[str, List[str]]] = None,
        mrl_dims: int = 0,
        line_topk: int = 0,  # 0 => auto top-k
        block_weight: float = 0.4,  # weight for block-vs-late-interaction in embed score
        enable_rerank: bool = False,
        rerank_model: str = "BAAI/bge-reranker-v2-m3",  # 568M
        # rerank_model: str = "BAAI/bge-reranker-base", # 278M
        rerank_top_m: int = 12,
        rerank_batch_size: int = 4,
        device: Optional[str] = None,
        # debug
        debug: bool = True,
        debug_printer: Optional[Callable[[str], None]] = None,
        debug_max_windows: int = 60,
        # calibration knobs
        emb_flat_eps: float = 1e-3,  # flatness guard for embed channel
        rr_flat_eps: float = 1e-4,  # flatness guard for reranker channel (often tiny logits/probs)
        # gating / shaping knobs
        abs_floor_lo: float = 0.20,  # raw embed floor start (typical Arctic: 0.15–0.25)
        abs_floor_hi: float = 0.45,  # raw embed floor full-pass (0.35–0.50)
        min_alnum_tokens: int = 6,  # short windows get penalized
        len_gate_hi: int = 18,  # where length prior saturates
        overlap_gamma: float = 0.75,  # lexical attenuation strength (0.5–1.0)
        zsig_gain: float = 2.0,  # z-sigmoid shaping; try 2.0–3.0 for sharper separation
    ):
        self.model = model
        try:
            self.model.max_seq_length = max(
                getattr(self.model, "max_seq_length", 0) or 0, 8192
            )
        except Exception:
            pass

        self.debug = bool(debug)
        self._print = debug_printer if debug_printer else (lambda s: print(s))
        self.debug_max_windows = int(debug_max_windows)

        print(f"MultiLinePatternMatcher initialized with model: {model}, ")
        print(f"debug={self.debug}, ")
        print(
            f"threshold={threshold}, mrl_dims={mrl_dims}, line_topk={line_topk}, block_weight={block_weight}, "
        )

        self.threshold = threshold
        self.mrl_dims = mrl_dims
        self.line_topk = line_topk
        self.block_weight = block_weight

        self.enable_rerank = enable_rerank
        self.rerank_top_m = rerank_top_m
        self.rerank_batch_size = rerank_batch_size
        self.reranker: Optional[Reranker] = (
            _get_reranker_cached(rerank_model, device) if enable_rerank else None
        )

        # per-channel flatness guards
        self.emb_flat_eps = float(emb_flat_eps)
        self.rr_flat_eps = float(rr_flat_eps)

        # gates / priors
        self.abs_floor_lo = float(abs_floor_lo)
        self.abs_floor_hi = float(abs_floor_hi)
        self.min_alnum_tokens = int(min_alnum_tokens)
        self.len_gate_hi = int(len_gate_hi)
        self.overlap_gamma = float(overlap_gamma)
        self.zsig_gain = float(zsig_gain)

        # concept vectors cache (built on demand when concept_view=True)
        self._concept_vectors: Dict[str, np.ndarray] = {}

        # concept drift stats (EMA)
        self._concept_ema: Dict[str, float] = {}
        self._concept_ema2: Dict[str, float] = {}
        self._concept_count: Dict[str, int] = {}

        self.reference_texts: Dict[str, str] = {}
        self.reference_embs: Dict[str, np.ndarray] = {}
        self.reference_lines: Dict[str, List[str]] = {}
        self.reference_line_embs: Dict[str, np.ndarray] = {}

        if reference_blocks:
            for label, block in reference_blocks.items():
                self.add_reference_block(label, block)

    # ---------- helpers ----------
    def _log(self, msg: str):
        if self.debug:
            self._print(msg)

    def _normalize_for_embed(self, text: str) -> str:
        t = (text or "").strip()
        t = self._collapse_ws.sub(" ", t)
        return t

    def _maybe_truncate(self, x: np.ndarray) -> np.ndarray:
        if self.mrl_dims and x.ndim == 2 and x.shape[1] > self.mrl_dims:
            return x[:, : self.mrl_dims]
        return x

    def _encode(self, texts: Union[str, List[str]], is_query: bool) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        normed = [self._normalize_for_embed(t) for t in texts]
        kwargs = {"normalize_embeddings": True}
        if is_query:
            kwargs["prompt_name"] = "query"
        embs = self.model.encode(normed, **kwargs)
        return self._maybe_truncate(np.asarray(embs, dtype=np.float32))

    def _join_block(self, block: Union[str, List[str]]) -> str:
        if isinstance(block, list):
            parts = [(b or "").strip() for b in block if (b or "").strip()]
            return "\n".join(parts)
        return (block or "").strip()

    # --- concepts: prep and activation ---
    def _prepare_concepts(
        self, concepts: Optional[Dict[str, Union[str, List[str]]]]
    ) -> None:
        """
        Build normalized concept vectors from prototype texts (averaged if list).
        Stored in self._concept_vectors.
        """
        if not concepts:
            self._concept_vectors = {}
            return
        concept_vecs: Dict[str, np.ndarray] = {}
        for name, proto in concepts.items():
            if isinstance(proto, str):
                proto_list = [proto]
            else:
                proto_list = [p for p in (proto or []) if str(p).strip()]
            if not proto_list:
                continue
            emb = self._encode(proto_list, is_query=False)  # normalized embeddings
            vec = emb.mean(axis=0)
            norm = np.linalg.norm(vec) + 1e-12
            concept_vecs[name] = (vec / norm).astype(np.float32)
        self._concept_vectors = concept_vecs

    def _concept_activations_map(self, block_text: str) -> Dict[str, float]:
        """
        Return a dict {concept: score} for all prepared concepts.
        """
        if not self._concept_vectors:
            return {}
        q = self._encode(block_text, is_query=True)[0]  # normalized
        return {
            name: float(np.dot(q, vec)) for name, vec in self._concept_vectors.items()
        }

    def _concept_activations(
        self, block_text: str, top_k: int = 5
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Project the block onto prepared concepts using cosine (dot) with normalized vectors.
        Returns top-k activations as [{concept, score}].
        """
        scores_map = self._concept_activations_map(block_text)
        if not scores_map:
            return []
        items = sorted(scores_map.items(), key=lambda x: x[1], reverse=True)
        return [{"concept": n, "score": s} for n, s in items[: max(0, top_k)]]

    def _apply_concept_policy(
        self, label: str, act_map: Dict[str, float], policy: Optional[Dict[str, Dict]]
    ) -> Tuple[bool, List[str]]:
        """
        Apply concept policy for a given label.
        Policy schema:
          {
            "<label>" or "*": {
              "min_thresholds": {"CONCEPT_A": 0.6, "CONCEPT_B": 0.5},
              "any_of": [
                 {"concepts": ["A","B","C"], "threshold": 0.6, "count": 1}
              ]
            }
          }
        Returns (passed, violations[])
        """
        if not policy:
            return True, []
        rules = policy.get(label) or policy.get("*")
        if not rules:
            return True, []
        violations = []

        # min_thresholds: every listed concept must be >= threshold
        mins = rules.get("min_thresholds", {}) or {}
        for cname, thr in mins.items():
            val = act_map.get(cname, 0.0)
            if val < float(thr):
                violations.append(f"{cname}<{float(thr):.2f} (got {val:.2f})")

        # any_of: for each group, at least 'count' concepts must be >= threshold
        groups = rules.get("any_of", []) or []
        for g in groups:
            lst = g.get("concepts", []) or []
            thr = float(g.get("threshold", 0.0))
            need = int(g.get("count", 1))
            ok = sum(1 for c in lst if act_map.get(c, 0.0) >= thr)
            if ok < need:
                violations.append(
                    f"any_of({','.join(lst)}) need {need} @ ≥{thr:.2f} (got {ok})"
                )

        passed = len(violations) == 0
        return passed, violations

    def _update_and_check_drift(
        self, act_map: Dict[str, float], alpha: float = 0.1, alert_z: float = 3.0
    ) -> List[Dict[str, float]]:
        """
        Update EMA stats per concept and compute drift alerts.
        Returns a list of {concept, value, mean, std, z, alert}.
        """
        out = []
        eps = 1e-6
        for cname, val in act_map.items():
            if cname not in self._concept_count:
                # initialize
                self._concept_count[cname] = 0
                self._concept_ema[cname] = float(val)
                self._concept_ema2[cname] = float(val * val)
            else:
                # EMA updates
                self._concept_ema[cname] = (1 - alpha) * self._concept_ema[
                    cname
                ] + alpha * float(val)
                self._concept_ema2[cname] = (1 - alpha) * self._concept_ema2[
                    cname
                ] + alpha * float(val * val)
            self._concept_count[cname] += 1

            mean = self._concept_ema[cname]
            var = max(0.0, self._concept_ema2[cname] - mean * mean)
            std = float(np.sqrt(var + eps))
            z = (float(val) - mean) / (std + eps)
            alert = abs(z) >= float(alert_z)
            out.append(
                {
                    "concept": cname,
                    "value": float(val),
                    "mean": float(mean),
                    "std": float(std),
                    "z": float(z),
                    "alert": bool(alert),
                }
            )
        return out

    # ---------- reference prep ----------
    def add_reference_block(self, label: str, block: Union[str, List[str]]):
        joined = self._join_block(block)
        if not joined:
            raise ValueError(f"Reference '{label}' is empty.")
        self.reference_texts[label] = joined

        lines = [l.strip() for l in joined.split("\n") if l.strip()]
        self.reference_lines[label] = lines

        self.reference_embs[label] = self._encode(joined, is_query=False)[0]
        self.reference_line_embs[label] = self._encode(lines or [""], is_query=False)

        self._log(f"[ref] {label}: {len(lines)} line(s)")

    # ---------- scoring ----------
    def _score_block_embed(
        self, block_text: str
    ) -> Tuple[Optional[str], float, Dict[str, float]]:
        q_block = self._encode(block_text, is_query=True)[0]
        q_lines = [l.strip() for l in block_text.split("\n") if l.strip()]
        q_line_embs = self._encode(q_lines or [""], is_query=True)

        best_label, best_score = None, -1.0
        best_diag = {}
        for label, r_block in self.reference_embs.items():
            s_block = float(np.dot(q_block, r_block))
            r_line_embs = self.reference_line_embs.get(label)

            if r_line_embs is None or r_line_embs.size == 0 or q_line_embs.size == 0:
                # fall back to block-only
                s_rows = s_block
                s_cols = s_block
                k_rows = 1
                k_cols = 1
            else:
                sims = q_line_embs @ r_line_embs.T  # [n_q, n_r]

                # row-wise: each query line to best ref line
                row_max = sims.max(axis=1)
                k_rows = self.line_topk or min(row_max.shape[0], r_line_embs.shape[0])
                k_rows = max(k_rows, 1)
                s_rows = float(np.partition(row_max, -k_rows)[-k_rows:].mean())

                # col-wise: each ref line to best query line
                col_max = sims.max(axis=0)
                k_cols = self.line_topk or min(col_max.shape[0], q_line_embs.shape[0])
                k_cols = max(k_cols, 1)
                s_cols = float(np.partition(col_max, -k_cols)[-k_cols:].mean())

            # symmetric late interaction via harmonic mean
            if s_rows > 0 and s_cols > 0:
                s_li = 2 * s_rows * s_cols / (s_rows + s_cols)
            else:
                s_li = max(s_rows, s_cols)

            s_embed = self.block_weight * s_block + (1.0 - self.block_weight) * s_li

            if s_embed > best_score:
                best_score = s_embed
                best_label = label
                best_diag = {
                    "s_block": s_block,
                    "s_rows": s_rows,
                    "s_cols": s_cols,
                    "s_li": s_li,
                    "s_embed": s_embed,
                    "k_rows": k_rows,
                    "k_cols": k_cols,
                    "q_lines": len(q_lines),
                    "r_lines": (
                        r_line_embs.shape[0]
                        if r_line_embs is not None and r_line_embs.size
                        else 0
                    ),
                }

        return best_label, best_score, best_diag

    def _windows(
        self, lines: List[str], window_sizes: Tuple[int, ...]
    ) -> List[Tuple[str, int, int, int]]:
        out, n = [], len(lines)
        for w in sorted(set(k for k in window_sizes if k >= 1)):
            if w > n:
                continue
            for s in range(0, n - w + 1):
                e = s + w - 1
                out.append(("\n".join(lines[s : e + 1]), s, e, w))
        return out

    def find_matching_blocks(
        self,
        lines: List[str],
        window_sizes: Tuple[int, ...] = (2, 3, 4),
        per_label: bool = False,
        fallback_top1_if_none: bool = True,
        allow_overlap: bool = False,
        # concept outputs
        concept_view: bool = False,
        concepts: Optional[Dict[str, Union[str, List[str]]]] = None,
        concept_top_k: int = 5,
        # concept governance
        concept_policy: Optional[Dict[str, Dict]] = None,
        concept_policy_mode: str = "annotate",  # "annotate" | "filter"
        # drift monitoring
        drift_monitor: bool = False,
        drift_alpha: float = 0.1,
        drift_alert_z: float = 3.0,
        # NEW: match-quality thresholds
        good_threshold: Optional[float] = None,  # e.g., 0.75 for “GOOD”
        enforce_good_only: bool = False,  # if True, drop non-good matches
        annotate_good_flag: bool = True,  # add is_good flag to outputs
    ) -> List[Dict]:
        if not lines or not self.reference_embs:
            return []

        # prepare concepts when any concept-related feature is used
        if concept_view or concept_policy or drift_monitor:
            self._prepare_concepts(concepts)

        # 1) embed candidates
        cands: List[Tuple[float, str, int, int, int, str, Dict[str, float]]] = []
        for wtxt, s, e, w in self._windows(lines, window_sizes):
            lbl, sc, diag = self._score_block_embed(wtxt)
            if lbl is None:
                continue
            cands.append((float(sc), lbl, s, e, w, wtxt, diag))
        if not cands:
            return []

        # 2) re-rank per label on top-M + robust, variance-aware calibration + gates
        final_cands: List[Tuple[float, str, int, int, int, str, Dict[str, float]]] = []
        grouped: Dict[
            str, List[Tuple[float, str, int, int, int, str, Dict[str, float]]]
        ] = {}
        for c in cands:
            grouped.setdefault(c[1], []).append(c)

        for lbl, arr in grouped.items():
            arr.sort(key=lambda t: t[0], reverse=True)
            take = arr[: self.rerank_top_m]

            # per-label robust calibration on embed channel
            emb_raw = np.array([t[0] for t in take], dtype=np.float32)
            emb_mm, emb_conf = _robust_calibrate(
                emb_raw, lo_q=5.0, hi_q=95.0, flat_eps=self.emb_flat_eps
            )

            # reranker (window, reference)
            if self.enable_rerank and self.reranker is not None:
                pairs = [(t[5], self.reference_texts.get(lbl, "")) for t in take]
                rr_raw = self.reranker.score(pairs, batch_size=self.rerank_batch_size)
                rr_mm, rr_conf = _robust_calibrate(
                    rr_raw, lo_q=5.0, hi_q=95.0, flat_eps=self.rr_flat_eps
                )
            else:
                rr_raw = np.zeros_like(emb_raw)
                rr_mm, rr_conf = np.zeros_like(emb_mm), 0.0

            # confidence-adaptive base blend (no fixed alpha)
            w_rr, w_emb = float(rr_conf), float(emb_conf)
            den = max(1e-6, w_rr + w_emb)
            base = (w_rr * rr_mm + w_emb * emb_mm) / den  # ∈[0,1]

            # absolute embed floor
            abs_lo, abs_hi = self.abs_floor_lo, self.abs_floor_hi
            abs_gate = np.clip(
                (emb_raw - abs_lo) / max(1e-6, (abs_hi - abs_lo)), 0.0, 1.0
            )

            # lexical overlap attenuation
            ref_text = self.reference_texts.get(lbl, "")
            overlaps = np.array(
                [_overlap_score(t[5], ref_text) for t in take], dtype=np.float32
            )
            ov_floor = 0.55
            gam = max(1e-6, float(self.overlap_gamma))
            ov_gate = ov_floor + (1.0 - ov_floor) * np.power(
                np.clip(overlaps, 0.0, 1.0), gam
            )

            # short-window prior
            lens = np.array([len(_alnum_tokens(t[5])) for t in take], dtype=np.float32)
            len_gate = np.clip(
                (lens - self.min_alnum_tokens)
                / max(1.0, (self.len_gate_hi - self.min_alnum_tokens)),
                0.0,
                1.0,
            )

            # combine gates
            gated = base * abs_gate * ov_gate * len_gate

            # optional z-sigmoid
            if self.zsig_gain > 0.0:
                eps = 1e-6
                z = np.clip(gated, eps, 1 - eps)
                logits = np.log(z / (1 - z)) * float(self.zsig_gain)
                gated = 1 / (1 + np.exp(-logits))

            for i, t in enumerate(take):
                diag2 = dict(t[6])
                diag2.update(
                    {
                        "emb_raw": float(emb_raw[i]),
                        "emb_mm": float(emb_mm[i]),
                        "emb_conf": float(emb_conf),
                        "rr_raw": float(rr_raw[i]),
                        "rr_mm": float(rr_mm[i]),
                        "rr_conf": float(rr_conf),
                        "abs_gate": float(abs_gate[i]),
                        "overlap": float(overlaps[i]),
                        "ov_gate": float(ov_gate[i]),
                        "len_tokens": float(lens[i]),
                        "len_gate": float(len_gate[i]),
                        "final": float(gated[i]),
                    }
                )
                final_cands.append(
                    (float(gated[i]), lbl, t[2], t[3], t[4], t[5], diag2)
                )

        # 3) rank & optional debug
        final_cands.sort(key=lambda t: t[0], reverse=True)

        if self.debug:
            self._log("=== PatternMatcher DEBUG ===")
            self._log(f"labels: {list(self.reference_texts.keys())}")
            self._log(f"candidates: {len(final_cands)} (top {self.debug_max_windows})")
            for idx, (sc, lbl, s, e, w, txt, d) in enumerate(
                final_cands[: self.debug_max_windows], 1
            ):
                prev = txt.replace("\n", " ⏎ ")[:240]
                self._log(
                    f"[{idx:03d}] {lbl} lines={s + 1}-{e + 1} w={w} final={sc:.3f} "
                    f"| emb_mm={d.get('emb_mm', 0):.3f} (conf={d.get('emb_conf', 0):.2f}) "
                    f"rr_mm={d.get('rr_mm', 0):.3f} (conf={d.get('rr_conf', 0):.2f}) "
                    f"(raw: emb={d.get('emb_raw', 0):.3f} rr={d.get('rr_raw', 0):.3f})"
                )
                self._log(
                    f"      gates: abs={d.get('abs_gate', 0):.3f} "
                    f"ov={d.get('ov_gate', 0):.3f} len={d.get('len_gate', 0):.3f} "
                    f"(overlap={d.get('overlap', 0):.3f}, tokens={int(d.get('len_tokens', 0))})"
                )
                self._log(
                    f"      embed parts: block={d.get('s_block', 0):.3f} "
                    f"rows={d.get('s_rows', 0):.3f} cols={d.get('s_cols', 0):.3f} "
                    f"li={d.get('s_li', 0):.3f} k_r={d.get('k_rows', 0)} k_c={d.get('k_cols', 0)}"
                )
                self._log(f"      text: {prev}")
            self._log("=== /DEBUG ===")

        # 4) select (+ concept governance + drift)
        def build_item(
            label: str,
            score: float,
            start_idx: int,
            end_idx: int,
            window_size: int,
            window_text: str,
        ) -> Dict:
            item = {
                "pattern": label,
                "score": round(score, 3),
                "start_line": start_idx + 1,
                "end_line": end_idx + 1,
                "text": window_text,
                "window": window_size,
            }
            # concepts (optional)
            if concept_view and self._concept_vectors:
                item["concepts"] = self._concept_activations(
                    window_text, top_k=concept_top_k
                )
            # “GOOD” flag (optional)
            if annotate_good_flag and good_threshold is not None:
                item["is_good"] = bool(score >= float(good_threshold))
            return item

        matched_blocks: List[Dict] = []
        if per_label:
            best_by_label: Dict[str, Tuple[float, int, int, int, str]] = {}
            for (
                score,
                label,
                start_idx,
                end_idx,
                window_size,
                window_text,
                _diag,
            ) in final_cands:
                current = best_by_label.get(label)
                if (current is None) or (score > current[0]):
                    best_by_label[label] = (
                        score,
                        start_idx,
                        end_idx,
                        window_size,
                        window_text,
                    )

            for label, (
                score,
                start_idx,
                end_idx,
                window_size,
                window_text,
            ) in best_by_label.items():
                # base acceptance uses class threshold (existing behavior)
                if score >= self.threshold or not fallback_top1_if_none:
                    item = build_item(
                        label, score, start_idx, end_idx, window_size, window_text
                    )
                    # enforce “good” quality if requested
                    if (
                        enforce_good_only
                        and good_threshold is not None
                        and score < float(good_threshold)
                    ):
                        continue
                    matched_blocks.append(item)

            if not matched_blocks and fallback_top1_if_none and final_cands:
                score, label, start_idx, end_idx, window_size, window_text, _ = (
                    final_cands[0]
                )
                item = build_item(
                    label, score, start_idx, end_idx, window_size, window_text
                )
                if not (
                    enforce_good_only
                    and good_threshold is not None
                    and score < float(good_threshold)
                ):
                    matched_blocks = [item]
                else:
                    matched_blocks = []

            if not allow_overlap and len(matched_blocks) > 1:
                matched_blocks = suppress_overlapping_blocks(matched_blocks)
        else:
            if not final_cands:
                return []
            score, label, start_idx, end_idx, window_size, window_text, _ = final_cands[
                0
            ]
            item = build_item(
                label, score, start_idx, end_idx, window_size, window_text
            )
            if (
                enforce_good_only
                and good_threshold is not None
                and score < float(good_threshold)
            ):
                matched_blocks = []
            else:
                matched_blocks = [item]

        return matched_blocks
