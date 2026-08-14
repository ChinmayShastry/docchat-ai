# HYBRID RETRIEVER
# Combines BM25 (keyword matching) and semantic (embedding) search, then
# reranks the merged candidate pool with a CrossEncoder for precision.
#
# Flow:
#   1. BM25 scores every chunk        → top-N by keyword relevance
#   2. Vector store scores every chunk → top-N by embedding similarity
#   3. Both score sets are min-max normalised onto a comparable 0-1 scale
#   4. Scores are blended:  alpha * semantic + (1 - alpha) * bm25
#   5. Low-confidence candidates are filtered (with a floor on pool size)
#   6. CrossEncoder reranks the survivors → top-k returned
#
# Note on step 3/4: chunks found by only one arm have no score from the other,
# so the missing side contributes 0. This biases slightly toward chunks both
# arms agree on, which is the desired behaviour for grounded QA. Reciprocal
# Rank Fusion is the main alternative and avoids the scale-mixing question
# entirely, but it discards score magnitude, which the threshold filter needs.

import hashlib
import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from config import Config
from src.logging_setup import get_logger

log = get_logger(__name__)

VALID_MODES = ("bm25", "semantic", "hybrid", "hybrid_rerank")


@lru_cache(maxsize=4)
def get_reranker(model_name):
    """Load and cache the cross-encoder.

    Imported lazily so that unit tests and BM25-only evaluation runs never pay
    the multi-second model load, and so the package imports cleanly in
    environments without torch installed.
    """
    from sentence_transformers import CrossEncoder

    log.info("Loading cross-encoder reranker: %s", model_name)
    return CrossEncoder(model_name)


def _content_key(doc):
    """Stable identity for a chunk, based on its text.

    The previous implementation keyed score lookups on id(doc). Vector stores
    return freshly constructed Document objects on every query, so those ids
    never matched the original chunk objects held in memory — every BM25
    candidate silently scored 0 and was filtered out, making "hybrid" retrieval
    behave as semantic-only. Hashing the content gives an identity that is
    stable across object boundaries.
    """
    return hashlib.sha1(doc.page_content.encode("utf-8", "replace")).hexdigest()


def _normalise(scores):
    """Min-max scale an array onto 0-1, tolerating constant input."""
    if len(scores) == 0:
        return scores

    lo = float(np.min(scores))
    hi = float(np.max(scores))

    if hi - lo < 1e-9:
        # Every candidate scored identically — no signal to preserve.
        return np.zeros_like(scores, dtype=float)

    return (scores - lo) / (hi - lo)


class HybridRetriever:
    def __init__(self, chunks, vectorstore, alpha=None, mode=None):
        self.chunks = chunks
        self.vectorstore = vectorstore
        self.alpha = Config.HYBRID_ALPHA if alpha is None else alpha
        self.mode = mode or Config.RETRIEVAL_MODE

        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Unknown retrieval mode {self.mode!r}. Expected one of {VALID_MODES}."
            )

        tokenized = [self._tokenize(c.page_content) for c in chunks]

        # BM25Okapi raises on a fully empty corpus; guard so an unreadable
        # upload surfaces as a clean error upstream rather than a library trace.
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

        # Map content key → the canonical chunk object, so results always carry
        # the original metadata even when the vector store returns copies.
        self._by_key = {_content_key(c): c for c in chunks}

        log.info(
            "HybridRetriever ready: %d chunks | mode=%s | alpha=%.2f",
            len(chunks), self.mode, self.alpha,
        )

    @staticmethod
    def _tokenize(text):
        return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()

    # ── Scoring arms ──────────────────────────────────────────────────────

    def _bm25_scores(self, query):
        """Return {content_key: normalised_score} for the top BM25 candidates."""
        if self.bm25 is None:
            return {}

        raw = np.asarray(self.bm25.get_scores(self._tokenize(query)), dtype=float)
        if raw.size == 0:
            return {}

        normalised = _normalise(raw)
        top_idx = np.argsort(raw)[-Config.BM25_CANDIDATES:][::-1]

        return {_content_key(self.chunks[i]): float(normalised[i]) for i in top_idx}

    def _semantic_scores(self, query):
        """Return {content_key: normalised_score} for the top vector hits."""
        k = min(Config.SEMANTIC_CANDIDATES, len(self.chunks))
        if k == 0:
            return {}

        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
        if not results:
            return {}

        raw = np.asarray([score for _, score in results], dtype=float)
        normalised = _normalise(raw)

        scored = {}
        for (doc, _), score in zip(results, normalised, strict=True):
            key = _content_key(doc)
            # Register any chunk the store knows about but we haven't seen, so
            # results are never dropped just because identity drifted.
            self._by_key.setdefault(key, doc)
            scored[key] = float(score)

        return scored

    # ── Public API ────────────────────────────────────────────────────────

    def retrieve(self, query, k=None):
        """Return the top-k chunks for `query`."""
        return [chunk for chunk, _ in self.retrieve_with_scores(query, k=k)]

    def retrieve_with_scores(self, query, k=None):
        """Return [(chunk, score)] for `query`, highest scoring first.

        Exposing scores lets the UI show *why* an answer was grounded and lets
        the evaluation harness measure ranking quality directly.
        """
        k = k or Config.RETRIEVAL_K

        if not self.chunks:
            return []

        bm25 = self._bm25_scores(query) if self.mode != "semantic" else {}
        semantic = self._semantic_scores(query) if self.mode != "bm25" else {}

        # ── Blend ─────────────────────────────────────────────────────────
        blended = {}
        for key in set(bm25) | set(semantic):
            b = bm25.get(key, 0.0)
            s = semantic.get(key, 0.0)

            if self.mode == "bm25":
                blended[key] = b
            elif self.mode == "semantic":
                blended[key] = s
            else:
                blended[key] = self.alpha * s + (1.0 - self.alpha) * b

        if not blended:
            log.warning("No candidates scored for query; returning first %d chunks", k)
            return [(c, 0.0) for c in self.chunks[:k]]

        ranked = sorted(blended.items(), key=lambda kv: kv[1], reverse=True)

        # ── Filter, with a floor so we never starve the generator ─────────
        surviving = [(key, sc) for key, sc in ranked if sc > Config.SEMANTIC_THRESHOLD]

        if len(surviving) < Config.MIN_CANDIDATES:
            log.debug(
                "Threshold %.2f left only %d candidates; falling back to top %d",
                Config.SEMANTIC_THRESHOLD, len(surviving), Config.MIN_CANDIDATES,
            )
            surviving = ranked[:max(Config.MIN_CANDIDATES, k)]

        surviving = surviving[:Config.RERANK_CANDIDATES]
        candidates = [(self._by_key[key], sc) for key, sc in surviving if key in self._by_key]

        if not candidates:
            return [(c, 0.0) for c in self.chunks[:k]]

        # ── Rerank ────────────────────────────────────────────────────────
        if self.mode == "hybrid_rerank":
            candidates = self._rerank(query, [c for c, _ in candidates])

        log.debug("Query %r → %d candidates → top %d", query[:60], len(surviving), k)
        return candidates[:k]

    def _rerank(self, query, candidate_chunks):
        """Score (query, chunk) pairs with the cross-encoder and re-sort.

        A reranker failure degrades to the blended ordering rather than taking
        the whole request down — retrieval is still useful without it.
        """
        try:
            reranker = get_reranker(Config.RERANKER_MODEL)
            pairs = [(query, c.page_content) for c in candidate_chunks]
            scores = reranker.predict(pairs, batch_size=16)
        except Exception as exc:
            log.warning("Reranking failed (%s); using blended order", exc)
            return [(c, 0.0) for c in candidate_chunks]

        ranked = sorted(
            zip(candidate_chunks, (float(s) for s in scores), strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked
