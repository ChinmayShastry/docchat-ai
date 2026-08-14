"""Tests for hybrid retrieval scoring.

The central case is the regression test for the identity bug: score lookups
used to be keyed on id(doc), which never matched between vector-store results
and in-memory chunks, so BM25 candidates were always discarded.
"""

import numpy as np
import pytest

from src.retriever import HybridRetriever, _content_key, _normalise
from tests.conftest import FakeVectorStore


def make_retriever(docs, scored_texts=None, mode="hybrid", alpha=0.5):
    if scored_texts is None:
        scored_texts = [(d.page_content, 0.9 - 0.1 * i) for i, d in enumerate(docs)]
    return HybridRetriever(
        chunks=docs,
        vectorstore=FakeVectorStore(scored_texts),
        alpha=alpha,
        mode=mode,
    )


# ── Score normalisation ───────────────────────────────────────────────────

def test_normalise_maps_to_unit_range():
    result = _normalise(np.array([2.0, 4.0, 6.0]))
    assert result.min() == pytest.approx(0.0)
    assert result.max() == pytest.approx(1.0)


def test_normalise_handles_constant_scores():
    """A flat score array carries no ranking signal and must not divide by zero."""
    result = _normalise(np.array([3.0, 3.0, 3.0]))
    assert np.allclose(result, 0.0)


def test_normalise_handles_empty():
    assert len(_normalise(np.array([]))) == 0


# ── Content keying (the regression) ───────────────────────────────────────

def test_content_key_is_stable_across_object_identity(docs):
    """Two Documents with equal text must share a key despite different ids."""
    from langchain_core.documents import Document

    original = docs[0]
    copy = Document(page_content=original.page_content, metadata={"other": "metadata"})

    assert id(original) != id(copy)
    assert _content_key(original) == _content_key(copy)


def test_bm25_candidates_survive_scoring(docs):
    """Regression: BM25 hits must not be silently dropped.

    The vector store here returns only the third document, so a keyword-only
    match on the first document can only be returned if BM25 scores are
    actually blended in. Under the old id()-keyed lookup this returned
    nothing from BM25.
    """
    retriever = make_retriever(
        docs,
        scored_texts=[(docs[2].page_content, 0.9)],
        mode="hybrid",
    )

    results = retriever.retrieve("cross-border payments revenue", k=3)
    texts = [c.page_content for c in results]

    assert any("cross-border" in t for t in texts), (
        "BM25 keyword match was dropped — hybrid blending is not working"
    )


def test_alpha_zero_is_pure_bm25(docs):
    """alpha=0 must ignore semantic scores entirely."""
    retriever = make_retriever(
        docs,
        scored_texts=[(docs[2].page_content, 1.0)],
        mode="hybrid",
        alpha=0.0,
    )

    top = retriever.retrieve("headcount engineering sales", k=1)
    assert "headcount" in top[0].page_content


def test_alpha_one_is_pure_semantic(docs):
    """alpha=1 must rank purely on vector-store scores."""
    retriever = make_retriever(
        docs,
        scored_texts=[(docs[2].page_content, 1.0), (docs[0].page_content, 0.1)],
        mode="hybrid",
        alpha=1.0,
    )

    top = retriever.retrieve("headcount engineering sales", k=1)
    assert "satisfaction" in top[0].page_content


# ── Modes ─────────────────────────────────────────────────────────────────

def test_bm25_mode_does_not_query_vector_store(docs):
    store = FakeVectorStore([(d.page_content, 0.5) for d in docs])
    retriever = HybridRetriever(chunks=docs, vectorstore=store, mode="bm25")

    retriever.retrieve("revenue", k=2)
    assert store.calls == [], "bm25 mode should not call the vector store"


def test_semantic_mode_ranks_by_store_scores(docs):
    retriever = make_retriever(
        docs,
        scored_texts=[(docs[1].page_content, 0.95), (docs[0].page_content, 0.2)],
        mode="semantic",
    )

    top = retriever.retrieve("anything at all", k=1)
    assert "headcount" in top[0].page_content


def test_invalid_mode_rejected(docs):
    with pytest.raises(ValueError, match="Unknown retrieval mode"):
        HybridRetriever(chunks=docs, vectorstore=FakeVectorStore([]), mode="nonsense")


# ── Robustness ────────────────────────────────────────────────────────────

def test_empty_corpus_returns_nothing():
    retriever = HybridRetriever(chunks=[], vectorstore=FakeVectorStore([]), mode="hybrid")
    assert retriever.retrieve("anything") == []


def test_returns_at_most_k(docs):
    retriever = make_retriever(docs)
    assert len(retriever.retrieve("revenue", k=2)) == 2


def test_scores_are_returned_in_descending_order(docs):
    retriever = make_retriever(docs)
    scored = retriever.retrieve_with_scores("revenue growth", k=3)

    scores = [score for _, score in scored]
    assert scores == sorted(scores, reverse=True)


def test_metadata_preserved_from_original_chunks(docs):
    """Results must carry the original metadata, not the store's copy."""
    retriever = make_retriever(docs)
    results = retriever.retrieve("revenue", k=1)

    assert results[0].metadata.get("page") in {1, 2, 3}
    assert results[0].metadata.get("source") == "report.pdf"


def test_rerank_failure_degrades_gracefully(docs, monkeypatch):
    """A reranker crash must not take down the request."""
    def boom(_model):
        raise RuntimeError("model download failed")

    monkeypatch.setattr("src.retriever.get_reranker", boom)

    retriever = make_retriever(docs, mode="hybrid_rerank")
    results = retriever.retrieve("revenue", k=2)

    assert len(results) == 2, "should fall back to blended ordering"
