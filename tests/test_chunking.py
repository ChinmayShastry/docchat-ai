"""Tests for chunking strategy selection and its fallback behaviour."""

from langchain_core.documents import Document

from config import Config
from src import chunking
from src.chunking import MIN_VIABLE_CHUNKS, create_chunks


class DummyEmbeddings:
    """Never actually called — fixed chunking ignores the embedding model."""

    def embed_documents(self, texts):
        raise AssertionError("fixed chunking must not embed")


def make_docs(char_count, source="big.pdf"):
    # Sentence-shaped text so the recursive splitter has real boundaries.
    sentence = "This is a sentence about quarterly financial performance. "
    body = (sentence * (char_count // len(sentence) + 1))[:char_count]
    return [Document(page_content=body, metadata={"source": source, "page": 1})]


def test_empty_input_returns_empty():
    assert create_chunks([], DummyEmbeddings()) == []


def test_large_document_uses_fixed_chunking(monkeypatch):
    """Above the threshold we must not pay for semantic chunking."""
    monkeypatch.setattr(Config, "SMALL_DOC_THRESHOLD", 1_000)

    chunks = create_chunks(make_docs(5_000), DummyEmbeddings())

    assert len(chunks) > 1
    assert all(isinstance(c, Document) for c in chunks)


def test_fixed_chunking_respects_configured_size(monkeypatch):
    monkeypatch.setattr(Config, "SMALL_DOC_THRESHOLD", 100)
    monkeypatch.setattr(Config, "CHUNK_SIZE", 200)
    monkeypatch.setattr(Config, "CHUNK_OVERLAP", 20)

    chunks = create_chunks(make_docs(2_000), DummyEmbeddings())

    # Splitters may overshoot slightly on separator boundaries.
    assert max(len(c.page_content) for c in chunks) <= 260


def test_semantic_failure_falls_back_to_fixed(monkeypatch):
    """An embedding outage during chunking must not fail ingestion."""
    monkeypatch.setattr(Config, "SMALL_DOC_THRESHOLD", 10_000)
    monkeypatch.setattr(chunking, "SEMANTIC_AVAILABLE", True)

    class ExplodingChunker:
        def __init__(self, **kwargs):
            pass

        def create_documents(self, texts, metadatas):
            raise RuntimeError("embedding endpoint unavailable")

    monkeypatch.setattr(chunking, "SemanticChunker", ExplodingChunker, raising=False)

    chunks = create_chunks(make_docs(2_000), DummyEmbeddings())

    assert len(chunks) >= MIN_VIABLE_CHUNKS
    assert all(c.page_content.strip() for c in chunks)


def test_tiny_document_triggers_resplit(monkeypatch):
    """Too few chunks means a degenerate index — re-split smaller instead."""
    monkeypatch.setattr(Config, "SMALL_DOC_THRESHOLD", 1)
    monkeypatch.setattr(Config, "CHUNK_SIZE", 100_000)
    monkeypatch.setattr(Config, "FALLBACK_CHUNK_SIZE", 100)
    monkeypatch.setattr(Config, "FALLBACK_CHUNK_OVERLAP", 10)

    chunks = create_chunks(make_docs(1_200), DummyEmbeddings())

    assert len(chunks) >= MIN_VIABLE_CHUNKS


def test_metadata_survives_chunking(monkeypatch):
    monkeypatch.setattr(Config, "SMALL_DOC_THRESHOLD", 100)

    chunks = create_chunks(make_docs(2_000, source="quarterly.pdf"), DummyEmbeddings())

    assert all(c.metadata["source"] == "quarterly.pdf" for c in chunks)
