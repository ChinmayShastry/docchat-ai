"""Tests for history normalisation, query routing, and context assembly."""

import pytest
from langchain_core.documents import Document

from config import Config
from src.chat import (
    build_context,
    classify_query,
    normalize_history,
    verify_groundedness,
)

# ── History normalisation ─────────────────────────────────────────────────

def test_normalize_empty_history():
    assert normalize_history([]) == []
    assert normalize_history(None) == []


def test_normalize_tuple_format():
    history = [("What is revenue?", "42 million."), ("And headcount?", "1,250.")]
    assert normalize_history(history) == history


def test_normalize_dict_format():
    history = [
        {"role": "user", "content": "What is revenue?"},
        {"role": "assistant", "content": "42 million."},
    ]
    assert normalize_history(history) == [("What is revenue?", "42 million.")]


def test_normalize_drops_unanswered_turn():
    """A trailing question with no answer yet must not produce a half-pair."""
    history = [
        {"role": "user", "content": "First?"},
        {"role": "assistant", "content": "Answer."},
        {"role": "user", "content": "Pending question"},
    ]
    assert normalize_history(history) == [("First?", "Answer.")]


def test_normalize_drops_none_assistant_content():
    history = [
        {"role": "user", "content": "Question?"},
        {"role": "assistant", "content": None},
    ]
    assert normalize_history(history) == []


def test_normalize_handles_list_content():
    """Some clients deliver content as a list of parts."""
    history = [
        {"role": "user", "content": ["Tell", "me", "more"]},
        {"role": "assistant", "content": "Sure."},
    ]
    assert normalize_history(history) == [("Tell me more", "Sure.")]


def test_normalize_skips_blank_turns():
    history = [("   ", "non-empty"), ("real question", "real answer")]
    assert normalize_history(history) == [("real question", "real answer")]


# ── Query classification ──────────────────────────────────────────────────

@pytest.mark.parametrize("question,expected", [
    ("Give me a summary of this", "summary"),
    ("Provide an overview", "summary"),
    ("Compare the two divisions", "comparison"),
    ("What is the difference between them?", "comparison"),
    ("How many employees are there?", "numeric"),
    ("What is the total revenue?", "numeric"),
    ("Who signed the agreement?", "factual"),
])
def test_classify_query(question, expected):
    assert classify_query(question) == expected


def test_classify_query_is_case_insensitive():
    assert classify_query("COMPARE these") == "comparison"


# ── Context assembly ──────────────────────────────────────────────────────

def test_build_context_labels_each_chunk(docs):
    context, sources = build_context(docs)

    assert "[Doc1: report.pdf, Page 1]" in context
    assert "[Doc2: report.pdf, Page 2]" in context
    assert sources[0] == "report.pdf (Page 1)"


def test_build_context_truncates_at_doc_boundary(monkeypatch):
    """Truncation must not slice through a chunk and mis-attribute its text."""
    monkeypatch.setattr(Config, "MAX_CONTEXT_CHARS", 120)

    long_docs = [
        Document(page_content="A" * 100, metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="B" * 100, metadata={"source": "b.pdf", "page": 2}),
    ]

    context, _ = build_context(long_docs)

    assert len(context) <= 120
    assert "B" * 100 not in context, "second chunk should be dropped whole"


def test_build_context_empty():
    context, sources = build_context([])
    assert context == ""
    assert sources == []


# ── Groundedness verification ─────────────────────────────────────────────

def test_verify_groundedness_accepts_ok(fake_client):
    client = fake_client(["OK"])
    is_grounded, verdict = verify_groundedness("context", "answer", client)

    assert is_grounded is True
    assert verdict == "OK"


def test_verify_groundedness_accepts_ok_with_trailing_text(fake_client):
    client = fake_client(["OK, every claim is supported by the context."])
    is_grounded, _ = verify_groundedness("context", "answer", client)
    assert is_grounded is True


def test_verify_groundedness_flags_unsupported(fake_client):
    client = fake_client([
        "The claim about a merger is not supported anywhere in the context."
    ])
    is_grounded, verdict = verify_groundedness("context", "answer", client)

    assert is_grounded is False
    assert "merger" in verdict


def test_verify_groundedness_ignores_late_ok(fake_client):
    """'ok' appearing deep in a rejection must not be read as approval."""
    client = fake_client([
        "The revenue figure is unsupported and the date is wrong. "
        "Everything else looks ok."
    ])
    is_grounded, _ = verify_groundedness("context", "answer", client)
    assert is_grounded is False
