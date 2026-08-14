"""Shared fixtures.

Everything here is offline: no OpenAI calls, no model downloads. The suite is
meant to run in CI on every push, so it must be free and deterministic.
"""

import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

# Make the project root importable so `config` and `src.*` resolve the same way
# they do when Streamlit runs app.py from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeVectorStore:
    """Minimal stand-in for Chroma.

    Deliberately returns *newly constructed* Document objects on every call,
    exactly as a real vector store does. This is what broke the original
    retriever: score lookups keyed on id() never matched the in-memory chunks.
    """

    def __init__(self, scored_texts):
        # scored_texts: list of (text, score), highest relevance first
        self.scored_texts = scored_texts
        self.calls = []

    def similarity_search_with_relevance_scores(self, query, k=4):
        self.calls.append((query, k))
        return [
            (Document(page_content=text, metadata={"source": "fake.pdf", "page": 1}), score)
            for text, score in self.scored_texts[:k]
        ]


class FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self._responses.pop(0) if self._responses else "stub response"

        if kwargs.get("stream"):
            return _fake_stream(content)
        return _FakeResponse(content)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeStreamChoice:
    def __init__(self, content):
        self.delta = _FakeDelta(content)


class _FakeStreamChunk:
    def __init__(self, content):
        self.choices = [_FakeStreamChoice(content)]


def _fake_stream(content):
    """Yield the content a few characters at a time, like a real stream."""
    step = max(1, len(content) // 4)
    for i in range(0, len(content), step):
        yield _FakeStreamChunk(content[i:i + step])


class FakeClient:
    """Stands in for openai.OpenAI with scripted responses."""

    def __init__(self, responses=None):
        self.chat = type("chat", (), {})()
        self.chat.completions = FakeCompletions(responses or [])


@pytest.fixture
def fake_client():
    def _make(responses=None):
        return FakeClient(responses)
    return _make


@pytest.fixture
def docs():
    """Three short documents with distinct vocabulary."""
    return [
        Document(
            page_content=(
                "The quarterly revenue for the payments division reached 42 million "
                "dollars, driven by growth in cross-border transactions."
            ),
            metadata={"source": "report.pdf", "page": 1},
        ),
        Document(
            page_content=(
                "Employee headcount grew to 1,250 people across engineering, sales "
                "and support functions during the same period."
            ),
            metadata={"source": "report.pdf", "page": 2},
        ),
        Document(
            page_content=(
                "Customer satisfaction scores averaged 4.6 out of 5 in the annual "
                "survey, an improvement over the prior year."
            ),
            metadata={"source": "report.pdf", "page": 3},
        ),
    ]
