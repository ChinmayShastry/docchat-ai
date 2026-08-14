"""End-to-end indexing tests against a real (in-memory) Chroma store.

These exist because indexer.py is the one module that talks to the vector
store directly, and a Chroma major-version bump can break it in ways the
mocked retriever tests cannot see — `.get()` semantics, constructor keywords,
persistence behaviour. Embeddings are faked so the suite stays offline and
free, but Chroma itself is real.
"""

import hashlib

import pytest
from langchain_core.documents import Document

# Must precede the src.indexer import: that module imports both of these at
# load time, so a guard placed after it would never run.
pytest.importorskip("langchain_chroma")
pytest.importorskip("langchain_openai")

from src import indexer  # noqa: E402
from src.indexer import NoUsableTextError, _index_fingerprint, build_index  # noqa: E402

EMBED_DIM = 16


class FakeEmbeddings:
    """Deterministic hash-based embeddings — no network, stable across runs.

    Similar text does not produce similar vectors here, so these tests assert
    plumbing (documents in, documents out, metadata preserved) rather than
    retrieval quality, which the retriever tests cover.
    """

    def _vector(self, text):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(EMBED_DIM)]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def patched_embeddings(monkeypatch):
    monkeypatch.setattr(indexer, "OpenAIEmbeddings", lambda **kwargs: FakeEmbeddings())


def make_docs(n=6):
    return [
        Document(
            page_content=(
                f"Section {i}. Quarterly performance for division {i} showed "
                f"revenue of {i * 100} million euros against a target of "
                f"{i * 110} million, with headcount of {i * 40} people."
            ),
            metadata={"source": "report.pdf", "page": i + 1},
        )
        for i in range(n)
    ]


# ── Fingerprinting ────────────────────────────────────────────────────────

def test_fingerprint_is_stable_for_same_content():
    docs = make_docs()
    assert _index_fingerprint(docs) == _index_fingerprint(make_docs())


def test_fingerprint_changes_with_content():
    docs = make_docs()
    altered = make_docs()
    altered[0].metadata["source"] = "different.pdf"
    assert _index_fingerprint(docs) != _index_fingerprint(altered)


def test_fingerprint_changes_with_chunking_settings(monkeypatch):
    """A settings change must invalidate the cache, not silently reuse it."""
    from config import Config

    docs = make_docs()
    before = _index_fingerprint(docs)

    monkeypatch.setattr(Config, "CHUNK_SIZE", Config.CHUNK_SIZE + 137)
    assert _index_fingerprint(docs) != before


# ── Index building ────────────────────────────────────────────────────────

def test_build_index_returns_working_retriever(patched_embeddings, tmp_path, monkeypatch):
    from config import Config

    monkeypatch.setattr(Config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(Config, "SMALL_DOC_THRESHOLD", 0)  # force fixed chunking

    retriever, n_chunks = build_index(make_docs(), api_key="not-used")

    assert n_chunks > 0
    results = retriever.retrieve("revenue target division", k=3)

    assert results, "a built index must return something for a matching query"
    assert all(isinstance(d, Document) for d in results)
    assert all(d.metadata.get("source") == "report.pdf" for d in results)


def test_build_index_assigns_chunk_ids(patched_embeddings, tmp_path, monkeypatch):
    from config import Config

    monkeypatch.setattr(Config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(Config, "SMALL_DOC_THRESHOLD", 0)

    retriever, _ = build_index(make_docs(), api_key="not-used")
    assert all("chunk_id" in c.metadata for c in retriever.chunks)


def test_build_index_rejects_empty_input():
    with pytest.raises(NoUsableTextError):
        build_index([], api_key="not-used")


def test_build_index_rejects_whitespace_only(patched_embeddings, tmp_path, monkeypatch):
    from config import Config

    monkeypatch.setattr(Config, "CACHE_DIR", str(tmp_path / "cache"))

    blank = [Document(page_content="   \n  ", metadata={"source": "x.pdf", "page": 1})]
    with pytest.raises(NoUsableTextError):
        build_index(blank, api_key="not-used")


# ── Cache round-trip ──────────────────────────────────────────────────────

def test_second_build_reuses_cached_index(patched_embeddings, tmp_path, monkeypatch):
    """Re-indexing identical content must reload rather than re-embed.

    This is the whole point of the content-hash cache, and it exercises the
    Chroma `.get()` reload path — the part most likely to break on a Chroma
    major-version bump.
    """
    from config import Config

    monkeypatch.setattr(Config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(Config, "SMALL_DOC_THRESHOLD", 0)

    docs = make_docs()
    first_retriever, first_count = build_index(docs, api_key="not-used")

    calls = {"n": 0}
    real_from_documents = indexer.Chroma.from_documents

    def counting_from_documents(*args, **kwargs):
        calls["n"] += 1
        return real_from_documents(*args, **kwargs)

    monkeypatch.setattr(indexer.Chroma, "from_documents", counting_from_documents)

    second_retriever, second_count = build_index(docs, api_key="not-used")

    assert calls["n"] == 0, "cached index should not be rebuilt"
    assert second_count == first_count
    assert second_retriever.retrieve("revenue", k=2)


def test_cache_can_be_disabled(patched_embeddings, tmp_path, monkeypatch):
    from config import Config

    monkeypatch.setattr(Config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(Config, "SMALL_DOC_THRESHOLD", 0)

    docs = make_docs()
    build_index(docs, api_key="not-used", use_cache=False)
    retriever, count = build_index(docs, api_key="not-used", use_cache=False)

    assert count > 0
    assert retriever.retrieve("headcount", k=1)
