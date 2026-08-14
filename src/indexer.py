# INDEX BUILDER
# Turns extracted documents into a queryable HybridRetriever:
#   chunk → embed → persist to Chroma → wrap in BM25 + semantic retrieval.
#
# Indexes are cached on disk under a key derived from document content *and*
# the settings that affect chunking/embedding. Re-uploading the same file is
# then free instead of re-paying the embedding cost, and changing any relevant
# setting produces a different key rather than silently reusing a stale index.

import hashlib
import json
import os
import uuid

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config
from src.chunking import create_chunks
from src.logging_setup import get_logger
from src.retriever import HybridRetriever

log = get_logger(__name__)

COLLECTION_NAME = "docchat"

# Chunks shorter than this are punctuation fragments or page furniture; they
# dilute BM25 statistics and never contain a usable answer.
MIN_CHUNK_CHARS = 5


class NoUsableTextError(ValueError):
    """Raised when a document yields no indexable text."""


def _index_fingerprint(all_docs):
    """Content + settings hash identifying this exact index."""
    digest = hashlib.sha256()

    for doc in all_docs:
        digest.update(doc.page_content.encode("utf-8", "replace"))
        digest.update(str(doc.metadata.get("source", "")).encode("utf-8", "replace"))

    # Settings that change the resulting vectors or chunk boundaries must be
    # part of the key, or a config change would reuse an incompatible index.
    settings = json.dumps({
        "embedding_model": Config.EMBEDDING_MODEL,
        "chunk_size": Config.CHUNK_SIZE,
        "chunk_overlap": Config.CHUNK_OVERLAP,
        "small_doc_threshold": Config.SMALL_DOC_THRESHOLD,
    }, sort_keys=True)
    digest.update(settings.encode("utf-8"))

    return digest.hexdigest()[:16]


def _clean(chunks):
    return [c for c in chunks if c.page_content and len(c.page_content.strip()) > MIN_CHUNK_CHARS]


def _load_cached(persist_dir, embedding_model):
    """Rebuild a retriever from a persisted Chroma index, or return None."""
    try:
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=persist_dir,
            embedding_function=embedding_model,
        )

        stored = vectorstore.get(include=["documents", "metadatas"])
        texts = stored.get("documents") or []
        metadatas = stored.get("metadatas") or []

        if not texts:
            return None

        # BM25 operates over in-memory chunks, so reconstruct them from what
        # Chroma persisted rather than storing a second copy on disk.
        chunks = [
            Document(page_content=text, metadata=meta or {})
            # strict=False: Chroma can omit metadata entries for some rows.
            for text, meta in zip(texts, metadatas, strict=False)
        ]

        log.info("Reusing cached index (%d chunks) from %s", len(chunks), persist_dir)
        return HybridRetriever(chunks=chunks, vectorstore=vectorstore), len(chunks)

    except Exception as exc:
        log.warning("Could not load cached index at %s: %s", persist_dir, exc)
        return None


def build_index(all_docs, api_key, use_cache=None):
    """Chunk, embed and index `all_docs`, returning (retriever, chunk_count)."""
    if not all_docs:
        raise NoUsableTextError("No documents were provided to index.")

    use_cache = Config.ENABLE_INDEX_CACHE if use_cache is None else use_cache

    embedding_model = OpenAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        openai_api_key=api_key,
    )

    total_chars = sum(len(doc.page_content) for doc in all_docs)
    log.info("Indexing %d page(s), %d characters", len(all_docs), total_chars)

    # ── Cache lookup ──────────────────────────────────────────────────────
    fingerprint = _index_fingerprint(all_docs)
    persist_dir = os.path.join(Config.CACHE_DIR, fingerprint)

    if use_cache and os.path.isdir(persist_dir):
        cached = _load_cached(persist_dir, embedding_model)
        if cached:
            return cached

    # ── Chunk ─────────────────────────────────────────────────────────────
    all_chunks = _clean(create_chunks(all_docs, embedding_model))
    log.info("Produced %d usable chunk(s)", len(all_chunks))

    if not all_chunks:
        log.warning("No usable chunks after cleaning — retrying with fixed splitter")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.FALLBACK_CHUNK_SIZE,
            chunk_overlap=Config.FALLBACK_CHUNK_OVERLAP,
        )
        all_chunks = _clean(splitter.split_documents(all_docs))

        if not all_chunks:
            raise NoUsableTextError(
                "No readable text found. The document may be scanned, empty, or "
                "image-only — try a text-based PDF."
            )

    for chunk in all_chunks:
        chunk.metadata["chunk_id"] = str(uuid.uuid4())

    # ── Embed + persist ───────────────────────────────────────────────────
    os.makedirs(persist_dir, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir if use_cache else None,
    )

    log.info("Indexed %d chunk(s)%s", len(all_chunks), " (cached)" if use_cache else "")

    retriever = HybridRetriever(chunks=all_chunks, vectorstore=vectorstore)
    return retriever, len(all_chunks)
