# CHUNKING STRATEGY
# Small documents get semantic chunking (embedding-aware split points, better
# retrieval quality, but one embedding call per sentence). Large documents get
# fixed-size chunking, which is far cheaper and fast enough to keep ingestion
# interactive. Every path degrades to the same fallback configuration.

from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_experimental.text_splitter import SemanticChunker
    SEMANTIC_AVAILABLE = True
except ImportError:  # langchain-experimental is an optional dependency
    SEMANTIC_AVAILABLE = False

from config import Config
from src.logging_setup import get_logger

log = get_logger(__name__)

# Below this many chunks the retriever has too little to work with, so we
# re-split at a smaller size rather than accept a degenerate index.
MIN_VIABLE_CHUNKS = 3


def _fallback_splitter():
    """The one splitter configuration every failure path falls back to."""
    return RecursiveCharacterTextSplitter(
        chunk_size=Config.FALLBACK_CHUNK_SIZE,
        chunk_overlap=Config.FALLBACK_CHUNK_OVERLAP,
    )


def create_chunks(all_docs, embedding_model):
    """Split documents into retrievable chunks, choosing a strategy by size."""
    if not all_docs:
        return []

    total_chars = sum(len(doc.page_content) for doc in all_docs)
    use_semantic = total_chars < Config.SMALL_DOC_THRESHOLD and SEMANTIC_AVAILABLE

    if use_semantic:
        log.info("Small document (%d chars) — using semantic chunking", total_chars)
        chunks = _semantic_chunks(all_docs, embedding_model)
    else:
        reason = "large document" if SEMANTIC_AVAILABLE else "semantic chunker unavailable"
        log.info("Using fixed chunking (%s, %d chars)", reason, total_chars)
        chunks = _fixed_chunks(all_docs)

    # ── Safety net ────────────────────────────────────────────────────────
    if len(chunks) < MIN_VIABLE_CHUNKS:
        log.warning(
            "Only %d chunk(s) produced — re-splitting at %d chars",
            len(chunks), Config.FALLBACK_CHUNK_SIZE,
        )
        chunks = _fallback_splitter().split_documents(all_docs)

    return chunks


def _semantic_chunks(all_docs, embedding_model):
    """Split on embedding-similarity breakpoints, falling back on failure."""
    splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,
    )

    try:
        return splitter.create_documents(
            texts=[doc.page_content for doc in all_docs],
            metadatas=[doc.metadata for doc in all_docs],
        )
    except Exception as exc:
        # Semantic chunking needs a working embedding endpoint; a network or
        # quota failure here should not fail ingestion outright.
        log.warning("Semantic chunking failed (%s) — falling back to fixed", exc)
        return _fallback_splitter().split_documents(all_docs)


def _fixed_chunks(all_docs):
    """Split at fixed size, preferring paragraph then sentence boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_documents(all_docs)
