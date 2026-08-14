# CONFIGURATION
# Central place to tune all model names, thresholds, and retrieval settings.
# Change values here — no need to hunt through the rest of the code.
#
# Any value can also be overridden by an environment variable of the same name
# prefixed with DOCCHAT_ (e.g. DOCCHAT_RETRIEVAL_K=8). The evaluation harness
# relies on this to sweep retrieval strategies without editing source.

import os


def _env_str(name, default):
    return os.getenv(f"DOCCHAT_{name}", default)


def _env_int(name, default):
    raw = os.getenv(f"DOCCHAT_{name}")
    return int(raw) if raw else default


def _env_float(name, default):
    raw = os.getenv(f"DOCCHAT_{name}")
    return float(raw) if raw else default


class Config:
    # ── Models ────────────────────────────────────────────────────────────
    EMBEDDING_MODEL = _env_str("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL = _env_str("LLM_MODEL", "gpt-4o-mini")

    # ── Chunking ──────────────────────────────────────────────────────────
    # Fixed chunking settings (used when total document size > SMALL_DOC_THRESHOLD)
    CHUNK_SIZE = _env_int("CHUNK_SIZE", 600)
    CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 100)

    # Documents under this character count get semantic chunking (slower but
    # smarter). Larger documents fall back to fixed chunking for speed.
    SMALL_DOC_THRESHOLD = _env_int("SMALL_DOC_THRESHOLD", 50_000)

    # Chunk sizing used by every fallback path, so a failure in semantic
    # chunking degrades to one predictable configuration rather than three.
    FALLBACK_CHUNK_SIZE = _env_int("FALLBACK_CHUNK_SIZE", 500)
    FALLBACK_CHUNK_OVERLAP = _env_int("FALLBACK_CHUNK_OVERLAP", 100)

    # ── Retrieval ─────────────────────────────────────────────────────────
    # Number of chunks returned per retrieval call
    RETRIEVAL_K = _env_int("RETRIEVAL_K", 4)

    # Strategy used by HybridRetriever.retrieve(). One of:
    #   "bm25"          — keyword scoring only
    #   "semantic"      — embedding similarity only
    #   "hybrid"        — alpha-weighted blend of both
    #   "hybrid_rerank" — blend, then cross-encoder rerank (production default)
    RETRIEVAL_MODE = _env_str("RETRIEVAL_MODE", "hybrid_rerank")

    # Weight balance between BM25 (keyword) and semantic search.
    # 0.0 = pure BM25, 1.0 = pure semantic, 0.5 = equal blend
    HYBRID_ALPHA = _env_float("HYBRID_ALPHA", 0.5)

    # How many candidates each retrieval arm contributes before merging, and
    # how many survive into the (expensive) cross-encoder rerank stage.
    BM25_CANDIDATES = _env_int("BM25_CANDIDATES", 50)
    SEMANTIC_CANDIDATES = _env_int("SEMANTIC_CANDIDATES", 50)
    RERANK_CANDIDATES = _env_int("RERANK_CANDIDATES", 40)

    # Cross-encoder model used for reranking retrieved candidates
    RERANKER_MODEL = _env_str("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Minimum blended score a chunk must reach to survive filtering. Chunks
    # below this are dropped unless doing so would leave too few candidates.
    SEMANTIC_THRESHOLD = _env_float("SEMANTIC_THRESHOLD", 0.2)

    # Never filter the candidate pool below this many chunks — a too-aggressive
    # threshold on an unusual query should degrade recall, not empty the pool.
    MIN_CANDIDATES = _env_int("MIN_CANDIDATES", 5)

    # ── Generation ────────────────────────────────────────────────────────
    # Hard cap on context characters sent to the LLM to avoid token overflow
    MAX_CONTEXT_CHARS = _env_int("MAX_CONTEXT_CHARS", 12_000)

    # Turns of raw history kept before switching to a rolling summary
    HISTORY_SUMMARY_THRESHOLD = _env_int("HISTORY_SUMMARY_THRESHOLD", 5)

    # Run the post-answer groundedness check. Costs one extra LLM call per
    # question, so the eval harness disables it when measuring latency.
    ENABLE_GROUNDEDNESS_CHECK = _env_str("ENABLE_GROUNDEDNESS_CHECK", "1") == "1"

    # ── Ingestion limits ──────────────────────────────────────────────────
    # Reject oversized uploads before they reach the embedding API.
    MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 25)

    # ── Caching ───────────────────────────────────────────────────────────
    # Directory for persisted Chroma indexes, keyed by document content hash.
    CACHE_DIR = _env_str("CACHE_DIR", ".docchat_cache")
    ENABLE_INDEX_CACHE = _env_str("ENABLE_INDEX_CACHE", "1") == "1"
