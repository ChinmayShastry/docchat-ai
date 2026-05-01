# CONFIGURATION
# Central place to tune all model names, thresholds, and retrieval settings.
# Change values here — no need to hunt through the rest of the code.

class Config:
    # OpenAI Models
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt-4o-mini"

    # Fixed chunking settings (used when total document size > SMALL_DOC_THRESHOLD)
    CHUNK_SIZE = 600
    CHUNK_OVERLAP = 100

    # Documents under this character count get semantic chunking (slower but smarter).
    # Larger documents fall back to fixed chunking for speed.
    SMALL_DOC_THRESHOLD = 50_000

    # Number of chunks returned per retrieval call
    RETRIEVAL_K = 4

    # Weight balance between BM25 (keyword) and semantic search.
    # 0.0 = pure BM25, 1.0 = pure semantic, 0.5 = equal blend 
    HYBRID_ALPHA = 0.5

    # Cross-encoder model used for reranking retrieved candidates
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Hard cap on context characters sent to the LLM to avoid token overflow
    MAX_CONTEXT_CHARS = 12_000 
