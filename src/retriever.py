import re
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from functools import lru_cache
from config import Config


# Global cache 
@lru_cache()
def get_reranker():
    return CrossEncoder(Config.RERANKER_MODEL)


class HybridRetriever:
    def __init__(self, chunks, vectorstore, alpha=0.5):
        self.chunks = chunks
        self.vectorstore = vectorstore
        self.alpha = alpha

        # BM25 index
        tokenized = [self._tokenize(c.page_content) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

        # Cached reranker
        self.reranker = get_reranker()

    def _tokenize(self, text):
        return re.sub(r'[^a-z0-9\s]', '', text.lower()).split()

    def retrieve(self, query, k=4):
        tokenized_query = self._tokenize(query)

        # ── BM25 scores ─────────────────────────
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_scores = bm25_scores / (bm25_scores.max() + 1e-6)

        # ── Semantic scores ─────────────────────
        k_sem = min(50, len(self.chunks))
        sem_results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k_sem)

        # ✅ FIX: stable mapping
        sem_score_map = {id(doc): score for doc, score in sem_results}

        # ── Top candidates ──────────────────────
        bm25_top_idx = bm25_scores.argsort()[-50:][::-1]
        bm25_top = [self.chunks[i] for i in bm25_top_idx]
        sem_top = [doc for doc, _ in sem_results]

        # Merge + deduplicate
        candidate_chunks = list({id(c): c for c in (bm25_top + sem_top)}.values())

        # ── Filtering ───────────────────────────
        filtered = [
            c for c in candidate_chunks
            if sem_score_map.get(id(c), 0) > Config.SEMANTIC_THRESHOLD    
        ]

        # Fallback if too aggressive
        if len(filtered) < 5:
            filtered = candidate_chunks[:20]

        candidate_chunks = filtered[:40]

        # Hard fallback
        if not candidate_chunks:
            return self.chunks[:k]

        # ── Reranking ───────────────────────────
        pairs = [(query, chunk.page_content) for chunk in candidate_chunks]
        scores = self.reranker.predict(pairs, batch_size=16)

        reranked = sorted(zip(scores, candidate_chunks), key=lambda x: x[0], reverse=True)

        return [chunk for _, chunk in reranked[:k]]
