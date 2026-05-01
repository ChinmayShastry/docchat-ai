# HYBRID RETRIEVER
# Combines BM25 (keyword matching) and semantic (embedding) search, then
# reranks the merged candidate pool with a CrossEncoder for precision.
#
# Flow:
#   1. BM25 scores all chunks → top-50 by keyword relevance
#   2. Semantic search retrieves top-50 by embedding similarity
#   3. Both lists are merged and deduplicated
#   4. Low-confidence semantic results are filtered out
#   5. CrossEncoder reranks surviving candidates → top-k returned

import re
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from config import Config

class HybridRetriever:
    def __init__(self, chunks, vectorstore, alpha=0.5):
        self.chunks      = chunks
        self.vectorstore = vectorstore
        self.alpha       = alpha

        # Build BM25 index over all chunk texts
        tokenized  = [self._tokenize(c.page_content) for c in chunks]
        self.bm25  = BM25Okapi(tokenized)

        # CrossEncoder loaded once per retriever instance (not at import time)
        self.reranker = CrossEncoder(Config.RERANKER_MODEL)

    def _tokenize(self, text):
        """Lowercase, strip punctuation, split on whitespace."""
        return re.sub(r'[^a-z0-9\s]', '', text.lower()).split()

    def retrieve(self, query, k=4):
        """Return the top-k most relevant chunks for a given query."""
        tokenized_query = self._tokenize(query)

        # ── Step 1: BM25 keyword scores (normalised 0–1) ──────────────────
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_scores = bm25_scores / (bm25_scores.max() + 1e-6)

        # ── Step 2: Semantic similarity scores ────────────────────────────
        k_sem       = min(50, len(self.chunks))
        sem_results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k_sem)
        sem_score_map = {doc.page_content: score for doc, score in sem_results}

        # ── Step 3: Gather top-50 from each source ────────────────────────
        bm25_top_idx = bm25_scores.argsort()[-50:][::-1]
        bm25_top     = [self.chunks[i] for i in bm25_top_idx]
        sem_top      = [doc for doc, _ in sem_results]

        # Merge and deduplicate by object identity
        candidate_chunks = list({id(c): c for c in (bm25_top + sem_top)}.values())

        # ── Step 4: Filter weak semantic candidates ───────────────────────
        # Keep chunks that cleared the semantic confidence threshold.
        filtered = [
            c for c in candidate_chunks
            if sem_score_map.get(c.page_content, 0) > 0.2
        ]

        # Fallback: if filtering was too aggressive, keep top-20 candidates
        if len(filtered) < 10:
            filtered = candidate_chunks[:20]

        candidate_chunks = filtered[:40]

        # Hard fallback: if somehow nothing survived, return first k chunks
        if not candidate_chunks:
            return self.chunks[:k]

        # ── Step 5: CrossEncoder reranking ────────────────────────────────
        pairs    = [(query, chunk.page_content) for chunk in candidate_chunks]
        scores   = self.reranker.predict(pairs, batch_size=16)
        reranked = sorted(zip(scores, candidate_chunks), key=lambda x: x[0], reverse=True)

        return [chunk for _, chunk in reranked[:k]]
