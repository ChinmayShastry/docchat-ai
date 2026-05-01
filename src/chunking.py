# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING
# Decides between semantic chunking (small docs) and fixed chunking (large docs).
#
# Semantic chunking:  uses embedding similarity to find natural topic boundaries.
#                     Better quality but makes extra API calls — only viable for
#                     small documents where the cost is acceptable.
#
# Fixed chunking:     splits by character count with overlap. Fast and cheap,
#                     suitable for large documents.

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from config import Config

def create_chunks(all_docs, embedding_model):
    total_chars = sum(len(doc.page_content) for doc in all_docs)

    if total_chars < Config.SMALL_DOC_THRESHOLD:
        print("🔬 Small document — using Semantic Chunking")
        semantic_splitter = SemanticChunker(
            embeddings=embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=85
        )

        # FIX: pass all documents in a single batched call instead of
        # looping (avoids N separate embedding API round-trips)
        texts     = [doc.page_content for doc in all_docs]
        metadatas = [doc.metadata     for doc in all_docs]

        try:
            chunks = semantic_splitter.create_documents(
                texts=texts,
                metadatas=metadatas
            )
        except Exception as e:
            print(f"[SemanticChunker failed, falling back to fixed] {e}")
            chunks = all_docs   # fall through to fixed chunking below

    else:
        print("⚡ Large document — using Fixed Chunking (faster)")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(all_docs)

    # Safety net: if chunking produced too few pieces, re-chunk with fixed splitter
    if len(chunks) < 3:
        print("⚠️ Too few chunks produced — re-chunking with fixed splitter")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks   = splitter.split_documents(all_docs)

    return chunks
