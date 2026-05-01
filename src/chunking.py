from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_experimental.text_splitter import SemanticChunker
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

from config import Config


def create_chunks(all_docs, embedding_model):
    total_chars = sum(len(doc.page_content) for doc in all_docs)

    # ── Semantic Chunking ─────────────────────────
    if total_chars < Config.SMALL_DOC_THRESHOLD and SEMANTIC_AVAILABLE:
        print("🔬 Small document — using Semantic Chunking")

        semantic_splitter = SemanticChunker(
            embeddings=embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=85
        )

        texts = [doc.page_content for doc in all_docs]
        metadatas = [doc.metadata for doc in all_docs]

        try:
            chunks = semantic_splitter.create_documents(
                texts=texts,
                metadatas=metadatas
            )
        except Exception as e:
            print(f"[SemanticChunker failed → fallback] {e}")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )
            chunks = splitter.split_documents(all_docs)

    # ── Fixed Chunking ────────────────────────────
    else:
        print("⚡ Using Fixed Chunking")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " "]
        )

        chunks = splitter.split_documents(all_docs)

    # ── Safety fallback ───────────────────────────
    if len(chunks) < 3:
        print("⚠️ Re-chunking fallback triggered")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(all_docs)

    return chunks
