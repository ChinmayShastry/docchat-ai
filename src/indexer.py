# ─────────────────────────────────────────────────────────────────────────────
# INDEX BUILDER
# Embeds all chunks into ChromaDB and wires up the HybridRetriever.
# Called once per document upload session.
# ─────────────────────────────────────────────────────────────────────────────

import uuid
import tempfile

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from src.chunking import create_chunks
from src.retriever import HybridRetriever
from config import Config

def build_index(all_docs, api_key):
    embedding_model = OpenAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        openai_api_key=api_key
    )

    total_pages = len(all_docs)
    total_chars = sum(len(doc.page_content) for doc in all_docs)
    print(f"📄 Pages: {total_pages} | Characters: {total_chars:,}")

    all_chunks = create_chunks(all_docs, embedding_model)

    # Drop near-empty chunks that would add noise to retrieval
    all_chunks = [c for c in all_chunks if len(c.page_content.strip()) > 50]

    # Assign a unique ID to each chunk (useful for dedup / future caching)
    for chunk in all_chunks:
        chunk.metadata["chunk_id"] = str(uuid.uuid4())

    print(f"✅ Chunks created: {len(all_chunks)}")

    # Use a temp directory so each session gets an isolated vector store
    persist_dir = tempfile.mkdtemp()
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_model,
        persist_directory=persist_dir
    )

    retriever = HybridRetriever(
        chunks=all_chunks,
        vectorstore=vectorstore,
        alpha=Config.HYBRID_ALPHA
    )

    return retriever, len(all_chunks)

