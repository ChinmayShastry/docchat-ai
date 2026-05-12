import uuid
import tempfile

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.chunking import create_chunks
from src.retriever import HybridRetriever
from config import Config


def build_index(all_docs, api_key):

    print(f"Total docs: {len(all_docs)}")

    embedding_model = OpenAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        openai_api_key=api_key
    )

    total_pages = len(all_docs)
    total_chars = sum(len(doc.page_content) for doc in all_docs)
    print(f"📄 Pages: {total_pages} | Characters: {total_chars:,}")

    # ── Chunking ─────────────────────────
    all_chunks = create_chunks(all_docs, embedding_model)    

    if len(all_chunks) < 3:
        print("⚠️ Very low chunk count — results may be weak")
    
    print(f"Total chunks before filter: {len(all_chunks)}")

    # ── Clean chunks ─────────────────────
    all_chunks = [
        c for c in all_chunks
        if c.page_content and len(c.page_content.strip()) > 5
    ]

    print(f"Clean chunks after filter: {len(all_chunks)}")

    # ── Fallback if empty ─────────────────
    if not all_chunks:
        print("⚠️ No valid chunks → applying fallback chunking")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        all_chunks = splitter.split_documents(all_docs)

        # 🔥 Clean again after fallback
        all_chunks = [
            c for c in all_chunks
            if c.page_content and len(c.page_content.strip()) > 5
        ]

        print(f"Chunks after fallback: {len(all_chunks)}")

        if not all_chunks:
            raise ValueError(
                "❌ No usable text found. Document may be scanned or empty."
            )

    # ── Add metadata ─────────────────────
    for chunk in all_chunks:
        chunk.metadata["chunk_id"] = str(uuid.uuid4())

    print(f"✅ Final chunks stored: {len(all_chunks)}")

    # Debug preview (optional but useful)
    for i, c in enumerate(all_chunks[:2]):
        print(f"Chunk {i} preview:", c.page_content[:100])

    # ── Vector DB ─────────────────────────
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
