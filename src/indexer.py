import uuid
import tempfile

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

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

    print(f"Total chunks before filter: {len(all_chunks)}")

    # Filter small chunks
    all_chunks = [
        c for c in all_chunks if len(c.page_content.strip()) > 10
    ]

    print(f"Total chunks after filter: {len(all_chunks)}")

    # Safety check
    if not all_chunks:
    print("⚠️ No chunks after filtering → using raw documents")

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    all_chunks = splitter.split_documents(all_docs)
    
    # Add metadata
    for chunk in all_chunks:
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
        
    print("Chunks after filter:", len(all_chunks))
    print(f"✅ Final chunks stored: {len(all_chunks)}")

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
