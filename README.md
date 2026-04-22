# 📄 DocChat AI — Advanced RAG System

**Visit**: https://huggingface.co/spaces/ChinmayShastry9/docchat-ai

An advanced multi-document conversational AI system that allows users to upload documents and interact with them using natural language.

---

## 🚀 Features

- 📂 Multi-format support (PDF, DOCX, TXT, CSV, XLSX)
- 🧠 RAG-based architecture
- 🔍 Hybrid retrieval (BM25 + Semantic Search)
- ⚡ CrossEncoder reranking for improved relevance
- 🧩 Adaptive chunking (semantic + fixed)
- 📊 Map-Reduce summarization for large documents
- 💬 Streaming responses with citations
- 🧠 Conversation memory compression
- ✅ Groundedness verification (anti-hallucination)

---

## 🧠 Architecture

1. Document ingestion  
2. Adaptive chunking  
3. Embedding generation (OpenAI)  
4. Vector storage (ChromaDB)  
5. Hybrid retrieval (BM25 + embeddings)  
6. CrossEncoder reranking  
7. LLM response generation  
8. Groundedness verification  

---

## 🛠 Tech Stack

- OpenAI API  
- LangChain  
- ChromaDB  
- BM25  
- Sentence Transformers  
- Gradio  

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
python app.py
