# 📄 DocChat AI — Advanced RAG System

> **Upload documents → Get instant summaries → Chat with source-backed answers**

---

## 🚀 Live Applications

- **🌐 [Streamlit App](https://docchat-ai.streamlit.app)**
- **🤗 [Hugging Face Space](https://huggingface.co/spaces/ChinmayShastry9/docchat-ai)**

---

## 🧠 Overview

DocChat AI is a **production-ready Retrieval-Augmented Generation (RAG) system** that enables users to:
- Upload documents
- Generate structured summaries
- Ask questions conversationally
- Receive answers grounded in document content

This project demonstrates **end-to-end AI system design**, including:
- Document ingestion
- Hybrid retrieval
- LLM-based reasoning
- Real-time streaming responses

---

## ✨ Features

### 📂 Multi-Format Document Support
- PDF, DOCX, TXT, CSV, XLSX
- Multi-document upload
- Robust extraction pipeline with fallbacks:
  - PyPDFLoader
  - PyMuPDF
  - pdfplumber

### 🧠 Intelligent Processing
#### 🔹 Smart Chunking
- Semantic chunking (small documents)
- Fixed chunking (large documents)
- Automatic fallback for reliability

#### 🔹 Hybrid Retrieval
- BM25 (keyword search)
- Semantic embeddings
- Cross-encoder reranking
  **→ Ensures highly relevant results**

### 💬 Conversational AI
- Multi-turn chat
- Context-aware responses
- Query classification:
  - Summary
  - Comparison
  - Numeric queries
  - Factual queries

### 📄 Source Citations
Every answer includes:
- Document name
- Page reference
- Groundedness verification to reduce hallucinations

### ⚡ Real-Time Streaming
- Token-by-token response streaming
- Improves UX and perceived speed

### 🧾 Advanced Summarization
- Small documents → direct summary
- Large documents → Map-Reduce summarization

---
---
## 🏗️ Architecture



User Upload

↓

Text Extraction (PDF / DOCX / CSV / XLSX)

↓

Chunking (Semantic / Fixed)

↓

Embeddings → Vector DB (Chroma)

↓

Hybrid Retrieval (BM25 + Semantic)

↓

Cross-Encoder Reranking

↓

LLM (RAG Pipeline)

↓

Streaming Answer + Citations

---
---
## 🛠️ Tech Stack

Tech Stack
    
      Category & Tools

      LLM - OpenAI (GPT-4o-mini)
    
      Embeddings - text-embedding-3-small
    
      Vector DB - ChromaDB
    
      Retrieval - BM25 + Semantic + CrossEncoder
    
      UI - Streamlit, Gradio
    
      Parsing - PyMuPDF, pdfplumber, python-docx, pandas
 

## 📦 Installation

```bash
git clone https://github.com/ChinmayShastry/docchat-ai.git
cd docchat-ai
pip install -r requirements.txt
```

---

## 🔑 Setup

Provide your OpenAI API key:
```env
OPENAI_API_KEY=your_api_key_here
```

---
## ▶️ Run Locally

- **Streamlit:**
  ```bash
  streamlit run app.py
  ```

- **Gradio (Hugging Face style):**
  ```bash
  python app.py
  ```

---
## 📊 Example Use Cases

- 📚 Study assistant for academic PDFs
- 📄 Legal / policy document analysis
- 📊 Business reports exploration
- 📈 Research summarization

---
## ⚠️ Limitations

- OCR not supported in deployed environments (Tesseract unavailable)
- Very large documents may increase response time
- Semantic chunking depends on optional dependencies

---
## 🚧 Future Improvements

- FAISS integration for faster retrieval
- Persistent vector storage
- Multi-modal support (image + text)
- Improved memory handling
- RAG evaluation (RAGAS)

---
## 📁 Project Structure

```
docchat-ai/
│
├── app.py
├── config.py
├── requirements.txt
│
├── src/
│   ├── extractor.py
│   ├── chunking.py
│   ├── indexer.py
│   ├── retriever.py
│   ├── summarizer.py
│   └── chat.py
```

---
## 💡 Key Highlights

- Built hybrid retrieval system
- Implemented Map-Reduce summarization
- Designed streaming LLM pipeline
- Solved real-world PDF extraction issues
- Created multi-platform deployment (Streamlit + Hugging Face)

---
## 👨‍💻 Author

**Chinmay Shastry**
- GitHub: [@ChinmayShastry](https://github.com/ChinmayShastry)

---
## ⭐ Support

If you found this project useful, consider giving it a **star ⭐**





