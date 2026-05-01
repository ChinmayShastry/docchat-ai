import streamlit as st
from src.extractor import extract_text
from src.chunking import create_chunks
from src.retriever import HybridRetriever
from src.summarizer import generate_summary
from src.chat import ask_document_stream
from src.indexer import build_index
from openai import OpenAI
import os

# -----------------------------
# SESSION STATE INIT
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "client" not in st.session_state:
    st.session_state.client = None

if "history_summary" not in st.session_state:
    st.session_state.history_summary = None


# -----------------------------
# UI HEADER
# -----------------------------
st.title("📄 DocChat AI — Advanced RAG System")

# -----------------------------
# SIDEBAR (UPLOAD)
# -----------------------------
st.sidebar.header("Upload Documents")

api_key = st.sidebar.text_input("OpenAI API Key", type="password")
files = st.sidebar.file_uploader(
    "Upload files",
    type=["pdf", "docx", "txt", "xlsx", "csv"],
    accept_multiple_files=True
)

if st.sidebar.button("Process Documents"):

    if not api_key:
        st.error("Enter API key")
    elif not files:
        st.error("Upload files")
    else:
        try:
            client = OpenAI(api_key=api_key)

            all_docs = []
            filenames = []

            for file in files:
                filepath = f"temp_{file.name}"
                with open(filepath, "wb") as f:
                    f.write(file.getbuffer())

                docs = extract_text(filepath, file.name)

                all_docs.extend(docs)
                filenames.append(file.name)

            retriever, n_chunks = build_index(all_docs, api_key)

            summary = generate_summary(all_docs, filenames, client)

            st.session_state.retriever = retriever
            st.session_state.client = client
            st.session_state.messages = []

            st.success("Documents processed!")
            st.write(summary)

        except Exception as e:
            st.error(str(e))


# -----------------------------
# CHAT SECTION
# -----------------------------
st.subheader("💬 Chat")

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Input
if prompt := st.chat_input("Ask about your document..."):

    if not st.session_state.retriever:
        st.warning("Upload documents first")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_container = st.empty()

            full_response = ""

            history = [
                (m["content"], "")
                if m["role"] == "user"
                else ("", m["content"])
                for m in st.session_state.messages
            ]

            for chunk in ask_document_stream(
                prompt,
                history,
                st.session_state.retriever,
                st.session_state.client,
                st.session_state
            ):
                full_response = chunk
                response_container.markdown(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
