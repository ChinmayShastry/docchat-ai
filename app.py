"""DocChat AI — Streamlit interface.

Upload documents, get a structured summary, then ask questions answered only
from the uploaded content, with inline citations and a groundedness warning
when the model strays beyond its sources.
"""

import os
import tempfile
from contextlib import contextmanager

import streamlit as st
from openai import OpenAI

from config import Config
from src.chat import ask_document_stream
from src.extractor import extract_text, validate_upload
from src.indexer import build_index
from src.logging_setup import get_logger
from src.summarizer import generate_summary

log = get_logger(__name__)

st.set_page_config(page_title="DocChat AI", page_icon="📄", layout="wide")

DEFAULTS = {
    "messages": [],
    "retriever": None,
    "client": None,
    "history_summary": None,
    "summary": None,
    "doc_names": [],
    "last_retrieval": [],
}

for key, default in DEFAULTS.items():
    st.session_state.setdefault(key, default)


@contextmanager
def staged_uploads(files):
    """Write uploads to temp files, guaranteeing cleanup afterwards.

    The parsers need real paths on disk, but the previous implementation used
    delete=False and never removed them — every upload leaked a file for the
    lifetime of the host.
    """
    paths = []
    try:
        for file in files:
            suffix = os.path.splitext(file.name)[1] or ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file.getbuffer())
                paths.append((tmp.name, file.name))
        yield paths
    finally:
        for path, _ in paths:
            try:
                os.unlink(path)
            except OSError as exc:
                log.warning("Could not remove temp file %s: %s", path, exc)


def resolve_api_key(entered):
    """Prefer a key typed into the sidebar, else fall back to the environment."""
    return entered.strip() or os.getenv("OPENAI_API_KEY", "")


def process_documents(api_key, files):
    """Extract, index and summarise uploads. Returns True on success."""
    progress = st.sidebar.progress(0.0, text="Reading files…")

    try:
        client = OpenAI(api_key=api_key)

        with staged_uploads(files) as staged:
            all_docs = []
            names = []

            for i, (path, name) in enumerate(staged):
                validate_upload(path, name, Config.MAX_UPLOAD_MB)

                docs = extract_text(path, name)
                if not docs:
                    st.sidebar.warning(f"No readable text found in {name} — skipping.")
                    continue

                all_docs.extend(docs)
                names.append(name)
                progress.progress((i + 1) / len(staged) * 0.4, text=f"Read {name}")

        if not all_docs:
            st.sidebar.error(
                "None of the uploaded files contained readable text. "
                "Scanned or image-only PDFs need OCR, which is not enabled here."
            )
            return False

        progress.progress(0.5, text="Building index…")
        retriever, n_chunks = build_index(all_docs, api_key)

        progress.progress(0.75, text="Summarising…")
        summary = generate_summary(all_docs, names, client)

        st.session_state.update(
            retriever=retriever,
            client=client,
            summary=summary,
            doc_names=names,
            messages=[],
            history_summary=None,
            last_retrieval=[],
        )

        progress.progress(1.0, text=f"Ready — {n_chunks} chunks indexed")

        if n_chunks < 5:
            st.sidebar.warning(
                "Only a few chunks were extracted, so answers may be thin."
            )
        return True

    except Exception as exc:
        log.exception("Document processing failed")
        progress.empty()
        st.sidebar.error(str(exc))
        return False


def build_history(messages):
    """Pair consecutive user/assistant messages into turns."""
    history = []
    i = 0
    while i < len(messages) - 1:
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            history.append((messages[i]["content"], messages[i + 1]["content"]))
            i += 2
        else:
            i += 1
    return history


def render_retrieval_detail():
    """Show which chunks the last answer was built from, and how they scored."""
    retrieved = st.session_state.last_retrieval
    if not retrieved:
        return

    with st.expander(f"🔍 Retrieved context ({len(retrieved)} chunks)"):
        for i, item in enumerate(retrieved, start=1):
            st.markdown(
                f"**[Doc{i}] {item['source']} — page {item['page']}** "
                f"· relevance `{item['score']:.3f}`"
            )
            st.text(item["text"][:600] + ("…" if len(item["text"]) > 600 else ""))
            st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────────

st.sidebar.header("📄 DocChat AI")

entered_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Leave blank to use the OPENAI_API_KEY environment variable.",
)

uploads = st.sidebar.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "txt", "xlsx", "csv"],
    accept_multiple_files=True,
    help=f"Up to {Config.MAX_UPLOAD_MB} MB per file.",
)

col_process, col_reset = st.sidebar.columns(2)

if col_process.button("Process", use_container_width=True, type="primary"):
    api_key = resolve_api_key(entered_key)

    if not api_key:
        st.sidebar.error("Enter an API key or set OPENAI_API_KEY.")
    elif not uploads:
        st.sidebar.error("Upload at least one document.")
    else:
        with st.spinner("Processing documents…"):
            process_documents(api_key, uploads)

if col_reset.button("Reset", use_container_width=True):
    st.session_state.update(DEFAULTS, messages=[], doc_names=[], last_retrieval=[])
    st.rerun()

if st.session_state.doc_names:
    st.sidebar.caption("Loaded: " + ", ".join(st.session_state.doc_names))

# ── Main pane ─────────────────────────────────────────────────────────────

st.title("📄 DocChat AI")
st.caption("Retrieval-augmented document chat with source citations.")

if st.session_state.summary:
    with st.expander("📋 Document summary", expanded=not st.session_state.messages):
        st.markdown(st.session_state.summary)

if not st.session_state.retriever:
    st.info("Upload one or more documents in the sidebar to get started.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.messages:
    render_retrieval_detail()

if prompt := st.chat_input("Ask about your documents…"):
    if not st.session_state.retriever:
        st.warning("Upload and process documents first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            container = st.empty()
            full_response = ""

            try:
                for partial in ask_document_stream(
                    prompt,
                    build_history(st.session_state.messages),
                    st.session_state.retriever,
                    st.session_state.client,
                    st.session_state,
                ):
                    full_response = partial
                    container.markdown(full_response)
            except Exception as exc:
                log.exception("Chat request failed")
                full_response = f"❌ {exc}"
                container.error(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
        st.rerun()
