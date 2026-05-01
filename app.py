from src.extractor import extract_text
from src.chunking import create_chunks
from src.retriever import HybridRetriever
from src.summarizer import generate_summary
from src.chat import ask_document_stream, normalize_history
from config import Config


# ─────────────────────────────────────────────────────────────────────────────
# INDEX BUILDER
# Embeds all chunks into ChromaDB and wires up the HybridRetriever.
# Called once per document upload session.
# ─────────────────────────────────────────────────────────────────────────────

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



# ----------------------------------------------------------------------------------------------------



# ─────────────────────────────────────────────────────────────────────────────
# STREAMING CHAT WITH CITATIONS
# Retrieves relevant chunks, builds a context window, streams the LLM response
# token-by-token, then appends a source footer and a grounded-ness check.
#
# Conversation history is compressed into a rolling summary after 5 turns to
# keep the token count under control without losing context.
# ─────────────────────────────────────────────────────────────────────────────

def ask_document_stream(question, history, retriever, client, session_state):
    """Stream the answer to `question` given the retrieved document context."""

    def classify_query(q):
        """Route the query to a retrieval strategy based on detected intent."""
        q = q.lower()
        if "summary" in q or "overview" in q:
            return "summary"
        elif "compare" in q:
            return "comparison"
        elif any(w in q for w in ["how many", "number", "total"]):
            return "numeric"
        return "factual"

    query_type = classify_query(question)

    # Comparison queries benefit from seeing more chunks (more documents in context)
    k = 6 if query_type == "comparison" else Config.RETRIEVAL_K

    chunks = retriever.retrieve(question, k=k)

    # Deduplicate chunks by content to avoid redundant context
    seen          = set()
    unique_chunks = []
    for c in chunks:
        if c.page_content not in seen:
            unique_chunks.append(c)
            seen.add(c.page_content)
    chunks = unique_chunks

    print(f"  Retrieved {len(chunks)} unique chunks")
    print(f"  Sources  : {[c.metadata.get('source') for c in chunks]}")

    # Build labelled context string with source tags the LLM can cite
    context_parts = []
    sources_used  = []
    for i, chunk in enumerate(chunks):
        src  = chunk.metadata.get("source", "Document")
        page = chunk.metadata.get("page", 1)
        context_parts.append(f"[Doc{i+1}: {src}, Page {page}]\n{chunk.page_content}")
        sources_used.append(f"{src} (Page {page})")

    context_str = "\n\n".join(context_parts)

    # Trim context to stay within the LLM's effective token window.
    # rsplit on a chunk boundary so we don't cut inside a [DocN] block.
    if len(context_str) > Config.MAX_CONTEXT_CHARS:
        context_str = context_str[:Config.MAX_CONTEXT_CHARS].rsplit("[Doc", 1)[0]

    # ── Build message list ────────────────────────────────────────────────
    messages = [{
        "role": "system",
        "content": (
            "You are a document-based AI assistant.\n\n"
            "Rules:\n"
            "- Answer ONLY using the provided document context.\n"
            "- If the answer is not in the documents, say: "
            "\"I don't see that information in the document(s).\"\n"
            "- Never hallucinate or invent facts.\n"
            "- Cite sources inline using [Doc1], [Doc2], etc.\n"
            "- Be concise, accurate, and structured."
        )
    }]

    # ── Conversation history (compressed after 5 turns) ───────────────────
    # Re-summarise every 5 new turns so the summary stays fresh as the
    # conversation grows (not just once at turn 6 and then frozen forever).
    if len(history) > 5:
        should_refresh = (
            not session_state.get("history_summary")
            or len(history) % 5 == 0
        )
        if should_refresh:
            summary_prompt = "Summarise this conversation briefly:\n"
            for turn in history:
                summary_prompt += f"User: {turn[0]}\nAssistant: {turn[1]}\n"

            summary_resp = client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3,
                max_tokens=150
            )
            session_state["history_summary"] = summary_resp.choices[0].message.content

        messages.append({
            "role": "system",
            "content": f"Conversation summary so far:\n{session_state['history_summary']}"
        })
    else:
        # For short histories, include turns verbatim for full accuracy
        for turn in history:
            messages.append({"role": "user",      "content": turn[0]})
            messages.append({"role": "assistant", "content": turn[1]})

    messages.append({
        "role": "user",
        "content": f"Document context:\n{context_str}\n\nQuestion: {question}"
    })

    # ── Stream response token-by-token ────────────────────────────────────
    stream = client.chat.completions.create(
        model=Config.LLM_MODEL,
        messages=messages,
        temperature=0.2,
        stream=True
    )

    partial = ""
    for chunk in stream:
        delta = getattr(chunk.choices[0].delta, "content", None)
        if delta:
            partial += delta
            yield partial   # push each token increment to the UI

    # ── Post-stream: groundedness verification ────────────────────────────
    # Ask the LLM to flag any part of its own answer not supported by context.
    # FIX: check for "ok" case-insensitively in the first 20 chars to avoid
    # false positives from GPT saying "OK." or "OK, the answer is supported."
    verification_prompt = (
        f"Context:\n{context_str}\n\n"
        f"Answer:\n{partial}\n\n"
        f"Task: Reply 'OK' if every claim is fully supported by the context. "
        f"Otherwise list the unsupported parts."
    )

    verif   = client.chat.completions.create(
        model=Config.LLM_MODEL,
        messages=[{"role": "user", "content": verification_prompt}],
        temperature=0
    )
    verdict = verif.choices[0].message.content.strip()

    if "ok" not in verdict.lower()[:20]:
        partial += f"\n\n⚠️ **Potentially unsupported content:**\n{verdict}"

    # Append deduplicated source footer
    unique_sources = list(dict.fromkeys(sources_used))
    footer = f"\n\n---\n📄 **Sources:** {' | '.join(unique_sources)}"
    yield partial + footer

# ----------------------------------------------------------------------------------------------------


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO HANDLER — DOCUMENT PROCESSING
# Called when the user clicks "Process Documents".
# Validates input, extracts text, builds the index, generates a summary,
# and stores everything in per-session state.
# ─────────────────────────────────────────────────────────────────────────────

def process_documents(api_key, files, session_state):
    # ── Input validation ──────────────────────────────────────────────────
    if not api_key or not api_key.startswith("sk-"):
        return (
            "❌ Please enter a valid OpenAI API key (starts with `sk-`).",
            session_state
        )

    if not files:
        return (
            "❌ Please upload at least one document.",
            session_state
        )

    try:
        # Quick connectivity check — catches expired/invalid keys before any
        # expensive embedding or summarisation calls are made
        client = OpenAI(api_key=api_key)
        client.models.list()

    except Exception:
        return (
            "❌ API key rejected by OpenAI. Please check that it is valid and has available credits.",
            session_state
        )

    try:
        all_docs  = []
        filenames = []

        for file in files:
            filename = os.path.basename(file.name)
            docs     = extract_text(file.name, filename)

            if not docs:
                # Surface a clear per-file warning instead of silently skipping
                return (
                    f"❌ Could not extract text from **{filename}**. "
                    f"If it is a scanned PDF, make sure Tesseract and Poppler are installed.",
                    session_state
                )

            all_docs.extend(docs)
            filenames.append(filename)

        if not all_docs:
            return (
                "❌ No text could be extracted from any of the uploaded file(s).",
                session_state
            )

        # Build vector index + BM25 index
        retriever, n_chunks = build_index(all_docs, api_key)

        # Generate document summary shown in the Upload tab
        summary = generate_summary(all_docs, filenames, client)

        # Store everything in per-session state (never shared between users)
        session_state["retriever"]       = retriever
        session_state["client"]          = client
        session_state["filenames"]       = filenames
        session_state["history_summary"] = None  # reset on new upload

        doc_info = (
            f"\n\n---\n"
            f"📁 **Loaded:** {', '.join(filenames)} | "
            f"**Chunks:** {n_chunks}"
        )
        return (summary + doc_info, session_state)

    except Exception as e:
        return (f"❌ Unexpected error: {str(e)}", session_state)


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO HANDLER — CHAT
# Called on every user message in the Chat tab.
# Delegates to the streaming generator; yields partial responses to the UI.
# ─────────────────────────────────────────────────────────────────────────────

def chat_with_doc(message, history, session_state):
    """Stream the assistant reply for a chat message."""
    if not session_state.get("retriever"):
        yield "⚠️ Please upload and process a document first (go to the Upload & Summary tab)."
        return

    try:
        # FIX: pass session_state as the 5th argument (was missing — caused crash)
        for partial in ask_document_stream(
            message,
            history,
            session_state["retriever"],
            session_state["client"],
            session_state           # ← required for history compression
        ):
            yield partial

    except Exception as e:
        yield f"❌ Error during chat: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO UI
# Two-tab layout:
#   Tab 1 — Upload documents + view auto-generated summary
#   Tab 2 — Multi-turn chat with streaming responses and source citations
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="DocChat AI", theme=gr.themes.Soft()) as demo:

    # Isolated per-user session state — never shared across browser sessions
    session_state = gr.State({})

    gr.Markdown("""
    # 📄 DocChat AI — Advanced RAG System
    **Upload documents → Get an instant summary → Chat with source citations**
    > Supports PDF · DOCX · TXT · XLSX · CSV &nbsp;|&nbsp; Multi-document Q&A &nbsp;|&nbsp; Streaming responses
    """)

    with gr.Tab("📂 Upload & Summary"):
        with gr.Row():
            with gr.Column(scale=1):
                api_input = gr.Textbox(
                    label="🔑 OpenAI API Key",
                    placeholder="sk-...",
                    type="password"
                )
                file_input = gr.File(
                    label="📁 Upload Documents (select multiple)",
                    file_types=[".pdf", ".docx", ".txt", ".xlsx", ".xls", ".csv"],
                    file_count="multiple"
                )
                process_btn = gr.Button(
                    "⚡ Process Documents",
                    variant="primary",
                    size="lg"
                )
                gr.Markdown("""
                **What happens when you click Process:**
                - Text is extracted from your file(s)
                - Documents are split into semantic or fixed chunks
                - Chunks are embedded and stored in a local vector store
                - A BM25 keyword index is built in parallel
                - An automatic summary is generated
                """)

            with gr.Column(scale=2):
                summary_output = gr.Markdown(
                    label="📋 Document Summary",
                    value="*Upload documents and click Process — your summary will appear here.*"
                )

        # FIX: outputs now matches the 2-value tuple returned by process_documents.
        # The old code had a dangling gr.State() as a 3rd output that was never
        # connected to anything, causing the gr.update(interactive=...) to be lost.
        process_btn.click(
            fn=process_documents,
            inputs=[api_input, file_input, session_state],
            outputs=[summary_output, session_state]
        )

    with gr.Tab("💬 Chat with Documents"):
        gr.ChatInterface(
            fn=chat_with_doc,
            additional_inputs=[session_state],
            chatbot=gr.Chatbot(height=480),
            textbox=gr.Textbox(
                placeholder="Ask anything about your document(s)...",
                container=False
            ),
            examples=[
                ["Give me a brief overview of these documents"],
                ["What are the most important points?"],
                ["Summarise the key statistics or numbers"],
                ["What are the main risks or concerns mentioned?"],
                ["Compare the key differences between the documents"],
            ],
        )

    gr.Markdown("""
    ---
    > ⚠️ Your API key is used only within your session and is never stored or logged.
    > Built with LangChain · ChromaDB · Hybrid BM25 + Semantic Search · Cross-Encoder Reranking · RAGAS Evaluated
    """)


demo.launch()
