# HISTORY NORMALIZER
# Gradio changed its chat history format across major versions:
#
#   Gradio ≤ 3.x / tuples mode  →  [[user_str, assistant_str], ...]
#   Gradio 4.x  / messages mode →  [{"role": "user",      "content": "..."},
#                                    {"role": "assistant", "content": "..."}, ...]
#
# The rest of the code expects plain (user, assistant) string tuples.
# This function normalises both formats and silently drops any incomplete
# turns (assistant = None) that Gradio can produce mid-stream.

from config import Config

def normalize_history(history):
    """Return history as a list of (user_text, assistant_text) string tuples."""
    if not history:
        return []

    normalized = []

    # ── Detect format ─────────────────────────────────────────────────────
    if history and isinstance(history[0], dict):
        # Messages format: flat list of role/content dicts.
        # Walk through pairs: user message followed immediately by assistant message.
        i = 0
        while i < len(history) - 1:
            user_entry = history[i]
            asst_entry = history[i + 1]
            if (user_entry.get("role") == "user" and
                    asst_entry.get("role") == "assistant"):
                user_text = user_entry.get("content") or ""
                asst_text = asst_entry.get("content") or ""
                # Skip turns where the assistant reply hasn't arrived yet (None/empty)
                if user_text.strip() and asst_text.strip():
                    normalized.append((str(user_text), str(asst_text)))
                i += 2
            else:
                i += 1  # skip malformed entries
    else:
        # Tuples / lists format: each item is [user_str, assistant_str].
        for turn in history:
            if (len(turn) >= 2
                    and turn[0] is not None
                    and turn[1] is not None
                    and str(turn[0]).strip()
                    and str(turn[1]).strip()):
                normalized.append((str(turn[0]), str(turn[1])))

    return normalized


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

    # Normalise history immediately so the rest of this function never has
    # to worry about Gradio's format differences or None assistant values.
    history = normalize_history(history)

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
    # Split on a [Doc boundary so we never cut inside a labelled chunk.
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
    # history is now guaranteed to be List[Tuple[str, str]] — safe to index.
    # Re-summarise every 5 new turns so the summary stays fresh as the
    # conversation grows (not just once at turn 6 and then frozen forever).
    if len(history) > 5:
        should_refresh = (
            not session_state.get("history_summary")
            or len(history) % 5 == 0
        )
        if should_refresh:
            summary_prompt = "Summarise this conversation briefly:\n"
            for user_msg, asst_msg in history:
                summary_prompt += f"User: {user_msg}\nAssistant: {asst_msg}\n"

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
        for user_msg, asst_msg in history:
            messages.append({"role": "user",      "content": user_msg})
            messages.append({"role": "assistant", "content": asst_msg})

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
