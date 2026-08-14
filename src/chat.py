# STREAMING CHAT WITH CITATIONS
# Retrieves relevant chunks, builds a labelled context window, streams the LLM
# response token-by-token, then appends a source footer and a groundedness
# check that flags claims the context does not support.
#
# Conversation history is compressed into a rolling summary past a threshold so
# token cost stays flat as a conversation grows, instead of climbing linearly.

from config import Config
from src.logging_setup import get_logger

log = get_logger(__name__)

REFUSAL_TEXT = "I don't see that information in the document(s)."

SYSTEM_PROMPT = (
    "You are a document-based AI assistant.\n\n"
    "Rules:\n"
    "- Answer ONLY using the provided document context.\n"
    f'- If the answer is not in the documents, say: "{REFUSAL_TEXT}"\n'
    "- Never hallucinate or invent facts.\n"
    "- Cite sources inline using [Doc1], [Doc2], etc.\n"
    "- Be concise, accurate, and structured."
)


def _safe_str(value):
    """Convert an arbitrary history payload to a plain string."""
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(map(str, value))
    if isinstance(value, dict):
        return str(value.get("content", ""))
    return str(value)


def normalize_history(history):
    """Return history as clean (user, assistant) string tuples.

    Accepts either the role/content dict format used by the OpenAI API and the
    Streamlit message list, or plain (user, assistant) pairs. Incomplete turns
    — an unanswered question, a None assistant value mid-stream — are dropped
    so downstream code can index into pairs without defensive checks.
    """
    if not history:
        return []

    normalized = []

    if isinstance(history[0], dict):
        i = 0
        while i < len(history) - 1:
            user_entry, asst_entry = history[i], history[i + 1]

            if user_entry.get("role") == "user" and asst_entry.get("role") == "assistant":
                user_text = _safe_str(user_entry.get("content"))
                asst_text = _safe_str(asst_entry.get("content"))

                if user_text.strip() and asst_text.strip():
                    normalized.append((user_text, asst_text))
                i += 2
            else:
                i += 1
    else:
        for turn in history:
            if len(turn) >= 2:
                user_text = _safe_str(turn[0])
                asst_text = _safe_str(turn[1])

                if user_text.strip() and asst_text.strip():
                    normalized.append((user_text, asst_text))

    return normalized


def classify_query(question):
    """Route the query to a retrieval strategy based on detected intent.

    Deliberately keyword-based: it is transparent, costs nothing, and an LLM
    classifier would add a round-trip to every question for a decision that
    only adjusts how many chunks we fetch.
    """
    q = question.lower()

    if "summary" in q or "overview" in q:
        return "summary"
    if "compare" in q or "difference" in q:
        return "comparison"
    if any(word in q for word in ("how many", "number", "total", "count")):
        return "numeric"
    return "factual"


def build_context(chunks):
    """Return (context_string, source_labels) from retrieved chunks."""
    context_parts = []
    sources = []

    for i, chunk in enumerate(chunks, start=1):
        src = chunk.metadata.get("source", "Document")
        page = chunk.metadata.get("page", 1)

        context_parts.append(f"[Doc{i}: {src}, Page {page}]\n{chunk.page_content}")
        sources.append(f"{src} (Page {page})")

    context = "\n\n".join(context_parts)

    # Trim to stay inside the token budget, cutting at a [Doc boundary so a
    # chunk is never truncated mid-sentence and mis-attributed.
    if len(context) > Config.MAX_CONTEXT_CHARS:
        context = context[:Config.MAX_CONTEXT_CHARS].rsplit("[Doc", 1)[0]

    return context, sources


def _dedupe(chunks):
    seen = set()
    unique = []
    for chunk in chunks:
        if chunk.page_content not in seen:
            seen.add(chunk.page_content)
            unique.append(chunk)
    return unique


def _summarise_history(history, client):
    prompt = "Summarise this conversation briefly:\n"
    for user_msg, asst_msg in history:
        prompt += f"User: {user_msg}\nAssistant: {asst_msg}\n"

    response = client.chat.completions.create(
        model=Config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=150,
    )
    return response.choices[0].message.content


def _build_messages(question, history, context, session_state):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    threshold = Config.HISTORY_SUMMARY_THRESHOLD

    if len(history) > threshold:
        # Refresh every `threshold` turns so the summary tracks the
        # conversation instead of freezing at the moment it was first built.
        needs_refresh = (
            not session_state.get("history_summary")
            or len(history) % threshold == 0
        )
        if needs_refresh:
            session_state["history_summary"] = _summarise_history(history, session_state["_client"])
            log.debug("Refreshed conversation summary at %d turns", len(history))

        messages.append({
            "role": "system",
            "content": f"Conversation summary so far:\n{session_state['history_summary']}",
        })
    else:
        for user_msg, asst_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": asst_msg})

    messages.append({
        "role": "user",
        "content": f"Document context:\n{context}\n\nQuestion: {question}",
    })

    return messages


def verify_groundedness(context, answer, client):
    """Ask the model to flag claims its own answer made without support.

    A second pass is cheap relative to a wrong answer in a document QA tool,
    and it catches the failure mode users care most about.
    """
    prompt = (
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}\n\n"
        "Task: Reply 'OK' if every claim is fully supported by the context. "
        "Otherwise list the unsupported parts."
    )

    response = client.chat.completions.create(
        model=Config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    verdict = response.choices[0].message.content.strip()

    # Check only the opening of the verdict: a supported answer starts with
    # "OK", while an unsupported one may still mention the word later on.
    is_grounded = "ok" in verdict.lower()[:20]
    return is_grounded, verdict


def ask_document_stream(question, history, retriever, client, session_state):
    """Stream the answer to `question` given the retrieved document context."""
    history = normalize_history(history)

    # _build_messages needs the client for history compression; passing it via
    # session_state keeps the helper's signature from growing.
    session_state["_client"] = client

    query_type = classify_query(question)
    # Comparison questions need to see more documents at once to contrast them.
    k = 6 if query_type == "comparison" else Config.RETRIEVAL_K

    scored = retriever.retrieve_with_scores(question, k=k)
    chunks = _dedupe([chunk for chunk, _ in scored])

    log.info(
        "Query type=%s | %d chunk(s) from %s",
        query_type, len(chunks), {c.metadata.get("source") for c in chunks},
    )

    if not chunks:
        yield REFUSAL_TEXT
        return

    context, sources = build_context(chunks)

    # Expose retrieval detail so the UI can show what the answer was built on.
    session_state["last_retrieval"] = [
        {
            "source": chunk.metadata.get("source", "Document"),
            "page": chunk.metadata.get("page", 1),
            "score": score,
            "text": chunk.page_content,
        }
        for chunk, score in scored
    ]

    messages = _build_messages(question, history, context, session_state)

    stream = client.chat.completions.create(
        model=Config.LLM_MODEL,
        messages=messages,
        temperature=0.2,
        stream=True,
    )

    partial = ""
    for chunk in stream:
        delta = getattr(chunk.choices[0].delta, "content", None)
        if delta:
            partial += delta
            yield partial

    # ── Post-stream verification ──────────────────────────────────────────
    if Config.ENABLE_GROUNDEDNESS_CHECK and partial.strip():
        try:
            is_grounded, verdict = verify_groundedness(context, partial, client)
            if not is_grounded:
                log.info("Groundedness check flagged unsupported content")
                partial += f"\n\n⚠️ **Potentially unsupported content:**\n{verdict}"
        except Exception as exc:
            # A verification failure must not discard an answer already shown.
            log.warning("Groundedness check failed: %s", exc)

    unique_sources = list(dict.fromkeys(sources))
    yield partial + f"\n\n---\n📄 **Sources:** {' | '.join(unique_sources)}"
