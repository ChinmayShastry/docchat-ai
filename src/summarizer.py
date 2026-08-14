# DOCUMENT SUMMARIZER
# Small documents are summarised in a single call. Large ones use Map-Reduce:
#   MAP    — summarise a representative sample of pages individually
#   REDUCE — combine those page-level summaries into one final summary
#
# Sampling rather than summarising every page keeps cost bounded: a 400-page
# PDF costs the same as a 40-page one, at the price of missing detail on pages
# that were not sampled.

from config import Config
from src.logging_setup import get_logger

log = get_logger(__name__)

# Documents at or under this size skip Map-Reduce entirely.
DIRECT_SUMMARY_CHAR_LIMIT = 15_000

# Characters of the combined text sent in a direct summary call.
DIRECT_SUMMARY_CONTEXT = 12_000

# Pages sampled for the MAP stage, and characters used from each.
MAP_SAMPLE_PAGES = 10
MAP_CHARS_PER_PAGE = 2_000

DIRECT_PROMPT = """You are an expert document analyst.
Analyse the document(s) and provide a structured summary.
Include:
1. **What these documents are about** (1-2 sentences)
2. **Key contents / topics covered** (bullet points)
3. **Important numbers or statistics** (if any)
4. **What these documents can be used for** (1-2 sentences)
Documents: {filenames}
Content:
{content}"""

REDUCE_PROMPT = """You are an expert document analyst.
Below are summaries of key pages from a {n_pages}-page document.
Create a comprehensive structured summary of the ENTIRE document.
Include:
1. **What this document is about** (2-3 sentences)
2. **Key contents / topics covered** (bullet points, be specific)
3. **Important numbers, dates, or statistics** (if any)
4. **Key findings or conclusions** (bullet points)
5. **What this document can be used for** (1-2 sentences)
Document: {filenames} ({n_pages} pages total)
Page Summaries:
{summaries}"""


def _sample_pages(docs):
    """Pick pages spread across the document, deduplicated by opening text.

    Even spacing beats taking the first N pages, which on most real documents
    would summarise the table of contents and nothing else.
    """
    n = len(docs)
    stride = max(1, n // MAP_SAMPLE_PAGES)
    candidates = docs[:MAP_SAMPLE_PAGES] if n < 2 * MAP_SAMPLE_PAGES else docs[::stride]

    seen = set()
    sampled = []
    for doc in candidates:
        # Fingerprint on the opening text so repeated boilerplate pages
        # (headers, disclaimers) are not summarised several times over.
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            sampled.append(doc)

    return sampled


def generate_summary(docs, filenames, client):
    """Summarise `docs`, choosing a direct or Map-Reduce strategy by size."""
    if not docs:
        return "No document content available to summarise."

    total_text = "\n\n".join(
        f"[{doc.metadata.get('source', 'doc')}]\n{doc.page_content}" for doc in docs
    )
    total_chars = len(total_text)
    names = ", ".join(filenames)

    log.info("Summarising %d characters across %d page(s)", total_chars, len(docs))

    if total_chars <= DIRECT_SUMMARY_CHAR_LIMIT:
        response = client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[{
                "role": "user",
                "content": DIRECT_PROMPT.format(
                    filenames=names,
                    content=total_text[:DIRECT_SUMMARY_CONTEXT],
                ),
            }],
            temperature=0.3,
        )
        return response.choices[0].message.content

    log.info("Large document — using Map-Reduce summarisation")
    return _map_reduce_summary(docs, names, client)


def _map_reduce_summary(docs, names, client):
    n_pages = len(docs)
    sampled = _sample_pages(docs)
    log.info("Sampled %d representative page(s) from %d", len(sampled), n_pages)

    # ── MAP ───────────────────────────────────────────────────────────────
    page_summaries = []
    for i, doc in enumerate(sampled):
        content = doc.page_content[:MAP_CHARS_PER_PAGE]
        if not content.strip():
            continue

        src = doc.metadata.get("source", "doc")
        page = doc.metadata.get("page", i + 1)

        response = client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "Summarise this page in 3-4 bullet points. Be concise and factual.\n"
                    "Focus on key information, numbers, names, and decisions.\n\n"
                    f"Source: {src}, Page {page}\n"
                    f"Content: {content}"
                ),
            }],
            temperature=0.2,
            max_tokens=200,
        )
        page_summaries.append(f"[Page {page}]\n{response.choices[0].message.content}")

    if not page_summaries:
        return "Could not generate a summary — no readable page content."

    # ── REDUCE ────────────────────────────────────────────────────────────
    response = client.chat.completions.create(
        model=Config.LLM_MODEL,
        messages=[{
            "role": "user",
            "content": REDUCE_PROMPT.format(
                n_pages=n_pages,
                filenames=names,
                summaries="\n\n".join(page_summaries),
            ),
        }],
        temperature=0.3,
        max_tokens=800,
    )

    log.info("Map-Reduce summarisation complete")
    return response.choices[0].message.content
