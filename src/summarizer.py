# DOCUMENT SUMMARIZER
# Uses a Map-Reduce strategy for large documents:
#   MAP    — summarise a representative sample of pages individually
#   REDUCE — combine those page-level summaries into one final summary
#
# Small documents (<15k chars) are summarised in a single direct call.

from config import Config

def generate_summary(docs, filenames, client):
    total_text = "\n\n".join([
        f"[{doc.metadata.get('source', 'doc')}]\n{doc.page_content}"
        for doc in docs
    ])
    total_chars = len(total_text)
    print(f"📝 Summarising {total_chars:,} chars across {len(docs)} page(s)...")

    # ── Direct summary for small documents ───────────────────────────────
    if total_chars <= 15_000:
        prompt = f"""You are an expert document analyst.
Analyse the document(s) and provide a structured summary.
Include:
1. **What these documents are about** (1-2 sentences)
2. **Key contents / topics covered** (bullet points)
3. **Important numbers or statistics** (if any)
4. **What these documents can be used for** (1-2 sentences)
Documents: {', '.join(filenames)}
Content:
{total_text[:12_000]}"""

        response = client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content

    # ── Map-Reduce for large documents ───────────────────────────────────
    print("📚 Large document detected — using Map-Reduce summarisation")
    n = len(docs)

    # Sample representative pages: evenly spaced so we cover the whole doc
    sampled_raw = docs[:10] if n < 20 else docs[::max(1, n // 10)]

    # FIX: deduplicate by content fingerprint (not object identity via id())
    seen    = set()
    sampled = []
    for d in sampled_raw:
        key = d.page_content[:100]
        if key not in seen:
            seen.add(key)
            sampled.append(d)

    print(f"  Sampled {len(sampled)} representative pages from {n} total")

    # MAP: summarise each sampled page in isolation
    page_summaries = []
    for i, doc in enumerate(sampled):
        src     = doc.metadata.get("source", "doc")
        page    = doc.metadata.get("page", i + 1)
        content = doc.page_content[:2_000]
        if not content.strip():
            continue

        resp = client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    f"Summarise this page in 3-4 bullet points. Be concise and factual.\n"
                    f"Focus on key information, numbers, names, and decisions.\n\n"
                    f"Source: {src}, Page {page}\n"
                    f"Content: {content}"
                )
            }],
            temperature=0.2,
            max_tokens=200
        )
        page_summaries.append(f"[Page {page}]\n{resp.choices[0].message.content}")

    # REDUCE: combine all page summaries into one final structured summary
    combined      = "\n\n".join(page_summaries)
    reduce_prompt = f"""You are an expert document analyst.
Below are summaries of key pages from a {n}-page document.
Create a comprehensive structured summary of the ENTIRE document.
Include:
1. **What this document is about** (2-3 sentences)
2. **Key contents / topics covered** (bullet points, be specific)
3. **Important numbers, dates, or statistics** (if any)
4. **Key findings or conclusions** (bullet points)
5. **What this document can be used for** (1-2 sentences)
Document: {', '.join(filenames)} ({n} pages total)
Page Summaries:
{combined}"""

    final_response = client.chat.completions.create(
        model=Config.LLM_MODEL,
        messages=[{"role": "user", "content": reduce_prompt}],
        temperature=0.3,
        max_tokens=800
    )
    print("✅ Map-Reduce summarisation complete")
    return final_response.choices[0].message.content
