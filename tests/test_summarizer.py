"""Tests for summarisation strategy selection and page sampling."""

from langchain_core.documents import Document

from src.summarizer import DIRECT_SUMMARY_CHAR_LIMIT, _sample_pages, generate_summary


def make_pages(n, chars=200, prefix="Page content"):
    return [
        Document(
            page_content=f"{prefix} {i} " + ("x" * chars),
            metadata={"source": "doc.pdf", "page": i + 1},
        )
        for i in range(n)
    ]


def test_empty_docs_returns_message(fake_client):
    assert "No document content" in generate_summary([], [], fake_client())


def test_small_document_uses_single_call(fake_client):
    client = fake_client(["A concise summary."])
    result = generate_summary(make_pages(2), ["doc.pdf"], client)

    assert result == "A concise summary."
    assert len(client.chat.completions.calls) == 1, "small docs need exactly one call"


def test_large_document_uses_map_reduce(fake_client):
    # Comfortably over the direct-summary limit.
    pages = make_pages(30, chars=DIRECT_SUMMARY_CHAR_LIMIT // 10)
    client = fake_client(["page summary"] * 40 + ["final combined summary"])

    result = generate_summary(pages, ["doc.pdf"], client)

    assert len(client.chat.completions.calls) > 1, "large docs need MAP + REDUCE calls"
    assert result  # the final REDUCE output


def test_map_reduce_samples_rather_than_reading_every_page(fake_client):
    """Cost must stay bounded as page count grows."""
    pages = make_pages(300, chars=DIRECT_SUMMARY_CHAR_LIMIT // 10)
    client = fake_client(["page summary"] * 400 + ["final"])

    generate_summary(pages, ["doc.pdf"], client)

    # One REDUCE call plus a bounded number of MAP calls — far fewer than 300.
    assert len(client.chat.completions.calls) < 30


def test_sample_pages_spreads_across_document():
    pages = make_pages(100)
    sampled = _sample_pages(pages)

    page_numbers = [d.metadata["page"] for d in sampled]

    assert len(sampled) <= 15
    assert max(page_numbers) > 50, "sampling must reach the back of the document"


def test_sample_pages_deduplicates_boilerplate():
    """Repeated header/disclaimer pages should be summarised once."""
    identical = [
        Document(page_content="CONFIDENTIAL — do not distribute.",
                 metadata={"source": "d.pdf", "page": i})
        for i in range(20)
    ]

    assert len(_sample_pages(identical)) == 1


def test_sample_pages_short_document_returns_all():
    pages = make_pages(4)
    assert len(_sample_pages(pages)) == 4
