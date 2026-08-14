"""RAG evaluation harness.

Measures two things separately, because they fail for different reasons:

  Retrieval  — did the pipeline surface the text containing the answer?
               Scored by exact span matching against ground truth, so these
               metrics are deterministic and cost nothing beyond embeddings.

  Generation — given what was retrieved, was the answer faithful and correct?
               Scored by an LLM judge, and only meaningful when retrieval
               succeeded.

Separating them matters: an answer can be wrong because the right chunk was
never retrieved, or because the model ignored a chunk that was. Averaging both
into a single score hides which one you need to fix.

Usage:
    export OPENAI_API_KEY=sk-...
    python eval/run_eval.py                      # all modes, retrieval + generation
    python eval/run_eval.py --retrieval-only     # free apart from embeddings
    python eval/run_eval.py --modes hybrid_rerank
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config  # noqa: E402
from src.logging_setup import get_logger  # noqa: E402

# Retrieval and vector-store imports are deliberately deferred into main().
# Dataset validation is the one operation that should work without an API key
# or the full runtime stack installed, so nothing heavy loads at import time.

log = get_logger("eval")

ALL_MODES = ("bm25", "semantic", "hybrid", "hybrid_rerank")

REFUSAL_MARKERS = (
    "i don't see that information",
    "i do not see that information",
    "not in the document",
    "does not contain",
    "no information",
    "not mentioned",
    "not stated",
    "not disclosed",
    "not provided",
)

JUDGE_PROMPT = """You are grading a document question-answering system.

QUESTION:
{question}

REFERENCE ANSWER (the known correct answer):
{ground_truth}

RETRIEVED CONTEXT (all the system was allowed to use):
{context}

SYSTEM ANSWER:
{answer}

Grade on two independent criteria:

1. faithful — Is every factual claim in the SYSTEM ANSWER supported by the
   RETRIEVED CONTEXT? Judge only against the context, not your own knowledge.
   If the answer says the information is unavailable and the context indeed
   lacks it, that is faithful.

2. correct — Does the SYSTEM ANSWER convey the same facts as the REFERENCE
   ANSWER? Ignore wording, formatting, and extra detail that does not
   contradict the reference.

Reply with strict JSON only, no markdown fence:
{{"faithful": true/false, "correct": true/false, "reason": "one short sentence"}}"""


def normalise_text(text):
    """Collapse whitespace so span matching survives re-wrapping during chunking."""
    return re.sub(r"\s+", " ", text).strip().lower()


def load_dataset():
    with open(EVAL_DIR / "dataset.json", encoding="utf-8") as handle:
        return json.load(handle)


def load_corpus_files():
    """Every document indexed during evaluation, primary first.

    The distractors matter as much as the target document: with a single short
    report the index is small enough that retrieving k=4 returns a large
    fraction of it, every strategy scores near-perfectly, and the benchmark
    cannot discriminate. Distractors that share vocabulary but hold different
    figures force retrieval to identify the right *source*, not just the right
    topic.
    """
    return sorted((EVAL_DIR / "corpus").glob("*.txt"))


def validate_dataset(dataset, corpus_text):
    """Fail fast if a ground-truth span is not actually in the corpus.

    Without this a typo in the dataset would silently look like a retrieval
    failure and corrupt every number the harness reports.
    """
    haystack = normalise_text(corpus_text)
    problems = []

    for question in dataset["questions"]:
        for span in question["answer_spans"]:
            if normalise_text(span) not in haystack:
                problems.append(f"  {question['id']}: {span[:60]!r}")

    if problems:
        raise SystemExit(
            "Ground-truth spans missing from corpus:\n" + "\n".join(problems)
        )

    log.info("Dataset validated: %d questions", len(dataset["questions"]))


# ── Retrieval metrics ─────────────────────────────────────────────────────

def score_retrieval(question, retrieved_chunks):
    """Return per-question retrieval metrics.

    hit          — at least one required span appeared in the retrieved text
    coverage     — fraction of required spans that appeared
    reciprocal_rank — 1/rank of the first chunk containing any required span
    """
    spans = [normalise_text(s) for s in question["answer_spans"]]
    if not spans:
        return None  # unanswerable questions have nothing to retrieve

    texts = [normalise_text(c.page_content) for c in retrieved_chunks]

    found = {span for span in spans if any(span in text for text in texts)}

    reciprocal_rank = 0.0
    for rank, text in enumerate(texts, start=1):
        if any(span in text for span in spans):
            reciprocal_rank = 1.0 / rank
            break

    return {
        "hit": len(found) > 0,
        "coverage": len(found) / len(spans),
        "reciprocal_rank": reciprocal_rank,
    }


# ── Generation ────────────────────────────────────────────────────────────

def strip_footer(answer):
    """Remove the citation footer the UI appends, leaving the answer text."""
    return answer.split("\n\n---\n📄 **Sources:**")[0].strip()


def generate_answer(question_text, retriever, client, k):
    from src.chat import ask_document_stream

    session = {}
    final = ""
    for partial in ask_document_stream(question_text, [], retriever, client, session):
        final = partial

    context_items = session.get("last_retrieval", [])
    context = "\n\n".join(
        f"[{item['source']} p{item['page']}]\n{item['text']}" for item in context_items
    )
    return strip_footer(final), context


def judge_answer(question, answer, context, client):
    """Ask the judge model to grade faithfulness and correctness."""
    prompt = JUDGE_PROMPT.format(
        question=question["question"],
        ground_truth=question["ground_truth"],
        context=context or "(nothing retrieved)",
        answer=answer,
    )

    response = client.chat.completions.create(
        model=Config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        verdict = json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        log.warning("Judge returned unparseable output for %s", question["id"])
        return {"faithful": False, "correct": False, "reason": "unparseable judge output"}

    return verdict


def looks_like_refusal(answer):
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


# ── Runner ────────────────────────────────────────────────────────────────

def evaluate_mode(mode, chunks, vectorstore, dataset, client, retrieval_only, k):
    from src.retriever import HybridRetriever

    retriever = HybridRetriever(chunks=chunks, vectorstore=vectorstore, mode=mode)

    rows = []
    started = time.time()

    for question in dataset["questions"]:
        row = {"id": question["id"], "type": question["type"]}

        retrieved = retriever.retrieve(question["question"], k=k)
        row["retrieval"] = score_retrieval(question, retrieved)

        if not retrieval_only:
            answer, context = generate_answer(question["question"], retriever, client, k)
            row["answer"] = answer

            if question["answerable"]:
                verdict = judge_answer(question, answer, context, client)
                row["faithful"] = bool(verdict.get("faithful"))
                row["correct"] = bool(verdict.get("correct"))
                row["reason"] = verdict.get("reason", "")
            else:
                # For unanswerable questions the only correct behaviour is to
                # decline. A confident answer here is a hallucination.
                refused = looks_like_refusal(answer)
                row["refused"] = refused
                row["faithful"] = refused
                row["correct"] = refused

        rows.append(row)
        log.info("[%s] %s done", mode, question["id"])

    return {
        "mode": mode,
        "elapsed_seconds": round(time.time() - started, 1),
        "rows": rows,
    }


def summarise(result, retrieval_only):
    rows = result["rows"]
    answerable = [r for r in rows if r["retrieval"] is not None]
    unanswerable = [r for r in rows if r["retrieval"] is None]

    summary = {
        "mode": result["mode"],
        "n_answerable": len(answerable),
        "hit_rate": _mean([r["retrieval"]["hit"] for r in answerable]),
        "span_coverage": _mean([r["retrieval"]["coverage"] for r in answerable]),
        "mrr": _mean([r["retrieval"]["reciprocal_rank"] for r in answerable]),
        "elapsed_seconds": result["elapsed_seconds"],
    }

    if not retrieval_only:
        summary["faithfulness"] = _mean([r["faithful"] for r in answerable])
        summary["correctness"] = _mean([r["correct"] for r in answerable])
        summary["refusal_accuracy"] = _mean([r.get("refused", False) for r in unanswerable])

    return summary


def _mean(values):
    values = list(values)
    return round(sum(float(v) for v in values) / len(values), 4) if values else 0.0


def render_table(summaries, retrieval_only):
    headers = ["Mode", "Hit@k", "Coverage", "MRR"]
    keys = ["hit_rate", "span_coverage", "mrr"]

    if not retrieval_only:
        headers += ["Faithful", "Correct", "Refusal acc."]
        keys += ["faithfulness", "correctness", "refusal_accuracy"]

    headers.append("Seconds")
    keys.append("elapsed_seconds")

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]

    for summary in summaries:
        cells = [f"`{summary['mode']}`"]
        for key in keys:
            value = summary[key]
            cells.append(f"{value:.1f}s" if key == "elapsed_seconds" else f"{value:.1%}")
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DocChat retrieval and generation.")
    parser.add_argument("--modes", nargs="+", default=list(ALL_MODES), choices=ALL_MODES)
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Skip LLM generation and judging (near-free).")
    parser.add_argument("-k", type=int, default=Config.RETRIEVAL_K,
                        help="Chunks retrieved per question.")
    parser.add_argument("--validate-only", action="store_true",
                        help="Check ground-truth spans against the corpus and exit.")
    args = parser.parse_args()

    # Validation needs no API key, so it runs before the key check — a broken
    # dataset should be catchable without spending anything.
    if args.validate_only:
        dataset = load_dataset()
        corpus_text = (EVAL_DIR / dataset["primary_corpus"]).read_text(encoding="utf-8")
        validate_dataset(dataset, corpus_text)
        print(f"All ground-truth spans found. {len(dataset['questions'])} questions OK.")
        print(f"Corpus: {len(load_corpus_files())} documents.")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Export it before running the eval.")

    from openai import OpenAI

    from src.extractor import extract_text
    from src.indexer import build_index

    # The groundedness check adds an LLM call per question and is not part of
    # what we are measuring here — the judge grades faithfulness directly.
    os.environ["DOCCHAT_ENABLE_GROUNDEDNESS_CHECK"] = "0"
    Config.ENABLE_GROUNDEDNESS_CHECK = False

    dataset = load_dataset()
    primary_path = EVAL_DIR / dataset["primary_corpus"]
    validate_dataset(dataset, primary_path.read_text(encoding="utf-8"))

    client = OpenAI(api_key=api_key)

    corpus_files = load_corpus_files()
    log.info("Building index from %d document(s)", len(corpus_files))

    docs = []
    for path in corpus_files:
        docs.extend(extract_text(str(path), path.name))

    retriever, n_chunks = build_index(docs, api_key)
    log.info("Index ready: %d chunks across %d documents", n_chunks, len(corpus_files))

    # Every mode shares one index, so embeddings are paid for exactly once.
    chunks, vectorstore = retriever.chunks, retriever.vectorstore

    results = []
    summaries = []

    for mode in args.modes:
        log.info("=== Evaluating mode: %s ===", mode)
        result = evaluate_mode(
            mode, chunks, vectorstore, dataset, client, args.retrieval_only, args.k
        )
        results.append(result)
        summaries.append(summarise(result, args.retrieval_only))

    table = render_table(summaries, args.retrieval_only)

    print("\n" + "=" * 78)
    print(f"RESULTS  (k={args.k}, {len(dataset['questions'])} questions, "
          f"{n_chunks} chunks)")
    print("=" * 78 + "\n")
    print(table + "\n")

    results_dir = EVAL_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    payload = {
        "run_at": datetime.now(UTC).isoformat(),
        "k": args.k,
        "n_chunks": n_chunks,
        "embedding_model": Config.EMBEDDING_MODEL,
        "llm_model": Config.LLM_MODEL,
        "retrieval_only": args.retrieval_only,
        "summaries": summaries,
        "detail": results,
    }

    out_path = results_dir / "latest.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    (results_dir / "latest.md").write_text(
        f"# Evaluation results\n\n"
        f"Run: {payload['run_at']}  \n"
        f"k={args.k} · {len(dataset['questions'])} questions · {n_chunks} chunks · "
        f"{Config.EMBEDDING_MODEL} / {Config.LLM_MODEL}\n\n"
        f"{table}\n",
        encoding="utf-8",
    )

    print(f"Wrote {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
