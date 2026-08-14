# 📄 DocChat AI

**Retrieval-augmented document chat with source citations and a measured retrieval pipeline.**

Upload PDFs, Word documents, spreadsheets or CSVs, get a structured summary, then ask
questions answered *only* from the uploaded content — with inline citations, a visible
view of the retrieved context, and a warning when the model strays beyond its sources.

[![CI](https://github.com/ChinmayShastry/docchat-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/ChinmayShastry/docchat-ai/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Why this project exists

Most RAG demos stop at "embed, retrieve top-k, prompt". That works on clean text and
fails quietly everywhere else. This project is an attempt to build the parts that
usually get skipped:

- **Retrieval that combines two signals** — keyword (BM25) and semantic (embeddings),
  blended and then reranked by a cross-encoder, because each catches what the other misses.
- **An evaluation harness that produces numbers**, so retrieval changes can be justified
  with measurements instead of intuition.
- **Ingestion that survives real documents** — a fallback chain of four PDF parsers,
  table extraction from Word files, per-sheet handling for Excel.
- **Honest failure modes** — the system refuses when the answer isn't in the documents,
  and flags its own unsupported claims.

---

## Measured results

Retrieval strategies evaluated on a 26-question set over a synthetic annual report,
where each question declares the exact text spans required to answer it. Retrieval
metrics are deterministic span matching; generation metrics use an LLM judge.

<!-- EVAL_RESULTS_START -->
*Populate by running `python eval/run_eval.py` — see [Evaluation](#evaluation).*
<!-- EVAL_RESULTS_END -->

**Metrics**

| Metric | Meaning |
|---|---|
| `Hit@k` | At least one required span appeared in the retrieved chunks |
| `Coverage` | Fraction of required spans retrieved (multi-fact questions need several) |
| `MRR` | Mean reciprocal rank of the first correct chunk — rewards ranking, not just recall |
| `Faithful` | Every claim in the answer is supported by the retrieved context |
| `Correct` | The answer matches the reference answer |
| `Refusal acc.` | On the 4 deliberately unanswerable questions, the system declined instead of inventing |

---

## Architecture

```
Upload (PDF / DOCX / TXT / CSV / XLSX)
   │
   ├─ Extraction ......... per-format handlers; PDFs try PyPDF → PyMuPDF → pdfplumber → OCR
   │
   ├─ Chunking ........... semantic (small docs) or fixed-size (large docs), with fallbacks
   │
   ├─ Indexing ........... OpenAI embeddings → Chroma, cached on disk by content hash
   │
   ├─ Retrieval .......... BM25 ─┐
   │                              ├─ normalise → α-blend → threshold → cross-encoder rerank
   │                   embeddings ┘
   │
   └─ Generation ......... labelled context → streamed answer → citations → groundedness check
```

### Retrieval, specifically

Both arms score the corpus, scores are min-max normalised onto a comparable scale, then
blended as `α · semantic + (1 − α) · bm25`. Chunks are identified by a **content hash**,
not object identity — vector stores return freshly constructed objects on every query, so
identity-based lookups silently drop candidates from the other arm. Survivors above the
score threshold go to a cross-encoder, which scores each `(query, chunk)` pair jointly
rather than comparing pre-computed vectors.

A floor on the candidate pool means an aggressive threshold degrades recall rather than
returning nothing.

---

## Quick start

```bash
git clone https://github.com/ChinmayShastry/docchat-ai.git
cd docchat-ai
pip install -r requirements.txt
```

Provide an OpenAI API key either in `.env` (copy from `.env.example`) or by typing it into
the app sidebar — the sidebar takes precedence.

```bash
streamlit run app.py
```

### Docker

```bash
docker build -t docchat-ai .
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... docchat-ai
```

The reranker weights are baked into the image so the first request doesn't wait on a
model download.

---

## Evaluation

```bash
python eval/run_eval.py --validate-only   # check ground truth, no API key needed
python eval/run_eval.py --retrieval-only  # retrieval metrics; embeddings cost only
python eval/run_eval.py                   # full run, adds generation + LLM judge
python eval/run_eval.py --modes hybrid_rerank -k 6
```

All four retrieval modes share a single index, so embeddings are paid for once per run.
Results are written to `eval/results/latest.json` and `latest.md`.

The dataset (`eval/dataset.json`) validates its own ground-truth spans against the corpus
before running — a typo in the dataset would otherwise look like a retrieval failure and
quietly corrupt every number.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

68 tests, fully mocked — no API calls, no model downloads, so CI runs free on every push.
Coverage focuses on the logic most likely to break silently: score blending and candidate
filtering, the PDF parser fallback chain, chunking fallbacks, per-format extraction,
history normalisation, and the groundedness verdict parser.

---

## Configuration

Every value in `config.py` can be overridden by an environment variable with a `DOCCHAT_`
prefix, which is how the eval harness sweeps retrieval strategies without editing source.

| Setting | Default | Notes |
|---|---|---|
| `DOCCHAT_RETRIEVAL_MODE` | `hybrid_rerank` | `bm25` · `semantic` · `hybrid` · `hybrid_rerank` |
| `DOCCHAT_RETRIEVAL_K` | `4` | Chunks sent to the LLM |
| `DOCCHAT_HYBRID_ALPHA` | `0.5` | 0 = pure keyword, 1 = pure semantic |
| `DOCCHAT_CHUNK_SIZE` | `600` | Fixed-chunking size |
| `DOCCHAT_MAX_CONTEXT_CHARS` | `12000` | Hard cap on context sent to the model |
| `DOCCHAT_MAX_UPLOAD_MB` | `25` | Per-file upload limit |
| `DOCCHAT_ENABLE_INDEX_CACHE` | `1` | Reuse an index when content and settings match |
| `DOCCHAT_LOG_LEVEL` | `INFO` | |

---

## Known limitations

- **OCR is not available in most deployments.** The code path exists, but it needs
  Tesseract and Poppler installed; scanned PDFs will fail on a stock Streamlit Cloud host.
- **The index cache is local disk.** Fine for a single container, wrong for a
  horizontally-scaled deployment — that needs a shared vector store.
- **Query classification is keyword-based.** It is transparent and free, but "how does X
  stack up against Y" won't be recognised as a comparison. An LLM classifier would add a
  round-trip to every question to change how many chunks get fetched.
- **The groundedness check costs an extra LLM call per answer** and is itself an LLM
  judgement, so it reduces rather than eliminates hallucination risk.
- **Multi-fact questions remain the weak point.** Retrieving one supporting span is much
  easier than retrieving all of them — which is why `Coverage` is reported separately
  from `Hit@k`.
- **The eval corpus is synthetic.** It gives exact ground truth, but real documents are
  messier; these numbers are a floor for comparing strategies, not a claim about
  production accuracy.

---

## Project structure

```
docchat-ai/
├── app.py                    Streamlit UI
├── config.py                 Central settings, env-overridable
├── src/
│   ├── extractor.py          Per-format text extraction + upload validation
│   ├── chunking.py           Semantic / fixed chunking with fallbacks
│   ├── indexer.py            Embedding, Chroma persistence, content-hash cache
│   ├── retriever.py          BM25 + semantic blending, cross-encoder reranking
│   ├── summarizer.py         Direct and Map-Reduce summarisation
│   ├── chat.py               Context assembly, streaming, groundedness check
│   └── logging_setup.py      Central logging config
├── tests/                    68 mocked unit tests
├── eval/
│   ├── corpus/               Synthetic evaluation document
│   ├── dataset.json          26 questions with ground-truth spans
│   └── run_eval.py           Retrieval + generation harness
└── .github/workflows/ci.yml  Lint, tests, dependency resolution
```

---

## License

MIT — see [LICENSE](LICENSE).
