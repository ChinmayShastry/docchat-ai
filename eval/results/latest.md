# Evaluation results

Run: 2026-08-14T10:36:07.921072+00:00  
k=4 · 26 questions · 33 chunks · text-embedding-3-small / gpt-4o-mini

| Mode | Hit@k | Coverage | MRR | Faithful | Correct | Refusal acc. | Seconds |
|---|---|---|---|---|---|---|---|
| `bm25` | 95.5% | 93.2% | 71.2% | 100.0% | 86.4% | 100.0% | 66.2s |
| `semantic` | 86.4% | 86.4% | 59.9% | 100.0% | 81.8% | 100.0% | 88.4s |
| `hybrid` | 100.0% | 97.7% | 78.0% | 100.0% | 95.5% | 100.0% | 86.8s |
| `hybrid_rerank` | 95.5% | 93.2% | 76.9% | 100.0% | 86.4% | 100.0% | 199.0s |
