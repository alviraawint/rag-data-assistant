# Evaluation Harness

Most RAG demos stop at "it returns something." This harness **measures** the
pipeline, so changes (chunk size, embedding model, normalization, reranking) can
be compared with numbers instead of vibes.

## What it measures

### 1. Retrieval quality — no API key needed
For each question in [`qa_dataset.json`](qa_dataset.json) we know the gold
passage (the `evidence` text). We retrieve the top-k chunks and check whether the
gold passage is among them.

| Metric | Definition |
|--------|------------|
| **Hit@k** | Fraction of questions where a gold passage appears in the top-k. |
| **Recall@k** | Fraction of gold passages retrieved in the top-k (per question, then averaged). |
| **MRR** | Mean Reciprocal Rank — average of `1 / rank` of the first relevant chunk. Rewards ranking the right chunk higher. |

Evidence is matched case-insensitively after whitespace normalization, so the
labels stay valid even if you change `--chunk-size`.

### 2. Answer faithfulness — optional, uses Claude as judge
With `--judge` and an `ANTHROPIC_API_KEY`, each generated answer is scored by an
independent Claude call (LLM-as-judge):

| Metric | Definition |
|--------|------------|
| **Faithful rate** | Fraction of answers whose every claim is supported by the retrieved context (hallucination check). |
| **Relevant rate** | Fraction of answers that actually address the question. |

## Running it

```bash
# Retrieval metrics only (fast, no key required)
python eval/evaluate.py

# Sweep a parameter to compare configurations
python eval/evaluate.py --k 5
python eval/evaluate.py --chunk-size 300
python eval/evaluate.py --chunk-size 800

# Add faithfulness judging (requires ANTHROPIC_API_KEY)
python eval/evaluate.py --judge

# Persist a JSON report
python eval/evaluate.py --judge --output eval/results.json
```

## Using it to drive improvements
This harness is the measurement tool for the next pipeline changes. For example,
switching the index from L2 to cosine similarity, or adding a cross-encoder
reranker, should move **MRR** and **Hit@k** — run before and after and report the
delta. That before/after number is the resume bullet.

## Extending the dataset
Add objects to `questions` in `qa_dataset.json`:

```json
{ "id": 11, "question": "...", "evidence": ["exact phrase from the document"] }
```

Keep `evidence` to a distinctive phrase that lives in a single sentence so it
falls inside one chunk.
