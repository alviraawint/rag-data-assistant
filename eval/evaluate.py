"""
Evaluation harness for the RAG Data Assistant.

Measures two things:

1. Retrieval quality (no API key needed) — for a hand-labeled question set,
   how often does the retriever surface the gold passage in its top-k results?
   Reports Hit@k, Recall@k, and Mean Reciprocal Rank (MRR).

2. Answer faithfulness (optional, needs ANTHROPIC_API_KEY) — uses Claude as an
   LLM-judge to check whether each generated answer is supported by the
   retrieved context (faithful) and addresses the question (relevant).

Usage:
    python eval/evaluate.py
    python eval/evaluate.py --k 5 --chunk-size 400
    python eval/evaluate.py --judge            # also run faithfulness eval
"""

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from utils.document_loader import DocumentLoader
from utils.retriever import EmbeddingRetriever
from utils.llm_generator import ClaudeAnswerGenerator


# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _chunk_contains(chunk: str, evidence: str) -> bool:
    return _normalize(evidence) in _normalize(chunk)


def evaluate_retrieval(retriever, questions, k):
    """Compute per-question retrieval metrics for the labeled question set."""
    per_question = []

    for item in questions:
        results = retriever.retrieve(item["question"], k=k)
        retrieved_chunks = [chunk for chunk, _ in results]
        evidences = item["evidence"]

        first_relevant_rank = None
        found = set()
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            for evidence in evidences:
                if _chunk_contains(chunk, evidence):
                    found.add(evidence)
                    if first_relevant_rank is None:
                        first_relevant_rank = rank

        per_question.append(
            {
                "id": item["id"],
                "question": item["question"],
                # Recall@k: fraction of gold passages retrieved in the top-k.
                "recall": len(found) / len(evidences),
                # Hit@k: did any gold passage appear in the top-k?
                "hit": 1.0 if first_relevant_rank is not None else 0.0,
                # Reciprocal rank: 1 / position of the first relevant chunk.
                "reciprocal_rank": (
                    1.0 / first_relevant_rank if first_relevant_rank else 0.0
                ),
                "first_relevant_rank": first_relevant_rank,
            }
        )

    return per_question


# --------------------------------------------------------------------------- #
# Faithfulness (LLM-as-judge)
# --------------------------------------------------------------------------- #

JUDGE_SYSTEM = (
    "You are a strict evaluator for a retrieval-augmented generation system. "
    "You are given a question, the retrieved context passages, and a candidate "
    "answer. Judge two things: (1) faithfulness - is every factual claim in the "
    "answer directly supported by the context? (2) relevance - does the answer "
    "address the question? An answer that correctly states the context lacks the "
    "information is both faithful and relevant. Respond with ONLY a JSON object: "
    '{"faithful": true|false, "relevant": true|false, "reason": "<one sentence>"}.'
)


def _parse_judge_json(text: str) -> dict:
    """Best-effort extraction of the judge's JSON verdict."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {"faithful": None, "relevant": None, "reason": "unparseable"}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"faithful": None, "relevant": None, "reason": "unparseable"}


def judge_answer(generator, question, context_chunks, answer):
    """Ask Claude whether `answer` is faithful to and relevant for the context."""
    context = "\n\n".join(
        f"[Chunk {i}] {chunk}" for i, (chunk, _) in enumerate(context_chunks, 1)
    )
    user = (
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        "Respond with JSON only."
    )
    message = generator.client.messages.create(
        model=generator.model,
        max_tokens=400,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    return _parse_judge_json(text)


def evaluate_faithfulness(generator, retriever, questions, k):
    """Generate an answer per question and judge it. Returns per-question rows."""
    per_question = []
    for item in questions:
        results = retriever.retrieve(item["question"], k=k)
        answer = generator.generate_answer(item["question"], results)
        verdict = judge_answer(generator, item["question"], results, answer)
        per_question.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": answer,
                "faithful": verdict.get("faithful"),
                "relevant": verdict.get("relevant"),
                "reason": verdict.get("reason", ""),
            }
        )
    return per_question


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def print_retrieval_report(per_question, k):
    print(f"\n=== Retrieval metrics (k={k}) ===")
    print(f"{'ID':>3}  {'Hit':>4}  {'Recall':>6}  {'Rank':>5}  Question")
    for row in per_question:
        rank = row["first_relevant_rank"] if row["first_relevant_rank"] else "-"
        print(
            f"{row['id']:>3}  {row['hit']:>4.0f}  {row['recall']:>6.2f}  "
            f"{str(rank):>5}  {row['question']}"
        )

    hit_rate = mean(r["hit"] for r in per_question)
    recall = mean(r["recall"] for r in per_question)
    mrr = mean(r["reciprocal_rank"] for r in per_question)
    print("\n--- Aggregate ---")
    print(f"Hit@{k}     : {hit_rate:.3f}")
    print(f"Recall@{k}  : {recall:.3f}")
    print(f"MRR        : {mrr:.3f}")
    return {"hit_rate": hit_rate, "recall": recall, "mrr": mrr}


def print_faithfulness_report(per_question):
    print("\n=== Answer faithfulness (LLM-as-judge) ===")
    print(f"{'ID':>3}  {'Faith':>5}  {'Relev':>5}  Reason")
    for row in per_question:
        faith = "yes" if row["faithful"] else ("no" if row["faithful"] is False else "?")
        relev = "yes" if row["relevant"] else ("no" if row["relevant"] is False else "?")
        print(f"{row['id']:>3}  {faith:>5}  {relev:>5}  {row['reason']}")

    scored = [r for r in per_question if r["faithful"] is not None]
    faithful_rate = mean(1.0 if r["faithful"] else 0.0 for r in scored) if scored else 0.0
    relevant_rate = mean(1.0 if r["relevant"] else 0.0 for r in scored) if scored else 0.0
    print("\n--- Aggregate ---")
    print(f"Faithful rate : {faithful_rate:.3f}")
    print(f"Relevant rate : {relevant_rate:.3f}")
    return {"faithful_rate": faithful_rate, "relevant_rate": relevant_rate}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--k", type=int, default=3, help="Top-k chunks to retrieve.")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).parent / "qa_dataset.json"),
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Also run LLM-as-judge faithfulness eval (needs ANTHROPIC_API_KEY).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write a JSON results file.",
    )
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    questions = dataset["questions"]

    document_path = REPO_ROOT / dataset["document"]
    text = document_path.read_text(encoding="utf-8")

    print(f"Document : {dataset['document']}")
    print(f"Questions: {len(questions)}")

    loader = DocumentLoader(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    chunks = loader.chunk_text(text)
    print(f"Chunks   : {len(chunks)} (chunk_size={args.chunk_size})")

    retriever = EmbeddingRetriever()
    retriever.build_index(chunks)

    retrieval_rows = evaluate_retrieval(retriever, questions, args.k)
    retrieval_summary = print_retrieval_report(retrieval_rows, args.k)

    faithfulness_rows = None
    faithfulness_summary = None
    if args.judge:
        if not ClaudeAnswerGenerator.is_available():
            print(
                "\n[skip] Faithfulness eval needs ANTHROPIC_API_KEY "
                "(and the anthropic package). Skipping."
            )
        else:
            generator = ClaudeAnswerGenerator()
            faithfulness_rows = evaluate_faithfulness(
                generator, retriever, questions, args.k
            )
            faithfulness_summary = print_faithfulness_report(faithfulness_rows)

    if args.output:
        payload = {
            "config": {
                "k": args.k,
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
                "num_chunks": len(chunks),
            },
            "retrieval": {"per_question": retrieval_rows, "summary": retrieval_summary},
        }
        if faithfulness_rows is not None:
            payload["faithfulness"] = {
                "per_question": faithfulness_rows,
                "summary": faithfulness_summary,
            }
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
