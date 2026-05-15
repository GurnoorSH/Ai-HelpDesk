"""
Lightweight RAG evaluation runner.

Provide a JSON file with cases like:
[
  {
    "question": "What is the electronics return window?",
    "expected": "30 days",
    "golden_answer": "Electronics can be returned within 30 days."
  }
]
"""

import argparse
import json
import re
from pathlib import Path

from pydantic import BaseModel

from Rag_Agent import (
    FAST_LLM_MODEL,
    extract_metadata_filter,
    generate_verified_answer,
    llm,
    retrieve,
)


class EvaluationResult(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    reason: str


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def rouge_l_f1(candidate: str, reference: str) -> float:
    """
    Dependency-free ROUGE-L F1. Use BERTScore later if semantic similarity
    becomes more important than lexical overlap for the golden answers.
    """
    candidate_tokens = tokenize(candidate)
    reference_tokens = tokenize(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0

    rows = len(candidate_tokens) + 1
    cols = len(reference_tokens) + 1
    dp = [[0] * cols for _ in range(rows)]

    for i, candidate_token in enumerate(candidate_tokens, start=1):
        for j, reference_token in enumerate(reference_tokens, start=1):
            if candidate_token == reference_token:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[-1][-1]
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    return (2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def judge_case(
    question: str,
    expected: str,
    golden_answer: str,
    generated_answer: str,
    context: str,
) -> EvaluationResult:
    prompt = (
        "Evaluate this RAG result. Score each metric from 0 to 1.\n"
        "- faithfulness: the generated answer only uses facts supported by the retrieved context.\n"
        "- answer_relevancy: the generated answer directly addresses the question.\n"
        "- context_precision: the retrieved context is mostly useful, with little filler or junk.\n"
        "- context_recall: the retrieved context contains all information needed for the expected/golden answer.\n"
        "Return only JSON with keys faithfulness, answer_relevancy, context_precision, "
        "context_recall, reason.\n\n"
        f"Question: {question}\n"
        f"Expected answer facts: {expected}\n"
        f"Golden answer: {golden_answer}\n"
        f"Generated answer: {generated_answer}\n"
        f"Retrieved context:\n{context}"
    )
    response = llm.chat.completions.create(
        model=FAST_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=200,
    )
    return EvaluationResult(**json.loads(response.choices[0].message.content or "{}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality.")
    parser.add_argument("test_set", type=Path, help="Path to a JSON test set.")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip answer generation and only judge retrieved context quality.",
    )
    args = parser.parse_args()

    cases = json.loads(args.test_set.read_text(encoding="utf-8"))
    results: list[EvaluationResult] = []
    rouge_scores: list[float] = []

    for index, case in enumerate(cases, start=1):
        question = case["question"]
        expected = case.get("expected", "")
        golden_answer = case.get("golden_answer") or expected
        passages = retrieve(question, metadata_filter=extract_metadata_filter(question))
        context = "\n\n---\n\n".join(passages)
        generated_answer = (
            "[retrieval-only]"
            if args.retrieval_only
            else generate_verified_answer(question, context, messages=[])
        )
        result = judge_case(question, expected, golden_answer, generated_answer, context)
        results.append(result)
        rouge_score = 0.0 if args.retrieval_only else rouge_l_f1(generated_answer, golden_answer)
        rouge_scores.append(rouge_score)
        print(
            f"{index}. faithfulness={result.faithfulness:.2f} "
            f"answer_relevancy={result.answer_relevancy:.2f} "
            f"context_precision={result.context_precision:.2f} "
            f"context_recall={result.context_recall:.2f} "
            f"rouge_l={rouge_score:.2f} - {result.reason}"
        )

    if results:
        avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
        avg_relevancy = sum(r.answer_relevancy for r in results) / len(results)
        avg_context_precision = sum(r.context_precision for r in results) / len(results)
        avg_context_recall = sum(r.context_recall for r in results) / len(results)
        avg_rouge = sum(rouge_scores) / len(rouge_scores)
        print("\nSummary")
        print(f"cases={len(results)}")
        print(f"faithfulness={avg_faithfulness:.2f}")
        print(f"answer_relevancy={avg_relevancy:.2f}")
        print(f"context_precision={avg_context_precision:.2f}")
        print(f"context_recall={avg_context_recall:.2f}")
        print(f"rouge_l={avg_rouge:.2f}")


if __name__ == "__main__":
    main()
