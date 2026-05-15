"""
RAG evaluation runner.

Writes timestamped JSON reports under reports/ by default and keeps the
dependency-light LLM judge as the fallback evaluator. If RAGAS dependencies are
installed, --ragas also adds RAGAS metrics using Groq for judge calls and local
FastEmbed embeddings for embedding-based metrics.
"""

import argparse
import json
import math
import os
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel

from Rag_Agent import (
    FAST_LLM_MODEL,
    LLM_BASE_URL,
    QDRANT_URL,
    extract_metadata_filter,
    generate_verified_answer,
    llm,
    qdrant,
    retrieve,
)
from observability import (
    PRICE_CONFIG,
    current_usage_report,
    record_llm_response,
    usage_run,
)


REPORTS_DIR = Path("reports")


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
    record_llm_response("evaluator", FAST_LLM_MODEL, response)
    return EvaluationResult(**json.loads(response.choices[0].message.content or "{}"))


def average(values: list[float]) -> float:
    return mean(values) if values else 0.0


def summarize_cases(case_reports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(case_reports),
        "faithfulness": average([case["metrics"]["faithfulness"] for case in case_reports]),
        "answer_relevancy": average([case["metrics"]["answer_relevancy"] for case in case_reports]),
        "context_precision": average([case["metrics"]["context_precision"] for case in case_reports]),
        "context_recall": average([case["metrics"]["context_recall"] for case in case_reports]),
        "rouge_l": average([case["metrics"]["rouge_l"] for case in case_reports]),
    }


def summarize_usage(case_reports: list[dict[str, Any]]) -> dict[str, Any]:
    case_usage = [case.get("usage") or {} for case in case_reports]
    records = [
        record
        for usage in case_usage
        for record in usage.get("records", [])
    ]
    total_known_cost = sum(float(usage.get("known_cost_usd") or 0.0) for usage in case_usage)
    unknown_cost_stages = sorted(
        {
            stage
            for usage in case_usage
            for stage in usage.get("unknown_cost_stages", [])
        }
    )
    by_stage: dict[str, dict[str, Any]] = {}
    for usage in case_usage:
        for stage_name, stage in (usage.get("by_stage") or {}).items():
            target = by_stage.setdefault(
                stage_name,
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "unknown_cost": False,
                    "calls": 0,
                },
            )
            target["input_tokens"] += int(stage.get("input_tokens") or 0)
            target["output_tokens"] += int(stage.get("output_tokens") or 0)
            target["total_tokens"] += int(stage.get("total_tokens") or 0)
            target["cost_usd"] += float(stage.get("cost_usd") or 0.0)
            target["calls"] += int(stage.get("calls") or 0)
            target["unknown_cost"] = bool(target["unknown_cost"] or stage.get("unknown_cost"))

    return {
        "total_input_tokens": sum(int(record.get("input_tokens") or 0) for record in records),
        "total_output_tokens": sum(int(record.get("output_tokens") or 0) for record in records),
        "total_tokens": sum(int(record.get("total_tokens") or 0) for record in records),
        "known_cost_usd": total_known_cost,
        "unknown_cost_stages": unknown_cost_stages,
        "by_stage": by_stage,
    }


def _ragas_metric_imports() -> tuple[Any, list[Any]]:
    from ragas import evaluate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    return evaluate, [faithfulness, answer_relevancy, context_precision, context_recall]


def _ragas_embeddings() -> Any:
    try:
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    except Exception:
        from langchain_community.embeddings import FastEmbedEmbeddings

    class NamedFastEmbedEmbeddings:
        """
        RAGAS 0.4 records embedding usage and expects `.model` to be a string.
        LangChain's FastEmbed wrapper uses `.model` for the TextEmbedding object,
        so this adapter keeps both RAGAS accounting and embedding calls happy.
        """

        def __init__(self, model_name: str):
            self.model = model_name
            self.model_name = model_name
            self._embeddings = FastEmbedEmbeddings(model_name=model_name)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._embeddings.embed_documents(texts)

        def embed_query(self, text: str) -> list[float]:
            return self._embeddings.embed_query(text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.embed_documents(texts)

        async def aembed_query(self, text: str) -> list[float]:
            return self.embed_query(text)

    return NamedFastEmbedEmbeddings(qdrant.embedding_model_name)


def _groq_chat_model(chat_groq: Any, model: str) -> Any:
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    kwargs: dict[str, Any] = {"model": model, "api_key": api_key, "temperature": 0}
    if "api.groq.com" not in LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL
    try:
        return chat_groq(**kwargs)
    except TypeError:
        fallback_kwargs: dict[str, Any] = {"model_name": model, "groq_api_key": api_key, "temperature": 0}
        if "api.groq.com" not in LLM_BASE_URL:
            fallback_kwargs["groq_api_base"] = LLM_BASE_URL
        return chat_groq(**fallback_kwargs)


def finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def run_ragas(case_reports: list[dict[str, Any]], model: str) -> dict[str, Any]:
    """
    Optional RAGAS pass. All imports live inside this function so the lightweight
    evaluator and dashboard still work without the heavier optional packages.
    """
    try:
        from datasets import Dataset
        from langchain_groq import ChatGroq

        evaluate, metrics = _ragas_metric_imports()
    except Exception as e:
        return {"enabled": False, "status": "unavailable", "error": str(e), "summary": {}, "cases": []}

    try:
        dataset = Dataset.from_list(
            [
                {
                    "question": case["question"],
                    "answer": case["generated_answer"],
                    "contexts": case["passages"],
                    "ground_truth": case["golden_answer"],
                    "user_input": case["question"],
                    "response": case["generated_answer"],
                    "retrieved_contexts": case["passages"],
                    "reference": case["golden_answer"],
                }
                for case in case_reports
            ]
        )
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=_groq_chat_model(ChatGroq, model),
            embeddings=_ragas_embeddings(),
        )
        rows = result.to_pandas().to_dict(orient="records")
        summary: dict[str, float] = {}
        for key in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            values = [
                numeric
                for row in rows
                for numeric in [finite_float(row.get(key))]
                if numeric is not None
            ]
            summary[key] = average(values)
        status = "ok" if any(finite_float(row.get(key)) is not None for row in rows for key in summary) else "failed"
        error = "" if status == "ok" else "RAGAS returned no finite metric values."
        return {"enabled": True, "status": status, "error": error, "summary": summary, "cases": rows}
    except Exception as e:
        return {"enabled": True, "status": "failed", "error": str(e), "summary": {}, "cases": []}


def write_report(report: dict[str, Any], reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / f"{report['run_id']}.json"
    output_path.write_text(json.dumps(sanitize_json(report), indent=2, allow_nan=False), encoding="utf-8")
    return output_path


def sanitize_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality.")
    parser.add_argument("test_set", type=Path, help="Path to a JSON test set.")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip answer generation and only judge retrieved context quality.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory for timestamped JSON reports.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Print results only and skip writing a JSON report.",
    )
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="Also run optional RAGAS metrics if dependencies are installed.",
    )
    parser.add_argument(
        "--ragas-model",
        default=os.getenv("RAGAS_MODEL", FAST_LLM_MODEL),
        help="Groq model to use for optional RAGAS judge calls.",
    )
    args = parser.parse_args()

    cases = json.loads(args.test_set.read_text(encoding="utf-8-sig"))
    case_reports: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        question = case["question"]
        expected = case.get("expected", "")
        golden_answer = case.get("golden_answer") or expected
        with usage_run(f"eval_case_{index}"):
            passages = retrieve(question, metadata_filter=extract_metadata_filter(question))
            context = "\n\n---\n\n".join(passages)
            generated_answer = (
                "[retrieval-only]"
                if args.retrieval_only
                else generate_verified_answer(question, context, messages=[])
            )
            result = judge_case(question, expected, golden_answer, generated_answer, context)
            usage_report = current_usage_report()
        rouge_score = 0.0 if args.retrieval_only else rouge_l_f1(generated_answer, golden_answer)
        result_metrics = result.model_dump(exclude={"reason"}) if hasattr(result, "model_dump") else result.dict(
            exclude={"reason"}
        )
        metrics = {
            **result_metrics,
            "rouge_l": rouge_score,
        }
        case_report = {
            "index": index,
            "question": question,
            "expected": expected,
            "golden_answer": golden_answer,
            "generated_answer": generated_answer,
            "passages": passages,
            "context": context,
            "metrics": metrics,
            "reason": result.reason,
            "tags": case.get("tags", []),
            "should_answer": case.get("should_answer"),
            "usage": usage_report,
        }
        case_reports.append(case_report)
        print(
            f"{index}. faithfulness={metrics['faithfulness']:.2f} "
            f"answer_relevancy={metrics['answer_relevancy']:.2f} "
            f"context_precision={metrics['context_precision']:.2f} "
            f"context_recall={metrics['context_recall']:.2f} "
            f"rouge_l={metrics['rouge_l']:.2f} - {result.reason}"
        )

    summary = summarize_cases(case_reports)
    usage_summary = summarize_usage(case_reports)
    ragas_report = run_ragas(case_reports, args.ragas_model) if args.ragas else {
        "enabled": False,
        "status": "not_requested",
        "error": "",
        "summary": {},
        "cases": [],
    }
    run_id = datetime.now(timezone.utc).strftime("rag_eval_%Y%m%d_%H%M%S")
    report = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_set": str(args.test_set),
        "retrieval_only": args.retrieval_only,
        "qdrant_url": QDRANT_URL,
        "judge_model": FAST_LLM_MODEL,
        "ragas": ragas_report,
        "summary": summary,
        "usage": usage_summary,
        "pricing": {
            "source": "GROQ_MODEL_PRICES_JSON",
            "configured_models": sorted(PRICE_CONFIG.keys()),
        },
        "cases": case_reports,
    }

    if case_reports:
        print("\nSummary")
        for key, value in summary.items():
            print(f"{key}={value:.2f}" if isinstance(value, float) else f"{key}={value}")
        print(f"tokens={usage_summary['total_tokens']}")
        print(f"known_cost_usd={usage_summary['known_cost_usd']:.6f}")
        if usage_summary["unknown_cost_stages"]:
            print(f"unknown_cost_stages={','.join(usage_summary['unknown_cost_stages'])}")
        if ragas_report["status"] == "ok":
            print("ragas_status=ok")
        elif args.ragas:
            print(f"ragas_status={ragas_report['status']} error={ragas_report['error']}")

    if not args.no_report:
        output_path = write_report(report, args.reports_dir)
        print(f"\nWrote report: {output_path}")


if __name__ == "__main__":
    main()
