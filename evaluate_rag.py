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
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel

from Rag_Agent import (
    FAST_LLM_MODEL,
    FINAL_LLM_MODEL,
    LLM_BASE_URL,
    QDRANT_URL,
    RERANK_TOP_N,
    extract_metadata_filter,
    generate_verified_answer,
    llm,
    qdrant,
    retrieve_structured,
)
from observability import (
    PRICE_CONFIG,
    current_usage_report,
    record_llm_response,
    usage_run,
)


REPORTS_DIR = Path("reports")
RETRIEVAL_EVAL_TOP_K = 5
GENERATION_METRIC_KEYS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "rouge_l",
]
RETRIEVAL_METRIC_KEYS = [
    "hit_at_5",
    "mrr",
    "id_context_precision",
    "id_context_recall",
]
SUMMARY_METRIC_KEYS = GENERATION_METRIC_KEYS + RETRIEVAL_METRIC_KEYS


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


def coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return max(0.0, min(1.0, score))


def parse_evaluation_result(content: str) -> EvaluationResult:
    payload = json.loads(content or "{}")
    reason = payload.get("reason", "")
    if not isinstance(reason, str):
        reason = json.dumps(reason, ensure_ascii=False)
    return EvaluationResult(
        faithfulness=coerce_score(payload.get("faithfulness")),
        answer_relevancy=coerce_score(payload.get("answer_relevancy")),
        context_precision=coerce_score(payload.get("context_precision")),
        context_recall=coerce_score(payload.get("context_recall")),
        reason=reason,
    )


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
    return parse_evaluation_result(response.choices[0].message.content or "{}")


def average(values: list[float]) -> float:
    return mean(values) if values else 0.0


def average_optional(values: list[Any]) -> float | None:
    finite_values = [
        numeric
        for value in values
        for numeric in [finite_float(value)]
        if numeric is not None
    ]
    return average(finite_values) if finite_values else None


def summarize_cases(case_reports: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "cases": len(case_reports),
        "case_errors": sum(1 for case in case_reports if case.get("error")),
    }
    for key in SUMMARY_METRIC_KEYS:
        summary[key] = average_optional([case.get("metrics", {}).get(key) for case in case_reports])
    return summary


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


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[index]


def summarize_latency(case_reports: list[dict[str, Any]]) -> dict[str, float]:
    durations = [
        float(usage.get("duration_ms"))
        for case in case_reports
        for usage in [case.get("usage") or {}]
        if finite_float(usage.get("duration_ms")) is not None
    ]
    return {
        "average_case_duration_ms": average(durations),
        "p50_case_duration_ms": percentile(durations, 50),
        "p95_case_duration_ms": percentile(durations, 95),
        "max_case_duration_ms": max(durations) if durations else 0.0,
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


def normalize_reference_chunks(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def deterministic_retrieval_metrics(
    retrieval_results: list[dict[str, Any]],
    reference_chunks: list[str],
    *,
    top_k: int = RETRIEVAL_EVAL_TOP_K,
) -> dict[str, float | None]:
    if not reference_chunks:
        return {
            "hit_at_5": None,
            "mrr": None,
            "id_context_precision": None,
            "id_context_recall": None,
        }

    expected = set(reference_chunks)
    retrieved_ids = [str(result.get("chunk_id", "")).strip() for result in retrieval_results if result.get("chunk_id")]
    top_k_ids = retrieved_ids[:top_k]
    first_match_rank = next(
        (rank for rank, chunk_id in enumerate(retrieved_ids, start=1) if chunk_id in expected),
        None,
    )
    retrieved_matches = {chunk_id for chunk_id in retrieved_ids if chunk_id in expected}

    return {
        "hit_at_5": 1.0 if any(chunk_id in expected for chunk_id in top_k_ids) else 0.0,
        "mrr": (1.0 / first_match_rank) if first_match_rank else 0.0,
        "id_context_precision": (len(retrieved_matches) / len(retrieved_ids)) if retrieved_ids else 0.0,
        "id_context_recall": len(retrieved_matches) / len(expected),
    }


def threshold_value(args: argparse.Namespace, name: str) -> float | None:
    return finite_float(getattr(args, name, None))


def evaluate_thresholds(
    summary: dict[str, Any],
    usage_summary: dict[str, Any],
    latency_summary: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    checks = [
        ("min_faithfulness", "faithfulness", ">=", summary.get("faithfulness")),
        ("min_answer_relevancy", "answer_relevancy", ">=", summary.get("answer_relevancy")),
        ("min_context_recall", "context_recall", ">=", summary.get("context_recall")),
        ("min_hit_at_5", "hit_at_5", ">=", summary.get("hit_at_5")),
        ("min_mrr", "mrr", ">=", summary.get("mrr")),
        ("max_average_latency_ms", "average_case_duration_ms", "<=", latency_summary.get("average_case_duration_ms")),
        ("max_known_cost_usd", "known_cost_usd", "<=", usage_summary.get("known_cost_usd")),
    ]
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if args.fail_on_threshold and int(summary.get("case_errors") or 0) > 0:
        case_error_check = {
            "argument": "fail_on_threshold",
            "metric": "case_errors",
            "operator": "==",
            "threshold": 0,
            "actual": int(summary["case_errors"]),
            "passed": False,
        }
        results.append(case_error_check)
        failures.append(case_error_check)
    for arg_name, metric_name, operator, actual_value in checks:
        limit = threshold_value(args, arg_name)
        if limit is None:
            continue
        actual = finite_float(actual_value)
        passed = actual is not None and (actual >= limit if operator == ">=" else actual <= limit)
        row = {
            "argument": arg_name,
            "metric": metric_name,
            "operator": operator,
            "threshold": limit,
            "actual": actual,
            "passed": passed,
        }
        results.append(row)
        if not passed:
            failures.append(row)
    return {
        "enabled": bool(args.fail_on_threshold),
        "passed": not failures,
        "checks": results,
        "failures": failures,
    }


def format_metric(value: Any) -> str:
    numeric = finite_float(value)
    return "n/a" if numeric is None else f"{numeric:.2f}"


def ragas_case_payload(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": case["question"],
        "answer": case["generated_answer"],
        "contexts": case["passages"],
        "ground_truth": case["golden_answer"],
        "user_input": case["question"],
        "response": case["generated_answer"],
        "retrieved_contexts": case["passages"],
        "reference": case["golden_answer"],
    }


def run_ragas(case_reports: list[dict[str, Any]], model: str, sleep_seconds: float = 65.0) -> dict[str, Any]:
    """
    Optional RAGAS pass. All imports live inside this function so the lightweight
    evaluator and dashboard still work without the heavier optional packages.
    """
    try:
        from datasets import Dataset
        from ragas.run_config import RunConfig
        from langchain_groq import ChatGroq

        evaluate, metrics = _ragas_metric_imports()
    except Exception as e:
        return {"enabled": False, "status": "unavailable", "error": str(e), "summary": {}, "cases": []}

    try:
        rows: list[dict[str, Any]] = []
        ragas_llm = _groq_chat_model(ChatGroq, model)
        ragas_embeddings = _ragas_embeddings()
        run_config = RunConfig(timeout=180, max_retries=2, max_wait=60, max_workers=1)
        for index, case in enumerate(case_reports, start=1):
            dataset = Dataset.from_list([ragas_case_payload(case)])
            result = evaluate(
                dataset,
                metrics=metrics,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                run_config=run_config,
                batch_size=1,
                raise_exceptions=False,
            )
            rows.extend(result.to_pandas().to_dict(orient="records"))
            print(f"ragas {index}/{len(case_reports)} complete")
            if index < len(case_reports) and sleep_seconds > 0:
                time.sleep(sleep_seconds)
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
        default=os.getenv("RAGAS_MODEL", FINAL_LLM_MODEL),
        help="Groq model to use for optional RAGAS judge calls.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N cases. Use 0 for all cases.",
    )
    parser.add_argument(
        "--case-sleep",
        type=float,
        default=float(os.getenv("EVAL_CASE_SLEEP_SECONDS", "30")),
        help="Seconds to sleep between local evaluator cases to respect API rate limits.",
    )
    parser.add_argument(
        "--ragas-sleep",
        type=float,
        default=float(os.getenv("RAGAS_CASE_SLEEP_SECONDS", "65")),
        help="Seconds to sleep between one-case RAGAS evaluations.",
    )
    parser.add_argument("--min-faithfulness", type=float, default=None, help="Minimum run faithfulness score.")
    parser.add_argument("--min-answer-relevancy", type=float, default=None, help="Minimum answer relevancy score.")
    parser.add_argument("--min-context-recall", type=float, default=None, help="Minimum LLM-judged context recall score.")
    parser.add_argument("--min-hit-at-5", type=float, default=None, help="Minimum deterministic Hit@5 score.")
    parser.add_argument("--min-mrr", type=float, default=None, help="Minimum deterministic MRR score.")
    parser.add_argument(
        "--max-average-latency-ms",
        type=float,
        default=None,
        help="Maximum average case duration in milliseconds.",
    )
    parser.add_argument("--max-known-cost-usd", type=float, default=None, help="Maximum known eval run cost in USD.")
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="Exit with status 1 when any configured threshold fails.",
    )
    args = parser.parse_args()

    cases = json.loads(args.test_set.read_text(encoding="utf-8-sig"))
    if args.limit > 0:
        cases = cases[:args.limit]
    case_reports: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        question = case["question"]
        expected = case.get("expected", "")
        golden_answer = case.get("golden_answer") or expected
        reference_chunks = normalize_reference_chunks(case.get("reference_chunks"))
        with usage_run(f"eval_case_{index}"):
            retrieval_results = retrieve_structured(
                question,
                metadata_filter=extract_metadata_filter(question),
                top_n=max(RETRIEVAL_EVAL_TOP_K, RERANK_TOP_N),
                compress=not args.retrieval_only,
            )
            passages = [
                result["formatted_context"]
                for result in retrieval_results[:RERANK_TOP_N]
            ]
            context = "\n\n---\n\n".join(passages)
            result: EvaluationResult | None = None
            case_error = ""
            generated_answer = "[generation-error]"
            if args.retrieval_only:
                generated_answer = "[retrieval-only]"
            else:
                try:
                    generated_answer = generate_verified_answer(question, context, messages=[])
                    result = judge_case(question, expected, golden_answer, generated_answer, context)
                except Exception as e:
                    case_error = f"{type(e).__name__}: {e}"
            usage_report = current_usage_report()
        rouge_score = None if args.retrieval_only else rouge_l_f1(generated_answer, golden_answer)
        if result is None:
            result_metrics = {
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
                "context_recall": None,
            }
            reason = (
                "Retrieval-only run skipped answer generation and the LLM judge."
                if args.retrieval_only
                else f"Case failed before judge metrics could be recorded: {case_error}"
            )
        else:
            result_metrics = result.model_dump(exclude={"reason"}) if hasattr(result, "model_dump") else result.dict(
                exclude={"reason"}
            )
            reason = result.reason
        retrieval_metrics = deterministic_retrieval_metrics(retrieval_results, reference_chunks)
        metrics = {
            **result_metrics,
            **retrieval_metrics,
            "rouge_l": rouge_score,
        }
        case_report = {
            "index": index,
            "question": question,
            "expected": expected,
            "golden_answer": golden_answer,
            "generated_answer": generated_answer,
            "passages": passages,
            "retrieval_results": retrieval_results,
            "context": context,
            "metrics": metrics,
            "reason": reason,
            "error": case_error,
            "tags": case.get("tags", []),
            "should_answer": case.get("should_answer"),
            "case_type": case.get("case_type"),
            "reviewed": case.get("reviewed"),
            "reviewer": case.get("reviewer"),
            "reference_chunks": reference_chunks,
            "usage": usage_report,
        }
        case_reports.append(case_report)
        print(
            f"{index}. faithfulness={format_metric(metrics['faithfulness'])} "
            f"answer_relevancy={format_metric(metrics['answer_relevancy'])} "
            f"context_precision={format_metric(metrics['context_precision'])} "
            f"context_recall={format_metric(metrics['context_recall'])} "
            f"hit_at_5={format_metric(metrics['hit_at_5'])} "
            f"mrr={format_metric(metrics['mrr'])} "
            f"rouge_l={format_metric(metrics['rouge_l'])} - {reason}"
        )
        if index < len(cases) and args.case_sleep > 0:
            time.sleep(args.case_sleep)

    summary = summarize_cases(case_reports)
    usage_summary = summarize_usage(case_reports)
    latency_summary = summarize_latency(case_reports)
    ragas_report = run_ragas(case_reports, args.ragas_model, args.ragas_sleep) if args.ragas and not args.retrieval_only else {
        "enabled": False,
        "status": "not_requested" if not args.retrieval_only else "skipped_for_retrieval_only",
        "error": "",
        "summary": {},
        "cases": [],
    }
    thresholds = evaluate_thresholds(summary, usage_summary, latency_summary, args)
    run_id = datetime.now(timezone.utc).strftime("rag_eval_%Y%m%d_%H%M%S")
    report = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_set": str(args.test_set),
        "case_limit": args.limit or None,
        "case_sleep_seconds": args.case_sleep,
        "retrieval_only": args.retrieval_only,
        "qdrant_url": QDRANT_URL,
        "judge_model": FAST_LLM_MODEL,
        "ragas_model": args.ragas_model if args.ragas else None,
        "ragas_sleep_seconds": args.ragas_sleep if args.ragas else None,
        "ragas": ragas_report,
        "summary": summary,
        "usage": usage_summary,
        "latency": latency_summary,
        "thresholds": thresholds,
        "pricing": {
            "source": "GROQ_MODEL_PRICES_JSON",
            "configured_models": sorted(PRICE_CONFIG.keys()),
        },
        "cases": case_reports,
    }

    if case_reports:
        print("\nSummary")
        for key, value in summary.items():
            print(f"{key}={value}" if key in {"cases", "case_errors"} else f"{key}={format_metric(value)}")
        print(f"tokens={usage_summary['total_tokens']}")
        print(f"known_cost_usd={usage_summary['known_cost_usd']:.6f}")
        print(f"average_case_duration_ms={latency_summary['average_case_duration_ms']:.2f}")
        print(f"p95_case_duration_ms={latency_summary['p95_case_duration_ms']:.2f}")
        if usage_summary["unknown_cost_stages"]:
            print(f"unknown_cost_stages={','.join(usage_summary['unknown_cost_stages'])}")
        if ragas_report["status"] == "ok":
            print("ragas_status=ok")
        elif args.ragas:
            print(f"ragas_status={ragas_report['status']} error={ragas_report['error']}")
        if thresholds["checks"]:
            print("thresholds=" + ("passed" if thresholds["passed"] else "failed"))
            for failure in thresholds["failures"]:
                print(
                    "threshold_failure="
                    f"{failure['metric']} actual={failure['actual']} "
                    f"{failure['operator']} {failure['threshold']}"
                )

    if not args.no_report:
        output_path = write_report(report, args.reports_dir)
        print(f"\nWrote report: {output_path}")

    if args.fail_on_threshold and not thresholds["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
