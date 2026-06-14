import argparse
import unittest

from evaluate_rag import (
    deterministic_retrieval_metrics,
    evaluate_thresholds,
    summarize_latency,
)


class DeterministicRetrievalMetricTests(unittest.TestCase):
    def test_scores_ranked_reference_chunks(self) -> None:
        results = [
            {"chunk_id": "policy.pdf:2"},
            {"chunk_id": "policy.pdf:0"},
            {"chunk_id": "policy.pdf:1"},
        ]

        metrics = deterministic_retrieval_metrics(results, ["policy.pdf:0", "policy.pdf:1"])

        self.assertEqual(metrics["hit_at_5"], 1.0)
        self.assertEqual(metrics["mrr"], 0.5)
        self.assertAlmostEqual(metrics["id_context_precision"], 2 / 3)
        self.assertEqual(metrics["id_context_recall"], 1.0)

    def test_returns_not_applicable_without_references(self) -> None:
        metrics = deterministic_retrieval_metrics([{"chunk_id": "policy.pdf:0"}], [])

        self.assertTrue(all(value is None for value in metrics.values()))


class ProductionSummaryTests(unittest.TestCase):
    def test_latency_summary_reports_nearest_rank_percentiles(self) -> None:
        cases = [
            {"usage": {"duration_ms": 100}},
            {"usage": {"duration_ms": 200}},
            {"usage": {"duration_ms": 900}},
        ]

        summary = summarize_latency(cases)

        self.assertEqual(summary["average_case_duration_ms"], 400)
        self.assertEqual(summary["p50_case_duration_ms"], 200)
        self.assertEqual(summary["p95_case_duration_ms"], 900)
        self.assertEqual(summary["max_case_duration_ms"], 900)

    def test_gated_run_fails_when_a_case_errors(self) -> None:
        args = argparse.Namespace(
            fail_on_threshold=True,
            min_faithfulness=None,
            min_answer_relevancy=None,
            min_context_recall=None,
            min_hit_at_5=None,
            min_mrr=None,
            max_average_latency_ms=None,
            max_known_cost_usd=None,
        )

        result = evaluate_thresholds(
            {"case_errors": 1},
            {"known_cost_usd": 0.0},
            {"average_case_duration_ms": 10.0},
            args,
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["failures"][0]["metric"], "case_errors")


if __name__ == "__main__":
    unittest.main()
