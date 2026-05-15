"""
Streamlit dashboard for saved RAG evaluation reports.

Run with:
    streamlit run rag_dashboard.py
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


REPORTS_DIR = Path("reports")
METRIC_KEYS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "rouge_l",
]


def load_reports(reports_dir: Path = REPORTS_DIR) -> list[dict[str, Any]]:
    if not reports_dir.exists():
        return []
    reports: list[dict[str, Any]] = []
    for path in sorted(reports_dir.glob("*.json"), reverse=True):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            report["_path"] = str(path)
            reports.append(report)
        except Exception:
            continue
    return reports


def summary_frame(reports: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for report in reports:
        summary = report.get("summary", {})
        ragas = report.get("ragas", {})
        usage = report.get("usage", {})
        row = {
            "run_id": report.get("run_id", ""),
            "created_at": report.get("created_at", ""),
            "test_set": report.get("test_set", ""),
            "cases": summary.get("cases", 0),
            "ragas_status": ragas.get("status", "not_requested"),
            "total_tokens": usage.get("total_tokens", 0),
            "known_cost_usd": usage.get("known_cost_usd", 0.0),
            "path": report.get("_path", ""),
        }
        for key in METRIC_KEYS:
            row[key] = summary.get(key)
        rows.append(row)
    return pd.DataFrame(rows)


def case_frame(report: dict[str, Any]) -> pd.DataFrame:
    rows = []
    ragas_cases = report.get("ragas", {}).get("cases", [])
    for case in report.get("cases", []):
        metrics = case.get("metrics", {})
        row = {
            "index": case.get("index"),
            "question": case.get("question", ""),
            "reason": case.get("reason", ""),
            "tags": ", ".join(case.get("tags") or []),
        }
        usage = case.get("usage") or {}
        row["total_tokens"] = usage.get("total_tokens", 0)
        row["known_cost_usd"] = usage.get("known_cost_usd", 0.0)
        row["unknown_cost_stages"] = ", ".join(usage.get("unknown_cost_stages") or [])
        for key in METRIC_KEYS:
            row[key] = metrics.get(key)
        if ragas_cases and len(ragas_cases) >= int(case.get("index", 0)):
            ragas_row = ragas_cases[int(case["index"]) - 1]
            for key in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
                if key in ragas_row:
                    row[f"ragas_{key}"] = ragas_row[key]
        rows.append(row)
    return pd.DataFrame(rows)


def metric_delta(current: dict[str, Any], previous: dict[str, Any] | None, key: str) -> float | None:
    if not previous:
        return None
    current_value = current.get("summary", {}).get(key)
    previous_value = previous.get("summary", {}).get(key)
    if current_value is None or previous_value is None:
        return None
    return float(current_value) - float(previous_value)


def main() -> None:
    st.set_page_config(page_title="RAG Eval Dashboard", layout="wide")
    st.title("RAG Eval Dashboard")

    reports = load_reports()
    if not reports:
        st.info("No reports found. Run `python evaluate_rag.py .\\rag_eval_set.json` first.")
        return

    df = summary_frame(reports)
    selected_run = st.sidebar.selectbox(
        "Report",
        options=df["run_id"].tolist(),
        format_func=lambda run_id: f"{run_id} ({df.loc[df['run_id'] == run_id, 'cases'].iloc[0]} cases)",
    )
    selected_index = df.index[df["run_id"] == selected_run][0]
    report = reports[int(selected_index)]
    previous_report = reports[int(selected_index) + 1] if int(selected_index) + 1 < len(reports) else None

    st.caption(f"Report: `{report.get('_path', '')}`")

    usage = report.get("usage", {})
    usage_cols = st.columns(4)
    usage_cols[0].metric("Eval Tokens", f"{int(usage.get('total_tokens') or 0):,}")
    usage_cols[1].metric("Input Tokens", f"{int(usage.get('total_input_tokens') or 0):,}")
    usage_cols[2].metric("Output Tokens", f"{int(usage.get('total_output_tokens') or 0):,}")
    usage_cols[3].metric("Known Cost", f"${float(usage.get('known_cost_usd') or 0.0):.6f}")
    unknown_stages = usage.get("unknown_cost_stages") or []
    if unknown_stages:
        st.info(
            "Cost is partial. Add prices to `GROQ_MODEL_PRICES_JSON` for: "
            + ", ".join(sorted(set(unknown_stages)))
        )

    cols = st.columns(len(METRIC_KEYS))
    for col, key in zip(cols, METRIC_KEYS):
        value = report.get("summary", {}).get(key, 0.0)
        delta = metric_delta(report, previous_report, key)
        col.metric(key.replace("_", " ").title(), f"{float(value):.2f}", None if delta is None else f"{delta:+.2f}")

    ragas = report.get("ragas", {})
    if ragas.get("status") == "ok":
        st.success("RAGAS metrics available for this run.")
        ragas_cols = st.columns(4)
        for col, key in zip(ragas_cols, ("faithfulness", "answer_relevancy", "context_precision", "context_recall")):
            col.metric(f"RAGAS {key.replace('_', ' ').title()}", f"{float(ragas['summary'].get(key, 0.0)):.2f}")
    elif ragas.get("status") not in {None, "not_requested"}:
        st.warning(f"RAGAS status: {ragas.get('status')} - {ragas.get('error', '')}")

    st.subheader("Trends")
    trend_df = df.sort_values("created_at")
    st.line_chart(trend_df.set_index("run_id")[METRIC_KEYS])
    if {"total_tokens", "known_cost_usd"}.issubset(trend_df.columns):
        st.line_chart(trend_df.set_index("run_id")[["total_tokens", "known_cost_usd"]])

    st.subheader("Cases")
    cases_df = case_frame(report)
    threshold = st.slider("Show cases with any metric below", 0.0, 1.0, 0.75, 0.05)
    metric_columns = [key for key in METRIC_KEYS if key in cases_df.columns]
    failing_mask = cases_df[metric_columns].lt(threshold).any(axis=1) if metric_columns else []
    show_failures_only = st.toggle("Failures only", value=False)
    visible_df = cases_df[failing_mask] if show_failures_only and metric_columns else cases_df
    st.dataframe(visible_df, use_container_width=True, hide_index=True)

    selected_case = st.selectbox(
        "Inspect case",
        options=[case.get("index") for case in report.get("cases", [])],
    )
    case = next(item for item in report.get("cases", []) if item.get("index") == selected_case)

    left, right = st.columns(2)
    with left:
        st.markdown("**Question**")
        st.write(case.get("question", ""))
        st.markdown("**Generated answer**")
        st.write(case.get("generated_answer", ""))
        st.markdown("**Golden answer**")
        st.write(case.get("golden_answer", ""))
    with right:
        st.markdown("**Judge reason**")
        st.write(case.get("reason", ""))
        st.markdown("**Retrieved context**")
        st.text_area("Context", value=case.get("context", ""), height=420, label_visibility="collapsed")


if __name__ == "__main__":
    main()
