"""
Optional LangSmith tracing plus lightweight usage/cost accounting.

Tracing is enabled only when ENABLE_LANGSMITH=true and LANGSMITH_API_KEY is set.
Cost accounting uses GROQ_MODEL_PRICES_JSON, for example:
{"llama-3.3-70b-versatile":{"input_per_1m":0.59,"output_per_1m":0.79}}
"""

from __future__ import annotations

import contextvars
import json
import os
import time
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Iterator


def env_enabled(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


ENABLE_LANGSMITH = env_enabled("ENABLE_LANGSMITH")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "ai-helpdesk-rag")
TRACING_ENABLED = ENABLE_LANGSMITH and bool(LANGSMITH_API_KEY)


def load_price_config() -> dict[str, dict[str, float]]:
    raw = os.getenv("GROQ_MODEL_PRICES_JSON", "{}")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    prices: dict[str, dict[str, float]] = {}
    for model, values in payload.items():
        if not isinstance(values, dict):
            continue
        try:
            prices[model] = {
                "input_per_1m": float(values.get("input_per_1m", 0.0)),
                "output_per_1m": float(values.get("output_per_1m", 0.0)),
            }
        except (TypeError, ValueError):
            continue
    return prices


PRICE_CONFIG = load_price_config()


@dataclass
class UsageRecord:
    stage: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "metadata": self.metadata,
        }


@dataclass
class UsageRun:
    name: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: float = field(default_factory=time.time)
    records: list[UsageRecord] = field(default_factory=list)

    def add(self, record: UsageRecord) -> None:
        self.records.append(record)

    def as_dict(self) -> dict[str, Any]:
        total_known_cost = sum(record.cost_usd for record in self.records if record.cost_usd is not None)
        unknown_cost_stages = [record.stage for record in self.records if record.cost_usd is None and record.total_tokens]
        by_stage: dict[str, dict[str, Any]] = {}
        for record in self.records:
            stage = by_stage.setdefault(
                record.stage,
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "unknown_cost": False,
                    "calls": 0,
                },
            )
            stage["input_tokens"] += record.input_tokens
            stage["output_tokens"] += record.output_tokens
            stage["total_tokens"] += record.total_tokens
            stage["calls"] += 1
            if record.cost_usd is None and record.total_tokens:
                stage["unknown_cost"] = True
            elif record.cost_usd is not None:
                stage["cost_usd"] += record.cost_usd

        return {
            "run_id": self.run_id,
            "name": self.name,
            "duration_ms": round((time.time() - self.started_at) * 1000, 2),
            "total_input_tokens": sum(record.input_tokens for record in self.records),
            "total_output_tokens": sum(record.output_tokens for record in self.records),
            "total_tokens": sum(record.total_tokens for record in self.records),
            "known_cost_usd": total_known_cost,
            "unknown_cost_stages": unknown_cost_stages,
            "by_stage": by_stage,
            "records": [record.as_dict() for record in self.records],
        }


_CURRENT_USAGE_RUN: contextvars.ContextVar[UsageRun | None] = contextvars.ContextVar(
    "current_usage_run",
    default=None,
)


@contextmanager
def usage_run(name: str) -> Iterator[UsageRun]:
    parent = _CURRENT_USAGE_RUN.get()
    if parent is not None:
        yield parent
        return

    run = UsageRun(name=name)
    token = _CURRENT_USAGE_RUN.set(run)
    try:
        yield run
    finally:
        _CURRENT_USAGE_RUN.reset(token)


def current_usage_report() -> dict[str, Any] | None:
    run = _CURRENT_USAGE_RUN.get()
    return run.as_dict() if run else None


def _token_attr(usage: Any, *names: str) -> int:
    for name in names:
        if isinstance(usage, dict) and usage.get(name) is not None:
            return int(usage[name])
        value = getattr(usage, name, None)
        if value is not None:
            return int(value)
    return 0


def cost_for_tokens(model: str, input_tokens: int, output_tokens: int) -> float | None:
    price = PRICE_CONFIG.get(model)
    if not price:
        return None
    return (
        (input_tokens / 1_000_000) * price["input_per_1m"]
        + (output_tokens / 1_000_000) * price["output_per_1m"]
    )


def record_llm_response(stage: str, model: str, response: Any, metadata: dict[str, Any] | None = None) -> None:
    usage = getattr(response, "usage", None)
    input_tokens = _token_attr(usage, "prompt_tokens", "input_tokens")
    output_tokens = _token_attr(usage, "completion_tokens", "output_tokens")
    total_tokens = _token_attr(usage, "total_tokens") or input_tokens + output_tokens
    record = UsageRecord(
        stage=stage,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_for_tokens(model, input_tokens, output_tokens),
        metadata=metadata or {},
    )

    run = _CURRENT_USAGE_RUN.get()
    if run:
        run.add(record)

    if TRACING_ENABLED:
        try:
            from langsmith import set_run_metadata

            set_run_metadata({"usage": record.as_dict()})
        except Exception:
            pass


def record_usage_snapshot(stage: str, model: str, usage: dict[str, Any]) -> None:
    input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
    record = UsageRecord(
        stage=stage,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_for_tokens(model, input_tokens, output_tokens),
        metadata=usage.get("metadata", {}),
    )
    run = _CURRENT_USAGE_RUN.get()
    if run:
        run.add(record)


@contextmanager
def trace_span(
    name: str,
    *,
    run_type: str = "chain",
    inputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    if not TRACING_ENABLED:
        with nullcontext():
            yield
        return

    try:
        from langsmith import trace

        with trace(
            name,
            run_type=run_type,
            inputs=inputs,
            metadata=metadata,
            project_name=LANGSMITH_PROJECT,
            exceptions_to_handle=(Exception,),
        ):
            yield
    except Exception:
        yield
