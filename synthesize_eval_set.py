"""
Generate a synthetic RAG evaluation set from the policy PDF.

The output schema is compatible with evaluate_rag.py:
[
  {
    "question": "...",
    "expected": "short expected facts",
    "golden_answer": "ideal customer-facing answer",
    "tags": ["returns", "edge_case"],
    "should_answer": true
  }
]
"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
POLICY_DOC_PATH = Path(os.getenv("POLICY_DOC_PATH", BASE_DIR / "Store_Return_Policy.pdf"))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")

llm = OpenAI(api_key=GROQ_API_KEY or "missing-groq-api-key", base_url=LLM_BASE_URL)


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()


def synthesize_case_batch(policy_text: str, count: int, batch_number: int) -> list[dict]:
    policy_excerpt = policy_text[:8000]
    prompt = (
        f"Generate {count} diverse RAG evaluation cases from this store policy text. "
        "Return only JSON with a top-level key named cases.\n\n"
        "Each case must include: question, expected, golden_answer, tags, should_answer.\n"
        "Requirements:\n"
        "- Cover normal policy questions, edge cases, ambiguous phrasing, and multi-part questions.\n"
        "- Include about 15 percent no-answer questions where should_answer is false and the "
        "golden answer says the information is not available in the policy.\n"
        "- Keep golden_answer concise and customer-facing.\n"
        "- expected should be a short list of required facts, not prose.\n"
        "- tags should be short strings like returns, refunds, electronics, no_answer, multi_part.\n\n"
        f"Batch number: {batch_number}\n"
        f"Policy text:\n{policy_excerpt}"
    )
    response = llm.chat.completions.create(
        model=FAST_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=min(3200, 700 + count * 240),
    )
    payload = json.loads(response.choices[0].message.content or "{}")
    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("Synthetic generator did not return a cases list.")
    return cases[:count]


def synthesize_cases(policy_text: str, count: int, batch_size: int, sleep_seconds: float) -> list[dict]:
    cases: list[dict] = []
    batch_number = 1
    while len(cases) < count:
        remaining = count - len(cases)
        current_batch_size = min(batch_size, remaining)
        cases.extend(synthesize_case_batch(policy_text, current_batch_size, batch_number))
        batch_number += 1
        if len(cases) < count and sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return cases[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic RAG evaluation cases.")
    parser.add_argument("--count", type=int, default=50, help="Number of cases to generate.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("SYNTH_EVAL_BATCH_SIZE", "5")),
        help="Cases to generate per Groq call. Keep low for small TPM limits.",
    )
    parser.add_argument(
        "--batch-sleep",
        type=float,
        default=float(os.getenv("SYNTH_EVAL_BATCH_SLEEP_SECONDS", "65")),
        help="Seconds to sleep between generation batches.",
    )
    parser.add_argument("--policy", type=Path, default=POLICY_DOC_PATH, help="Policy PDF path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "rag_eval_set.synthetic.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    policy_path = args.policy if args.policy.is_absolute() else BASE_DIR / args.policy
    policy_text = extract_pdf_text(policy_path)
    if not policy_text:
        raise ValueError(f"No text could be extracted from {policy_path}")

    cases = synthesize_cases(policy_text, args.count, args.batch_size, args.batch_sleep)
    args.output.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
