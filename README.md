# AI HelpDesk

A small AI helpdesk demo that routes customer support questions to either:

- an order-status API for tracking/order queries
- a RAG pipeline backed by Qdrant for policy/helpdesk questions

It is built for local demos and recordings: Qdrant and the mock order API run in Docker, while the main agent runs as a Python script.
Configuration is local-first: secrets and model settings come from `.env` / environment variables, not notebook or Colab helpers.

## What Is Inside

- `Rag_Agent.py` - main helpdesk agent with intent routing, retrieval, reranking, and answer generation.
- `evaluate_rag.py` - lightweight RAG evaluator for retrieval and generated-answer quality.
- `synthesize_eval_set.py` - creates a synthetic eval set from the policy PDF.
- `observability.py` - optional LangSmith tracing plus per-stage token and cost accounting.
- `rag_dashboard.py` - Streamlit dashboard for saved evaluation reports.
- `Store_Return_Policy.pdf` - sample policy document ingested into Qdrant by the demo runner.
- `reports/` - generated evaluation reports, ignored by Git.
- `qdrant_storage/` - generated local Qdrant data, ignored by Git.
- `orders_api/` - a FastAPI mock order service.
- `docker-compose.yml` - starts Qdrant and the orders API.
- `requirements.txt` - Python dependencies for the local agent.
- `orders_api/requirements.txt` - Python dependencies for the mock orders API container.
- `.env.example` - template for API keys and service URLs.

## Requirements

- Docker Desktop
- Python 3.11+
- Groq API key
- Cohere API key, optional but recommended for reranking

## Setup A Virtual Environment

From the project folder, create a virtual environment:

```powershell
py -3.11 -m venv venv
```

Use Python 3.11 for this project. Newer local runtimes such as Python 3.14 may not support every dependency used by the agent yet.

Activate it on Windows:

```powershell
venv\Scripts\activate
```

If activation worked, your terminal prompt should start with `(venv)`.

Install the Python dependencies:

```powershell
pip install -r requirements.txt
```

## Configure Secrets

Create your local `.env` file from the example:

```powershell
copy .env.example .env
```

Then edit `.env`:

```env
GROQ_API_KEY=gsk_...
LLM_BASE_URL=https://api.groq.com/openai/v1
FAST_LLM_MODEL=llama-3.1-8b-instant
FINAL_LLM_MODEL=llama-3.3-70b-versatile
COHERE_API_KEY=...
QDRANT_URL=http://localhost:6333
ORDER_API_URL=http://localhost:8000
POLICY_DOC_PATH=Store_Return_Policy.pdf
UNSTRUCTURED_STRATEGY=fast
ENABLE_HYDE=true
HYDE_MODEL=llama-3.1-8b-instant
HYDE_MAX_TOKENS=180
ENABLE_CONTEXT_COMPRESSION=true
COMPRESSION_MODEL=llama-3.1-8b-instant
SEMANTIC_BREAK_THRESHOLD=0.70
ENABLE_LANGSMITH=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=ai-helpdesk-rag
GROQ_MODEL_PRICES_JSON={"llama-3.1-8b-instant":{"input_per_1m":0.05,"output_per_1m":0.08},"llama-3.3-70b-versatile":{"input_per_1m":0.59,"output_per_1m":0.79}}
```

The agent uses `FAST_LLM_MODEL` for routing and order ID extraction, and `FINAL_LLM_MODEL` for final customer-facing answers.
`POLICY_DOC_PATH` points to the PDF that should be ingested into Qdrant.
`QDRANT_URL` defaults to local Docker Qdrant. Add `QDRANT_API_KEY` in `.env` only if you are connecting to Qdrant Cloud.
`UNSTRUCTURED_STRATEGY=fast` keeps PDF ingestion lightweight for normal text PDFs. Use `hi_res` only if you install the extra OCR/inference dependencies.
`ENABLE_HYDE` uses a Groq fast model to generate a hypothetical answer for dense retrieval while sparse/BM25 search still uses the original query.
`ENABLE_CONTEXT_COMPRESSION` uses a Groq fast model to trim reranked parent context before final answer generation.
`SEMANTIC_BREAK_THRESHOLD` controls how aggressively adjacent document blocks are split when semantic similarity drops.
`ENABLE_LANGSMITH=true` turns on LangSmith spans only when `LANGSMITH_API_KEY` is also set.
`GROQ_MODEL_PRICES_JSON` controls cost attribution without hardcoded prices. Use JSON shaped like `{"model-name":{"input_per_1m":0.0,"output_per_1m":0.0}}`.

## Run Everything

Use two terminals.

### Terminal 1: Start Docker Services

Start Qdrant and the mock order API:

```powershell
docker compose up --build
```

Services:

- Qdrant: `http://localhost:6333`
- Orders API: `http://localhost:8000/orders/123`

Example order IDs:

- `123`
- `456`
- `ORD-001`

You can quickly check the order API in your browser:

```text
http://localhost:8000/orders/123
```

### Terminal 2: Run The Agent

Open a second terminal in the project folder, then activate the virtual environment:

```powershell
venv\Scripts\activate
```

Run the agent:

```powershell
python Rag_Agent.py
```

The demo runner sends a few sample queries through the agent, including order tracking and policy-style questions.
At startup, the runner ingests `POLICY_DOC_PATH` into the `helpdesk_policy` Qdrant collection, then executes the sample turns.

## RAG Quality And Guardrails

The policy path now includes a few production-style safeguards:

- Metadata filtering: questions that mention a year such as `2026` are used as Qdrant payload filters, and ingested documents store `source`, `type`, `year`, and `chunk_index`.
- Structure-aware chunking: ingestion preserves section names, element categories such as titles/tables/lists, block indexes, and neighboring parent context in Qdrant payloads.
- HyDE retrieval: dense search can use a Groq-generated hypothetical answer for better semantic recall, while sparse/BM25 search keeps the original query for exact keyword matching.
- Small-to-big retrieval: the vector search and reranker operate on small chunks, while the answer model receives compressed neighboring parent context around each selected chunk.
- Contextual compression: after reranking, a Groq fast model removes irrelevant sentences from retrieved parent context before it reaches the final answer model.
- Guardrails: obvious sensitive inputs such as card numbers, SSNs, passwords, and API keys are blocked before they reach the LLM. Highly abusive input is also stopped before routing.
- Self-correction: policy answers pass through a second critic model. If the critic says the answer is not supported by the retrieved context, the agent falls back to the human-handoff response.

You can create a small JSON test set and run retrieval evaluation:

```json
[
  {
    "question": "What is your return policy for electronics?",
    "expected": "electronics return window",
    "golden_answer": "Electronics can be returned within the policy return window if they meet the return conditions.",
    "tags": ["returns", "electronics"],
    "should_answer": true
  }
]
```

For more thorough testing, generate a larger synthetic set from the PDF. The generator batches requests by default so it stays under Groq's smaller on-demand token limits:

```powershell
python synthesize_eval_set.py --count 50 --batch-size 5 --batch-sleep 65 --output .\rag_eval_set.synthetic.json
```

Aim for 50-100 cases before treating the score as production signal. The set should include normal questions, no-answer questions, ambiguous phrasing, and multi-part questions. Review generated `golden_answer` values before using the set for serious benchmarking.

If the Qdrant collection was deleted or reset, ingest the policy PDF before running evaluation:

```powershell
python -c "from Rag_Agent import POLICY_DOC_PATH, ingest_document; ingest_document(POLICY_DOC_PATH, doc_type='policy')"
```

You do not need to separately run `python Rag_Agent.py` for evaluation. `evaluate_rag.py` imports the RAG functions and triggers retrieval, answer generation, judging, and report writing itself.

For a quick smoke test with RAGAS:

```powershell
python evaluate_rag.py .\rag_eval_set.synthetic.json --ragas --limit 2 --case-sleep 30 --ragas-sleep 65
```

For the fuller 50-case report with RAGAS:

```powershell
python evaluate_rag.py .\rag_eval_set.synthetic.json --ragas --case-sleep 30 --ragas-sleep 65
```

Each evaluation writes a timestamped JSON report under `reports/` unless you pass `--no-report`.
RAGAS and the local evaluator can make many Groq calls. On Groq on-demand limits such as 30 requests/minute and about 6.5k tokens/minute for `llama-3.1-8b-instant`, keep `--case-sleep` and `--ragas-sleep` enabled. A 50-case RAGAS run can take more than an hour.

To run the lightweight evaluator without RAGAS:

```powershell
python evaluate_rag.py .\rag_eval_set.synthetic.json --case-sleep 30
```

RAGAS uses a Groq judge model through `langchain-groq` and local FastEmbed embeddings for embedding-based metrics. It runs one eval case at a time in this project to reduce rate-limit spikes. If optional RAGAS dependencies or calls fail, the lightweight evaluator still finishes and records the RAGAS error in the report.

The evaluator reports:

- `faithfulness`: whether the generated answer is supported by retrieved context.
- `answer_relevancy`: whether the generated answer directly addresses the question.
- `context_precision`: whether retrieval avoided filler or junk context.
- `context_recall`: whether retrieval captured all facts needed for the expected/golden answer.
- `hit_at_5`: whether at least one reviewed reference chunk appeared in the top five results.
- `mrr`: how highly the first reviewed reference chunk ranked.
- `id_context_precision`: the share of retrieved chunks that match reviewed reference chunks.
- `id_context_recall`: the share of reviewed reference chunks that retrieval found.
- `rouge_l`: lexical overlap between the generated answer and the `golden_answer`.
- latency, token, and cost metrics by stage, including router, HyDE, compression, final answer, critic, and evaluator stages when usage data is returned by Groq-compatible APIs.

Use `rag_eval_set.reviewed.json` for objective retrieval metrics and regression gates. It exists because synthetic questions and LLM judges cannot objectively prove that retrieval found the correct source chunk. Each reviewed case records the human-checked expected answer and stable `reference_chunks` such as `Store_Return_Policy.pdf:0`, allowing deterministic Hit@5, MRR, ID precision, and ID recall scores. The file also includes `case_type`, `reviewed`, and `reviewer` metadata so the trusted regression set stays separate from generated draft cases.

Run an inexpensive retrieval-only gate:

```powershell
python evaluate_rag.py .\rag_eval_set.reviewed.json --retrieval-only --fail-on-threshold --min-hit-at-5 0.90 --min-mrr 0.70
```

Run a full generation-quality gate:

```powershell
python evaluate_rag.py .\rag_eval_set.reviewed.json --fail-on-threshold --min-faithfulness 0.80 --min-answer-relevancy 0.80 --min-context-recall 0.80
```

Available operational gates include `--max-average-latency-ms` and `--max-known-cost-usd`. Gated runs also fail when any case errors before its metrics can be recorded.

Run this evaluator every time you change retrieval settings in `Rag_Agent.py`, especially `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_LIMIT`, `RERANK_TOP_N`, or `RERANK_MODEL`.

If `faithfulness` is low, chunks may be too small or the reranker may be dropping required context. If `answer_relevancy` is low, inspect the retrieved passages and tune hybrid retrieval or reranking. If `context_precision` is low, too much filler is being retrieved. If `context_recall` is low, retrieval is missing required facts.

For a larger production system, replace or supplement this with RAGAS, DeepEval, or BERTScore once you have a stable labeled test set.

### View The Eval Dashboard

After you have at least one report in `reports/`, launch the local Streamlit dashboard:

```powershell
streamlit run rag_dashboard.py
```

The dashboard shows latest scores, RAGAS status, metric trends across saved reports, low-scoring cases, generated answers, golden answers, and retrieved context.

When the dashboard is launched as a background process, `.streamlit-dashboard.out.log` and `.streamlit-dashboard.err.log` capture its standard output and errors. They are disposable runtime files, are ignored by Git, and can be deleted whenever the dashboard is stopped. If deleted while the dashboard is running, Windows may keep the active file handles open or the logs may be recreated on the next launch.

## Stop Everything

To stop Docker services, go back to Terminal 1 and press `Ctrl+C`.

To fully remove the running containers:

```powershell
docker compose down
```

## Notes

The policy/RAG path expects documents to be ingested into Qdrant before it can answer policy questions. The order-status path can work immediately once Docker Compose is running.
The current demo runner calls `ingest_document(POLICY_DOC_PATH, doc_type="policy")` automatically; if you later wrap the agent in an API, make ingestion a startup/admin step instead of doing it per request.

Do not commit your `.env` file. It contains secrets and is already ignored by Git.

`.python-version` pins this project to Python 3.11 for tools such as `uv` and pyenv. It contains no secrets and is recommended to keep and commit so contributors use a compatible Python version. It is safe to delete, but `uv` or pyenv may then choose a different installed Python version.

## Future Scope

- Durable memory: the current `Session` is intentionally in-memory for the demo. A production deployment should back this with Redis or PostgreSQL.
- Async processing: the agent still uses synchronous client calls. Move order lookups to `httpx.AsyncClient` and convert the agent loop to `asyncio` when serving concurrent users behind an API.
- Advanced guardrails: consider NVIDIA NeMo Guardrails, Llama Guard, or your platform moderation layer for stronger PII, safety, and policy enforcement.
