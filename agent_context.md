# Agent Context

This file gives future coding agents the project context and recent task history.

## Project Purpose

This is a small AI helpdesk/RAG demo. It combines:

- A Python RAG agent in `Rag_Agent.py`
- Local Qdrant vector search through Docker
- A mock FastAPI order-status service in `orders_api/`
- `.env` based secret loading for API keys
- Optional LangSmith tracing, token usage, and cost accounting through `observability.py`
- RAG evaluation reports and a local Streamlit dashboard

The intended demo flow is:

1. Start Qdrant and the orders API with Docker Compose.
2. Run the Python agent locally.
3. The demo runner ingests `Store_Return_Policy.pdf` into Qdrant.
4. The agent routes customer messages to either order lookup or policy/RAG answers.

## Current Repository State

The repo was initialized locally in:

`C:\Users\gurno\Desktop\AI helpdesk`

Remote:

`https://github.com/GurnoorSH/Ai-HelpDesk.git`

Branch:

`main`

The repository has commits for the initial project setup, Groq/environment refactor, orders API, RAG quality improvements, and observability/reporting features.

## Important Files

- `Rag_Agent.py` - main RAG/helpdesk agent script.
- `docker-compose.yml` - starts Qdrant and the orders API service.
- `orders_api/main.py` - mock FastAPI order status endpoint.
- `orders_api/Dockerfile` - container for the orders API.
- `orders_api/requirements.txt` - container dependencies for the orders API.
- `requirements.txt` - local Python dependencies for the RAG agent.
- `Store_Return_Policy.pdf` - sample policy PDF used by the demo ingestion path.
- `observability.py` - optional LangSmith tracing plus per-stage token and cost tracking.
- `evaluate_rag.py` - lightweight RAG evaluator for retrieval and generated-answer quality.
- `synthesize_eval_set.py` - synthetic eval-set generator from the policy PDF.
- `rag_dashboard.py` - Streamlit dashboard for timestamped reports in `reports/`.
- `reports/` - generated evaluator output directory, ignored by Git.
- `qdrant_storage/` - generated local Qdrant storage directory, ignored by Git.
- `spec_sheet.md` - project/spec notes for the helpdesk demo.
- `.env.example` - sample environment variables.
- `.gitignore` - excludes local secrets, virtualenvs, caches, and Qdrant storage.
- `README.md` - human-readable project overview and setup guide.

## Recent Task History

- Git was initialized in the workspace.
- The GitHub remote was added as `origin`.
- The branch was renamed from `master` to `main`.
- Git safe directory config was needed because Windows reported a different owner SID for the folder.
- Docker Compose contains:
  - `qdrant` on port `6333`
  - `orders-api` built from `./orders_api` on port `8000`
- The orders API container has its own small `orders_api/requirements.txt`.
- Updated `Rag_Agent.py` to:
  - load `.env` with `python-dotenv`
  - use only `.env` / process environment variables for config; Colab `userdata` support was removed to keep local code simple
  - default Qdrant to `http://localhost:6333`
  - use FastEmbed dense model `BAAI/bge-small-en-v1.5` and sparse model `Qdrant/bm25`
  - default order API to `http://localhost:8000`
  - default policy ingestion to `Store_Return_Policy.pdf`, overridable with `POLICY_DOC_PATH`
  - default document partitioning to `UNSTRUCTURED_STRATEGY=fast` to avoid heavy OCR/inference dependencies for normal PDFs
  - use `requests.get(...)` for order lookups
  - keep the old mock order function renamed as `_get_order_status_stub`
- Switched the LLM configuration to Groq using the OpenAI-compatible client:
  - `GROQ_API_KEY`
  - `LLM_BASE_URL=https://api.groq.com/openai/v1`
  - `FAST_LLM_MODEL=llama-3.1-8b-instant` for routing and extraction
  - `FINAL_LLM_MODEL=llama-3.3-70b-versatile` for final customer-facing answer generation
  - `OPENAI_API_KEY` is still accepted as a fallback if it was already used locally.
- Replaced OpenAI `beta.chat.completions.parse(...)` with Groq JSON object mode plus Pydantic validation.
- Added RAG quality improvements:
  - deterministic pre-LLM guardrails for obvious sensitive or abusive input
  - metadata-aware retrieval filters for fields such as `year` and `type`
  - small-to-big retrieval by reranking small chunks and feeding neighboring `parent_context`
  - HyDE dense retrieval using the fast Groq model while sparse/BM25 retrieval keeps the original query
  - contextual compression using the fast Groq model before final answer generation
  - a critic pass that evaluates unsupported policy answers and logs a warning (does not block the answer, preventing false positive refusals)
  - `evaluate_rag.py` for repeatable faithfulness, answer-relevancy, context precision/recall, and ROUGE-L scoring against golden answers
  - `synthesize_eval_set.py` for generating 50+ diverse eval cases from the PDF
- **Phase 0 (Generation Quality Fixes) completed**:
  - Faithfulness improved from ~0.40 to 0.92
  - Answer Relevancy improved from ~0.40 to 0.98
  - Upgraded critic model to 70B and improved prompt with deduction rules
  - Context compression max_tokens increased to 400 to preserve facts
- Added observability and reporting:
  - `observability.py` records LLM usage by stage and computes optional cost from `GROQ_MODEL_PRICES_JSON`
  - optional LangSmith spans are enabled only when `ENABLE_LANGSMITH=true` and `LANGSMITH_API_KEY` is set
  - `evaluate_rag.py` writes timestamped JSON reports under `reports/` by default
  - optional RAGAS metrics run through `python evaluate_rag.py .\rag_eval_set.json --ragas`
  - `rag_dashboard.py` displays latest scores, trends, low-scoring cases, generated answers, golden answers, retrieved context, token usage, and partial cost data
- Left durable memory and async/concurrent serving in future scope; `Session` remains intentionally in-memory for this demo.
- Added `.env.example`, `.gitignore`, and `requirements.txt`.
- Current local verification caveat: the checked-in `venv` can point to a missing Python install, and `python` / `py` may not be available on PATH. Use a fresh Python 3.11 virtualenv or the bundled Codex runtime path below when verifying.

## Notes For Future Agents

- Do not commit `.env`.
- Prefer local Qdrant at `http://localhost:6333` unless the user explicitly wants Qdrant Cloud.
- Add `QDRANT_API_KEY` only for Qdrant Cloud; local Docker Qdrant does not need it.
- The project no longer uses `google.colab` or Colab `userdata`; keep config local-first through `.env`.
- Generated `reports/` and `qdrant_storage/` are intentionally ignored by Git.
- `Rag_Agent.py` intentionally contains a few Unicode separators and arrows in comments/output labels.
- Use small targeted patches for `Rag_Agent.py`; exact comment text may be hard to match.
- There should be only one Dockerfile, at `orders_api/Dockerfile`.
- If verifying Python, use the bundled runtime if system Python is still missing:
  `C:\Users\gurno\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`
