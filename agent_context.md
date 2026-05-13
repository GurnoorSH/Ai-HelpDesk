# Agent Context

This file gives future coding agents the project context and recent task history.

## Project Purpose

This is a small AI helpdesk/RAG demo. It combines:

- A Python RAG agent in `Rag_Agent.py`
- Local Qdrant vector search through Docker
- A mock FastAPI order-status service in `orders_api/`
- `.env` based secret loading for API keys

The intended demo flow is:

1. Start Qdrant and the orders API with Docker Compose.
2. Run the Python agent locally.
3. The agent routes customer messages to either order lookup or policy/RAG answers.

## Current Repository State

The repo was initialized locally in:

`C:\Users\gurno\Desktop\AI helpdesk`

Remote:

`https://github.com/GurnoorSH/Ai-HelpDesk.git`

Branch:

`main`

There are no commits yet at the time this context file was added.

## Important Files

- `Rag_Agent.py` - main RAG/helpdesk agent script.
- `docker-compose.yml` - starts Qdrant and the orders API service.
- `orders_api/main.py` - mock FastAPI order status endpoint.
- `orders_api/Dockerfile` - container for the orders API.
- `requirements.txt` - local Python dependencies.
- `.env.example` - sample environment variables.
- `.gitignore` - excludes local secrets, virtualenvs, caches, and Qdrant storage.
- `README.md` - human-readable project overview and setup guide.

## Recent Task History

- Git was initialized in the workspace.
- The GitHub remote was added as `origin`.
- The branch was renamed from `master` to `main`.
- Git safe directory config was needed because Windows reported a different owner SID for the folder.
- Docker Compose already contained:
  - `qdrant` on port `6333`
  - `orders-api` built from `./orders_api` on port `8000`
- Added `orders_api/main.py` and `orders_api/Dockerfile` because Compose expected that folder.
- Updated `Rag_Agent.py` to:
  - load `.env` with `python-dotenv`
  - default Qdrant to `http://localhost:6333`
  - default order API to `http://localhost:8000`
  - use `requests.get(...)` for order lookups
  - keep the old mock order function renamed as `_get_order_status_stub`
- Added `.env.example`, `.gitignore`, and `requirements.txt`.
- Syntax check passed using the bundled Codex Python runtime because `python` and `py` were not available on PATH.

## Notes For Future Agents

- Do not commit `.env`.
- Prefer local Qdrant at `http://localhost:6333` unless the user explicitly wants Qdrant Cloud.
- The current `Rag_Agent.py` contains some mojibake in comments and strings from an earlier encoding issue. Be careful when patching around those lines.
- Use small targeted patches for `Rag_Agent.py`; exact comment text may be hard to match.
- The root `main.py` and root `Dockerfile` are still present from the original project state. The Docker Compose service now uses `orders_api/`.
- If verifying Python, use the bundled runtime if system Python is still missing:
  `C:\Users\gurno\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`

