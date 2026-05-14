# AI HelpDesk

A small AI helpdesk demo that routes customer support questions to either:

- an order-status API for tracking/order queries
- a RAG pipeline backed by Qdrant for policy/helpdesk questions

It is built for local demos and recordings: Qdrant and the mock order API run in Docker, while the main agent runs as a Python script.

## What Is Inside

- `Rag_Agent.py` - main helpdesk agent with intent routing, retrieval, reranking, and answer generation.
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
```

The agent uses `FAST_LLM_MODEL` for routing and order ID extraction, and `FINAL_LLM_MODEL` for final customer-facing answers.

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

## Stop Everything

To stop Docker services, go back to Terminal 1 and press `Ctrl+C`.

To fully remove the running containers:

```powershell
docker compose down
```

## Notes

The policy/RAG path expects documents to be ingested into Qdrant before it can answer policy questions. The order-status path can work immediately once Docker Compose is running.

Do not commit your `.env` file. It contains secrets and is already ignored by Git.
