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
- `.env.example` - template for API keys and service URLs.

## Requirements

- Docker Desktop
- Python 3.11+
- OpenAI API key
- Cohere API key, optional but recommended for reranking

## Setup

Create a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create your local `.env` file from the example:

```powershell
copy .env.example .env
```

Then edit `.env`:

```env
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...
QDRANT_URL=http://localhost:6333
ORDER_API_URL=http://localhost:8000
```

## Run The Services

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

## Run The Agent

In a second terminal, with the virtual environment activated:

```powershell
python Rag_Agent.py
```

The demo runner sends a few sample queries through the agent, including order tracking and policy-style questions.

## Notes

The policy/RAG path expects documents to be ingested into Qdrant before it can answer policy questions. The order-status path can work immediately once Docker Compose is running.

Do not commit your `.env` file. It contains secrets and is already ignored by Git.

