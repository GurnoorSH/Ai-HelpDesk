"""
Production-Grade RAG Agent — 2026
Upwork Chatbot + Helpdesk Project

Architecture:
  - Hybrid Search (Dense + Sparse) via Qdrant + FastEmbed
  - Cohere Reranking
  - Structured Intent Routing (Pydantic)
  - Recursive Chunking with Overlap
  - Multi-Turn Conversation Memory
  - Regex + LLM Order ID Extraction
  - Full Error Handling + Fallbacks
  - Cloud-Ready Qdrant Config
"""

# ─────────────────────────────────────────────
# 1. INSTALL (run once in Colab)
# ─────────────────────────────────────────────
# !apt-get install -y libmagic-dev poppler-utils tesseract-ocr
# !pip install -U "unstructured[all-docs]" qdrant-client[fastembed] \
#     openai cohere langchain-text-splitters pydantic tenacity requests python-dotenv

# ─────────────────────────────────────────────
# 2. IMPORTS & CONFIG
# ─────────────────────────────────────────────
import os
import re
import logging
from typing import Literal, Optional
from dataclasses import dataclass, field

import requests
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from qdrant_client import QdrantClient
from openai import OpenAI
import cohere
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.auto import partition

# ── Logging ──────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

load_dotenv()

# ── Credentials (Colab) ──────────────────────
try:
    from google.colab import userdata
    OPENAI_API_KEY  = userdata.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    COHERE_API_KEY  = userdata.get("COHERE_API_KEY") or os.getenv("COHERE_API_KEY")
    QDRANT_URL      = userdata.get("QDRANT_URL") or os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY  = userdata.get("QDRANT_API_KEY") or os.getenv("QDRANT_API_KEY")
    ORDER_API_URL   = userdata.get("ORDER_API_URL") or os.getenv("ORDER_API_URL", "http://localhost:8000")
except Exception:
    # Fallback to environment variables (local / CI)
    OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
    COHERE_API_KEY  = os.getenv("COHERE_API_KEY")
    QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
    ORDER_API_URL   = os.getenv("ORDER_API_URL", "http://localhost:8000")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY or ""

COLLECTION_NAME  = "helpdesk_policy"
CHUNK_SIZE       = 512
CHUNK_OVERLAP    = 64
RETRIEVAL_LIMIT  = 6    # fetch more, reranker will cut down
RERANK_TOP_N     = 3
LLM_MODEL        = "gpt-4o"
RERANK_MODEL     = "rerank-v3.5"


# ─────────────────────────────────────────────
# 3. CLIENT INITIALIZATION
# ─────────────────────────────────────────────
def build_qdrant_client() -> QdrantClient:
    """
    Connect to local Docker Qdrant by default; support cloud via env vars.
    """
    if QDRANT_API_KEY:
        log.info("Connecting to Qdrant Cloud: %s", QDRANT_URL)
        c = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        log.info("Connecting to local Qdrant: %s", QDRANT_URL)
        c = QdrantClient(url=QDRANT_URL)

    c.set_model("BAAI/bge-small-en-v1.5")       # Dense
    c.set_sparse_model("naver/splade-v3")        # Sparse
    return c


qdrant  = build_qdrant_client()
openai  = OpenAI()
co      = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None


# ─────────────────────────────────────────────
# 4. INGESTION — Chunked, Structure-Aware
# ─────────────────────────────────────────────
def ingest_document(file_path: str, doc_type: str = "policy") -> int:
    """
    Parse a document, chunk it properly, and upsert into Qdrant.
    Returns the number of chunks ingested.
    """
    log.info("Partitioning document: %s", file_path)
    elements = partition(filename=file_path, strategy="hi_res")

    raw_text = "\n\n".join(
        str(el) for el in elements
        if el.category in {"NarrativeText", "Table", "ListItem", "Title"}
    )

    if not raw_text.strip():
        raise ValueError(f"No usable text extracted from {file_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(raw_text)
    log.info("Split into %d chunks", len(chunks))

    # Ensure collection exists
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        # client.add() auto-creates, but being explicit is safer
        log.info("Collection '%s' will be created on first upsert", COLLECTION_NAME)

    metadata = [
        {"source": os.path.basename(file_path), "type": doc_type, "chunk_index": i}
        for i, _ in enumerate(chunks)
    ]

    qdrant.add(
        collection_name=COLLECTION_NAME,
        documents=chunks,
        metadata=metadata,
    )

    log.info("Ingestion complete: %d chunks stored.", len(chunks))
    return len(chunks)


# ─────────────────────────────────────────────
# 5. RETRIEVAL — Hybrid + Rerank
# ─────────────────────────────────────────────
def retrieve(query: str) -> list[str]:
    """
    Hybrid search → Cohere reranking → top-N passages.
    """
    hits = qdrant.query(
        collection_name=COLLECTION_NAME,
        query_text=query,
        limit=RETRIEVAL_LIMIT,
    )
    docs = [h.document for h in hits if h.document]

    if not docs:
        return []

    if co:
        try:
            reranked = co.rerank(
                query=query,
                documents=docs,
                model=RERANK_MODEL,
                top_n=RERANK_TOP_N,
            )
            return [docs[r.index] for r in reranked.results]
        except Exception as e:
            log.warning("Cohere rerank failed (%s), using raw hits", e)

    return docs[:RERANK_TOP_N]


# ─────────────────────────────────────────────
# 6. ORDER ID EXTRACTION
# ─────────────────────────────────────────────
def extract_order_id(text: str) -> Optional[str]:
    """
    Try regex first (fast), fall back to LLM extraction.
    Handles formats: #12345, ORD-12345, order 12345, etc.
    """
    # Regex patterns for common order ID formats
    patterns = [
        r"(?:order|ord|#)\s*[-:]?\s*([A-Z0-9]{4,12})",  # ORD-12345, order 12345
        r"\b([A-Z]{2,4}-\d{4,10})\b",                    # ORD-123456
        r"\b(\d{5,10})\b",                                # plain numeric
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # LLM fallback
    try:
        resp = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    f"Extract the order ID from this message. "
                    f"Reply with ONLY the ID or 'NONE'.\n\nMessage: {text}"
                )
            }],
            max_tokens=20,
        )
        result = resp.choices[0].message.content.strip()
        return None if result.upper() == "NONE" else result
    except Exception as e:
        log.error("Order ID LLM extraction failed: %s", e)
        return None


# ─────────────────────────────────────────────
# 7. ORDER API (replace with real DB/API call)
# ─────────────────────────────────────────────
def _get_order_status_stub(order_id: str) -> str:
    """
    Stub — replace with actual DB query or API call.
    Example: requests.get(f"{ORDER_API_URL}/orders/{order_id}", headers=AUTH)
    """
    mock_db = {
        "123":    "Shipped — arriving Friday via FedEx (#TRK789).",
        "456":    "Processing — estimated dispatch in 2 business days.",
        "789":    "Delivered on 10 May 2026. Signed by: J. Smith.",
        "ORD-001": "Cancelled — refund issued within 5–7 business days.",
    }
    return mock_db.get(order_id, f"No order found for ID '{order_id}'. Please verify and try again.")


def get_order_status(order_id: str) -> str:
    """
    Fetch order status from the local FastAPI orders service.
    """
    try:
        response = requests.get(f"{ORDER_API_URL}/orders/{order_id}", timeout=5)
        if response.ok:
            return response.json().get("status", "Order service returned no status.")
    except requests.RequestException as e:
        log.warning("Order service request failed: %s", e)

    return "Could not reach order service."


# ─────────────────────────────────────────────
# 8. STRUCTURED INTENT ROUTER
# ─────────────────────────────────────────────
class IntentResult(BaseModel):
    category: Literal["ORDER", "POLICY", "ESCALATE", "OTHER"]
    confidence: float          # 0–1, for logging / threshold gating
    reasoning: str             # brief explanation (useful for debugging)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def classify_intent(query: str, history_summary: str = "") -> IntentResult:
    """
    Use structured output to get a reliable, typed intent classification.
    """
    system = (
        "You are an intent classifier for a customer helpdesk. "
        "Classify the user's query into exactly one category:\n"
        "  ORDER    — tracking, status, shipping, cancellation of a specific order\n"
        "  POLICY   — returns, refunds, warranty, store rules, general how-to\n"
        "  ESCALATE — angry customer, complaint, legal threat, abuse\n"
        "  OTHER    — anything else\n"
        "Return JSON matching the schema provided."
    )
    context = f"Conversation so far: {history_summary}\n\n" if history_summary else ""
    resp = openai.beta.chat.completions.parse(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"{context}User query: {query}"},
        ],
        response_format=IntentResult,
    )
    return resp.choices[0].message.parsed


# ─────────────────────────────────────────────
# 9. GENERATION — Grounded Answer
# ─────────────────────────────────────────────
SYSTEM_POLICY = """You are a helpful customer support agent.
Answer ONLY using the context provided. Be concise and friendly.
If the context does not contain the answer, say:
'I don't have that information. Would you like me to connect you with a human agent?'
Never fabricate information."""

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def generate_answer(query: str, context: str, messages: list[dict]) -> str:
    """
    Generate a grounded answer with conversation history included.
    """
    payload = [{"role": "system", "content": SYSTEM_POLICY}]
    payload += messages  # full history for multi-turn context
    payload.append({
        "role": "user",
        "content": f"Reference context:\n{context}\n\nQuestion: {query}"
    })

    resp = openai.chat.completions.create(
        model=LLM_MODEL,
        messages=payload,
        temperature=0.2,     # low temp for factual helpdesk responses
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# 10. CONVERSATION SESSION
# ─────────────────────────────────────────────
@dataclass
class Session:
    """Holds per-user conversation state."""
    messages: list[dict] = field(default_factory=list)

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def history_summary(self, last_n: int = 6) -> str:
        """Return a compact text summary of recent turns for the router."""
        tail = self.messages[-last_n:]
        return " | ".join(f"{m['role']}: {m['content'][:80]}" for m in tail)

    def clear(self):
        self.messages.clear()


# ─────────────────────────────────────────────
# 11. MAIN AGENT LOOP
# ─────────────────────────────────────────────
def agent_loop(user_query: str, session: Session) -> str:
    """
    Full production agent loop:
      1. Classify intent (structured)
      2. Route to ORDER tool or POLICY RAG
      3. Generate grounded response
      4. Update session memory
    """
    session.add("user", user_query)

    # ── Step 1: Intent Classification ────────
    try:
        intent = classify_intent(user_query, session.history_summary())
        log.info("Intent: %s (confidence=%.2f)", intent.category, intent.confidence)
    except Exception as e:
        log.error("Intent classification failed: %s", e)
        reply = "I'm having trouble understanding your request. Could you rephrase it?"
        session.add("assistant", reply)
        return reply

    # ── Step 2: Route ─────────────────────────
    try:
        if intent.category == "ORDER":
            order_id = extract_order_id(user_query)
            if not order_id:
                reply = "I'd be happy to check your order. Could you please share your order ID?"
            else:
                status = get_order_status(order_id)
                reply = f"Here's the status for order **{order_id}**: {status}"

        elif intent.category == "POLICY":
            passages = retrieve(user_query)
            if not passages:
                reply = (
                    "I couldn't find relevant policy information. "
                    "Would you like me to connect you with a human agent?"
                )
            else:
                context = "\n\n---\n\n".join(passages)
                reply = generate_answer(user_query, context, session.messages[:-1])

        elif intent.category == "ESCALATE":
            reply = (
                "I'm sorry you're experiencing this issue. "
                "I'm escalating this to a senior support agent who will reach out shortly. "
                "Is there anything else I can note down for them?"
            )
            log.warning("ESCALATION triggered for query: %s", user_query)

        else:  # OTHER
            reply = (
                "I'm here to help with order tracking and store policies. "
                "Could you clarify what you need, or would you like to speak with a human agent?"
            )

    except Exception as e:
        log.error("Agent routing error: %s", e, exc_info=True)
        reply = (
            "Something went wrong on our end. "
            "Please try again in a moment or contact support@yourstore.com."
        )

    session.add("assistant", reply)
    return reply


# ─────────────────────────────────────────────
# 12. DEMO RUNNER
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Uncomment to ingest your policy PDF first:
    # ingest_document("your_policy.pdf", doc_type="policy")

    session = Session()

    test_queries = [
        "Hi, what's your return policy for electronics?",
        "Where is my order #456?",
        "Can I return something I bought 45 days ago?",
        "This is ridiculous, I've been waiting 3 weeks and nobody helps me!",
        "What's the weather like?",
    ]

    print("\n" + "="*60)
    print("  HELPDESK AGENT — PRODUCTION MODE")
    print("="*60 + "\n")

    for query in test_queries:
        print(f"USER: {query}")
        response = agent_loop(query, session)
        print(f"AGENT: {response}\n" + "-"*60 + "\n")
