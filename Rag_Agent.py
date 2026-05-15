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
import json
import logging
import uuid
from typing import Any, Literal, Optional
from pathlib import Path
from dataclasses import dataclass, field

import requests
from pypdf import PdfReader
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from qdrant_client import QdrantClient, models
from qdrant_client.hybrid.fusion import reciprocal_rank_fusion
from openai import OpenAI
import cohere
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Logging ──────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

load_dotenv()

# ── Credentials (Colab) ──────────────────────
try:
    from google.colab import userdata
    GROQ_API_KEY    = userdata.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    COHERE_API_KEY  = userdata.get("COHERE_API_KEY") or os.getenv("COHERE_API_KEY")
    QDRANT_URL      = userdata.get("QDRANT_URL") or os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY  = userdata.get("QDRANT_API_KEY") or os.getenv("QDRANT_API_KEY")
    ORDER_API_URL   = userdata.get("ORDER_API_URL") or os.getenv("ORDER_API_URL", "http://localhost:8000")
    LLM_BASE_URL    = userdata.get("LLM_BASE_URL") or os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    FAST_LLM_MODEL  = userdata.get("FAST_LLM_MODEL") or os.getenv("FAST_LLM_MODEL", "llama-3.1-8b-instant")
    FINAL_LLM_MODEL = userdata.get("FINAL_LLM_MODEL") or os.getenv("FINAL_LLM_MODEL", "llama-3.3-70b-versatile")
except Exception:
    # Fallback to environment variables (local / CI)
    GROQ_API_KEY    = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    COHERE_API_KEY  = os.getenv("COHERE_API_KEY")
    QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
    ORDER_API_URL   = os.getenv("ORDER_API_URL", "http://localhost:8000")
    LLM_BASE_URL    = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    FAST_LLM_MODEL  = os.getenv("FAST_LLM_MODEL", "llama-3.1-8b-instant")
    FINAL_LLM_MODEL = os.getenv("FINAL_LLM_MODEL", "llama-3.3-70b-versatile")

COLLECTION_NAME  = "helpdesk_policy"
BASE_DIR         = Path(__file__).resolve().parent
POLICY_DOC_PATH  = os.getenv("POLICY_DOC_PATH", str(BASE_DIR / "Store_Return_Policy.pdf"))
CHUNK_SIZE       = 512
CHUNK_OVERLAP    = 64
RETRIEVAL_LIMIT  = 6    # fetch more, reranker will cut down
RERANK_TOP_N     = 3
RERANK_MODEL     = "rerank-v3.5"
PARTITION_STRATEGY = os.getenv("UNSTRUCTURED_STRATEGY", "fast")
UNSUPPORTED_ANSWER = (
    "I don't have that information. Would you like me to connect you with a human agent?"
)

SENSITIVE_INPUT_PATTERNS = {
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "password": re.compile(r"\b(?:password|passcode|pin)\s*[:=]\s*\S+", re.IGNORECASE),
    "api_key": re.compile(r"\b(?:sk|gsk|pk|rk)_[A-Za-z0-9_-]{20,}\b"),
}
TOXIC_INPUT_PATTERN = re.compile(
    r"\b(?:kill yourself|kys|worthless|idiot|stupid|moron)\b",
    re.IGNORECASE,
)


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
    c.set_sparse_model("Qdrant/bm25")            # Sparse
    return c


qdrant  = build_qdrant_client()
llm     = OpenAI(
    api_key=GROQ_API_KEY or "missing-groq-api-key",
    base_url=LLM_BASE_URL,
)
co      = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None


def ensure_policy_collection() -> None:
    """
    Create the Qdrant collection used by the policy RAG path if needed.
    """
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME in existing:
        return

    log.info("Creating collection '%s'", COLLECTION_NAME)
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qdrant.get_fastembed_vector_params(),
        sparse_vectors_config=qdrant.get_fastembed_sparse_vector_params(),
    )


def point_id_for_chunk(source: str, chunk_index: int) -> str:
    """
    Stable IDs make repeated ingestion update chunks instead of duplicating them.
    """
    return uuid.uuid5(uuid.NAMESPACE_URL, f"{COLLECTION_NAME}:{source}:{chunk_index}").hex


# ─────────────────────────────────────────────
# 4. INGESTION — Chunked, Structure-Aware
# ─────────────────────────────────────────────
def infer_document_metadata(path: Path, doc_type: str) -> dict[str, Any]:
    """
    Infer simple filterable metadata from the file name and document type.
    Explicit metadata can be layered on top when ingesting.
    """
    year_match = re.search(r"\b(20\d{2})\b", path.stem)
    inferred = {
        "source": path.name,
        "type": doc_type,
    }
    if year_match:
        inferred["year"] = int(year_match.group(1))
    return inferred


def build_metadata_filter(metadata_filter: Optional[dict[str, Any]] = None) -> Optional[models.Filter]:
    """
    Convert a simple equality/range metadata filter into a Qdrant filter.
    Supported range keys: year_min, year_max.
    """
    if not metadata_filter:
        return None

    must: list[Any] = []
    for key, value in metadata_filter.items():
        if value is None:
            continue
        if key in {"year_min", "year_max"}:
            continue
        must.append(
            models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value),
            )
        )

    range_kwargs = {
        name: metadata_filter[source_key]
        for name, source_key in (("gte", "year_min"), ("lte", "year_max"))
        if metadata_filter.get(source_key) is not None
    }
    if range_kwargs:
        must.append(
            models.FieldCondition(
                key="year",
                range=models.Range(**range_kwargs),
            )
        )

    return models.Filter(must=must) if must else None


def ingest_document(
    file_path: str,
    doc_type: str = "policy",
    metadata: Optional[dict[str, Any]] = None,
) -> int:
    """
    Parse a document, chunk it properly, and upsert into Qdrant.
    Returns the number of chunks ingested.
    """
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path

    log.info("Partitioning document: %s", path)
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        raw_text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        from unstructured.partition.auto import partition

        elements = partition(filename=str(path), strategy=PARTITION_STRATEGY)
        raw_text = "\n\n".join(
            str(el) for el in elements
            if el.category in {"NarrativeText", "Table", "ListItem", "Title"}
        )

    if not raw_text.strip():
        raise ValueError(f"No usable text extracted from {path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(raw_text)
    log.info("Split into %d chunks", len(chunks))

    ensure_policy_collection()

    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=path.name),
                    )
                ]
            )
        ),
        wait=True,
    )

    base_metadata = infer_document_metadata(path, doc_type)
    if metadata:
        base_metadata.update(metadata)

    points = [
        models.PointStruct(
            id=point_id_for_chunk(path.name, i),
            vector={
                qdrant.get_vector_field_name(): models.Document(
                    text=chunk,
                    model=qdrant.embedding_model_name,
                ),
                qdrant.get_sparse_vector_field_name(): models.Document(
                    text=chunk,
                    model=qdrant.sparse_embedding_model_name,
                ),
            },
            payload={
                **base_metadata,
                "document": chunk,
                "chunk_index": i,
                "parent_context": "\n\n".join(chunks[max(0, i - 1):min(len(chunks), i + 2)]),
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    log.info("Ingestion complete: %d chunks stored.", len(chunks))
    return len(chunks)


# ─────────────────────────────────────────────
# 5. RETRIEVAL — Hybrid + Rerank
# ─────────────────────────────────────────────
def retrieve(query: str, metadata_filter: Optional[dict[str, Any]] = None) -> list[str]:
    """
    Hybrid search -> Cohere reranking -> top-N parent passages.
    Small chunks are searched and reranked; larger neighboring context is sent
    to the answer model.
    """
    qdrant_filter = build_metadata_filter(metadata_filter)
    dense_response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=models.Document(text=query, model=qdrant.embedding_model_name),
        using=qdrant.get_vector_field_name(),
        query_filter=qdrant_filter,
        limit=RETRIEVAL_LIMIT,
        with_payload=True,
    )
    sparse_response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=models.Document(text=query, model=qdrant.sparse_embedding_model_name),
        using=qdrant.get_sparse_vector_field_name(),
        query_filter=qdrant_filter,
        limit=RETRIEVAL_LIMIT,
        with_payload=True,
    )
    hits = reciprocal_rank_fusion(
        [dense_response.points, sparse_response.points],
        limit=RETRIEVAL_LIMIT,
    )
    candidates = [
        {
            "document": h.payload.get("document", ""),
            "parent_context": h.payload.get("parent_context") or h.payload.get("document", ""),
            "source": h.payload.get("source", "unknown"),
            "chunk_index": h.payload.get("chunk_index", "unknown"),
        }
        for h in hits
        if h.payload and h.payload.get("document")
    ]

    if not candidates:
        return []

    docs = [candidate["document"] for candidate in candidates]
    if co:
        try:
            reranked = co.rerank(
                query=query,
                documents=docs,
                model=RERANK_MODEL,
                top_n=RERANK_TOP_N,
            )
            return [
                (
                    f"Source: {candidates[r.index]['source']} "
                    f"(chunk {candidates[r.index]['chunk_index']})\n"
                    f"{candidates[r.index]['parent_context']}"
                )
                for r in reranked.results
            ]
        except Exception as e:
            log.warning("Cohere rerank failed (%s), using raw hits", e)

    return [
        (
            f"Source: {candidate['source']} (chunk {candidate['chunk_index']})\n"
            f"{candidate['parent_context']}"
        )
        for candidate in candidates[:RERANK_TOP_N]
    ]


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
        resp = llm.chat.completions.create(
            model=FAST_LLM_MODEL,
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


class GuardrailResult(BaseModel):
    allowed: bool
    reason: str = ""
    response: str = ""


class CriticResult(BaseModel):
    supported: bool
    reason: str = ""


def check_input_guardrails(user_query: str) -> GuardrailResult:
    """
    Deterministic pre-LLM guardrails for obvious sensitive or abusive input.
    Advanced moderation providers can replace this boundary later.
    """
    if TOXIC_INPUT_PATTERN.search(user_query):
        return GuardrailResult(
            allowed=False,
            reason="toxic_input",
            response=(
                "I can help with the support issue, but I can't continue with abusive language. "
                "Please rephrase your request and I'll help from there."
            ),
        )

    matched_sensitive_types = [
        name for name, pattern in SENSITIVE_INPUT_PATTERNS.items()
        if pattern.search(user_query)
    ]
    if matched_sensitive_types:
        return GuardrailResult(
            allowed=False,
            reason="sensitive_input",
            response=(
                "Please remove sensitive details like full card numbers, passwords, SSNs, "
                "or API keys before sending your message. I can still help with an order ID "
                "or a general policy question."
            ),
        )

    return GuardrailResult(allowed=True)


def extract_metadata_filter(query: str) -> dict[str, Any]:
    """
    Pull simple metadata constraints from natural language.
    Example: "only search 2026 policies" -> {"year": 2026, "type": "policy"}.
    """
    metadata_filter: dict[str, Any] = {}
    year_match = re.search(r"\b(20\d{2})\b", query)
    if year_match:
        metadata_filter["year"] = int(year_match.group(1))

    if re.search(r"\bpolic(?:y|ies)\b", query, re.IGNORECASE):
        metadata_filter["type"] = "policy"

    return metadata_filter


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def classify_intent(query: str, history_summary: str = "") -> IntentResult:
    """
    Use Groq JSON object mode and validate the result with Pydantic.
    """
    system = (
        "You are an intent classifier for a customer helpdesk. "
        "Classify the user's query into exactly one category:\n"
        "  ORDER    — tracking, status, shipping, cancellation of a specific order\n"
        "  POLICY   — returns, refunds, warranty, store rules, general how-to\n"
        "  ESCALATE — angry customer, complaint, legal threat, abuse\n"
        "  OTHER    — anything else\n"
        "Treat order IDs like #456, ORD-001, tracking numbers, or phrases like "
        "'where is my order' as strong ORDER signals.\n"
        "Return only a valid JSON object with these keys: "
        "category, confidence, reasoning. "
        "category must be ORDER, POLICY, ESCALATE, or OTHER. "
        "confidence must be a number between 0 and 1."
    )
    context = f"Conversation so far: {history_summary}\n\n" if history_summary else ""
    resp = llm.chat.completions.create(
        model=FAST_LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"{context}User query: {query}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=200,
    )
    content = resp.choices[0].message.content or "{}"
    return IntentResult(**json.loads(content))


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

    resp = llm.chat.completions.create(
        model=FINAL_LLM_MODEL,
        messages=payload,
        temperature=0.2,     # low temp for factual helpdesk responses
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4))
def critique_answer(query: str, context: str, answer: str) -> CriticResult:
    """
    Second-pass faithfulness check. The critic must reject answers that add
    policy claims not supported by the retrieved context.
    """
    resp = llm.chat.completions.create(
        model=FAST_LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict RAG faithfulness critic. Decide whether the answer is "
                    "fully supported by the reference context. Return only JSON with keys "
                    "supported and reason."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Reference context:\n{context}\n\n"
                    f"Question: {query}\n\n"
                    f"Answer: {answer}"
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=160,
    )
    content = resp.choices[0].message.content or "{}"
    return CriticResult(**json.loads(content))


def generate_verified_answer(query: str, context: str, messages: list[dict]) -> str:
    """
    Generate an answer, then fall back if a critic cannot verify support.
    """
    answer = generate_answer(query, context, messages)
    try:
        critic = critique_answer(query, context, answer)
        if not critic.supported:
            log.warning("Answer rejected by critic: %s", critic.reason)
            return UNSUPPORTED_ANSWER
    except Exception as e:
        log.warning("Answer critic failed (%s), returning generated answer", e)
    return answer


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
    guardrail = check_input_guardrails(user_query)
    if not guardrail.allowed:
        log.warning("Guardrail blocked query: %s", guardrail.reason)
        return guardrail.response

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
            passages = retrieve(user_query, metadata_filter=extract_metadata_filter(user_query))
            if not passages:
                reply = (
                    "I couldn't find relevant policy information. "
                    "Would you like me to connect you with a human agent?"
                )
            else:
                context = "\n\n---\n\n".join(passages)
                reply = generate_verified_answer(user_query, context, session.messages[:-1])

        elif intent.category == "ESCALATE":
            reply = (
                "I'm sorry you're experiencing this issue. "
                "I'm escalating this to a senior support agent who will reach out shortly. "
                "Is there anything else I can note down for them?"
            )
            log.warning("ESCALATION triggered for query: %s", user_query)

        else:  # OTHER
            reply = (
                "I can help with order tracking, returns, refunds, and store policies. "
                "For anything outside that, I can connect you with a human agent."
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
    ingest_document(POLICY_DOC_PATH, doc_type="policy")

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
