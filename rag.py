"""
RAG pipeline: embed query, retrieve top-k chunks from Chroma, apply retrieval gate,
build prompt, call LM Studio for answer. Returns structured response with answer,
sources, and confidence note. Conservative for legal use: weak retrieval → refuse.
"""

import re
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from openai import APIConnectionError, OpenAI

from config import (
    CHROMA_PATH,
    CHAT_MODEL,
    EMBEDDING_MODEL,
    LM_STUDIO_API_KEY,
    LM_STUDIO_BASE_URL,
    TOP_K,
)
from ingest import get_embedding

# Retrieval sufficiency gate: do not call LLM below these thresholds
MIN_CHUNKS = 2
MIN_COMBINED_LENGTH = 400
# Optional: reject if average L2 distance is above this (weaker match)
MAX_AVG_DISTANCE = 1.5

SAFETY_PHRASE = "I do not have enough verified legal context to answer this safely."
DRAFT_WARNING = "Draft only for internal use. Lawyer review may be required before external use."


def _normalize_reply(text: str) -> str:
    """
    Normalize model reply for refusal detection: trim, collapse internal whitespace,
    lowercase, and strip one trailing period. Enables reliable equality check against
    the safety phrase without substring false positives.
    """
    if not text or not text.strip():
        return ""
    s = text.strip().lower()
    s = re.sub(r"\s+", " ", s)
    if s.endswith("."):
        s = s[:-1].strip()
    return s


@dataclass
class RAGSource:
    """One retrieved chunk with metadata."""
    text: str
    filename: str
    page: int
    chunk_index: int
    distance: float  # lower = more similar in Chroma
    source_type: str = ""
    section_heading: str = ""
    article_marker: str = ""


@dataclass
class RAGResponse:
    """Structured RAG answer with sources and confidence."""
    answer: str
    sources: list[RAGSource]
    confidence_note: str


@dataclass
class DraftResponse:
    """Draft-mode response with first draft and supporting sources."""
    draft: str
    sources: list[RAGSource]
    warning: str


SYSTEM_PROMPT = """You are a legal information retrieval assistant for internal testing only. You are NOT a lawyer and cannot give final legal advice.

Rules:
- Answer ONLY from the retrieved legal context provided below. Do not use outside knowledge, assumptions, or memory.
- Do not invent legal rules, article numbers, or conclusions.
- Clearly separate: (1) user facts, (2) retrieved legal/source text, (3) explanation based only on that retrieved text.
- If the context is incomplete, ambiguous, conflicting, or insufficient to answer safely, you MUST reply with exactly: "I do not have enough verified legal context to answer this safely."
- Say when licensed lawyer review is needed. Do not claim certainty unless directly supported by retrieved text.
- Use source metadata (filename, page, section) when available to cite clearly.

Output format (use these exact section headers in this order):
1. Answer
2. Why I answered this
3. Source excerpts used
4. What is missing / what needs lawyer review"""

DRAFT_SYSTEM_PROMPT = """You are an internal legal drafting assistant.

Rules:
- Generate a FIRST DRAFT only, based strictly on the retrieved context and user facts.
- Do not claim final legal correctness.
- Do not add legal rules or facts that are not in the retrieved context.
- If context is missing, state assumptions explicitly and mark placeholders for lawyer review.
- Keep language practical and editable by legal staff.

Output format (use these exact headers):
1. Draft only notice
2. Proposed first draft
3. Supporting context used
4. Missing points / lawyer review required"""


def get_chroma_collection():
    """Return the persistent Chroma collection for legal_docs."""
    client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection(name="legal_docs", metadata={"description": "Legal document chunks"})


def has_sufficient_retrieval_context(sources: list[RAGSource]) -> bool:
    """
    Return True only if we have enough verified context to call the LLM.
    Requires: at least MIN_CHUNKS, combined text length >= MIN_COMBINED_LENGTH,
    and optionally average distance below MAX_AVG_DISTANCE.
    """
    if not sources or len(sources) < MIN_CHUNKS:
        return False
    combined_length = sum(len(s.text or "") for s in sources)
    if combined_length < MIN_COMBINED_LENGTH:
        return False
    # Optional distance check: if avg distance too high, treat as weak match
    distances = [s.distance for s in sources if hasattr(s, "distance")]
    if distances:
        avg = sum(distances) / len(distances)
        if avg > MAX_AVG_DISTANCE:
            return False
    return True


def retrieve_with_filter(
    openai_client: OpenAI,
    collection,
    query: str,
    top_k: int = TOP_K,
    where: dict | None = None,
) -> list[RAGSource]:
    """Embed the query and retrieve top_k chunks from Chroma with optional metadata filter."""
    count = collection.count()
    if count == 0:
        return []
    query_embedding = get_embedding(openai_client, query)
    n_results = min(top_k, count)
    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_args["where"] = where
    results = collection.query(**query_args)
    if not results or not results["ids"] or not results["ids"][0]:
        return []

    sources = []
    for i, _doc_id in enumerate(results["ids"][0]):
        doc = results["documents"][0][i] if results["documents"] else ""
        meta = (results["metadatas"][0][i] or {}) if results["metadatas"] else {}
        dist = (results["distances"][0][i] if results.get("distances") and results["distances"][0] else 0.0)
        sources.append(
            RAGSource(
                text=doc,
                filename=meta.get("filename", "unknown"),
                page=int(meta.get("page", 0)),
                chunk_index=int(meta.get("chunk_index", 0)),
                distance=float(dist),
                source_type=meta.get("source_type", ""),
                section_heading=meta.get("section_heading", ""),
                article_marker=meta.get("article_marker", ""),
            )
        )
    return sources


def retrieve(openai_client: OpenAI, collection, query: str, top_k: int = TOP_K) -> list[RAGSource]:
    """
    Embed the query, retrieve top_k chunks from Chroma, return list of RAGSource.
    """
    return retrieve_with_filter(openai_client=openai_client, collection=collection, query=query, top_k=top_k)


def search_chunks(query: str, top_k: int = TOP_K) -> list[RAGSource]:
    """Search mode retrieval: return matching chunks only, no generation."""
    openai_client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)
    collection = get_chroma_collection()
    try:
        return retrieve(openai_client, collection, query, top_k=top_k)
    except APIConnectionError as e:
        raise RuntimeError(
            "Cannot reach LM Studio for embeddings. Is it running at "
            f"{LM_STUDIO_BASE_URL}? Start LM Studio and load the embedding model."
        ) from e


def build_confidence_note(
    sources: list[RAGSource],
    retrieval_sufficient: bool,
    model_was_called: bool,
    model_refused: bool = False,
) -> str:
    """
    Confidence note: distinguish (1) gate blocked / model not called,
    (2) model was called but refused, (3) model returned a real answer.
    """
    if not retrieval_sufficient or not model_was_called:
        return (
            "Retrieval was insufficient (chunks or combined length below threshold, or distance too high); "
            "the model was not called."
        )
    if model_refused:
        return (
            "Retrieval passed, but the model still found the context insufficient or ambiguous and refused. "
            "No answer was produced from sources."
        )
    if not sources:
        return "No relevant chunks retrieved; confidence is none."
    n = len(sources)
    avg_dist = sum(s.distance for s in sources) / n if n else 0
    return f"Retrieved {n} chunk(s); average distance {avg_dist:.4f}. Answer is based only on these sources."


def build_user_message(query: str, sources: list[RAGSource]) -> str:
    """Build the user message with all chunks and metadata for the model. Easy to inspect and debug."""
    blocks = []
    for i, s in enumerate(sources, start=1):
        parts = [f"[Chunk {i} | {s.filename} | page {s.page}"]
        if s.source_type:
            parts.append(f" | type: {s.source_type}")
        if s.section_heading:
            parts.append(f" | section: {s.section_heading}")
        if s.article_marker:
            parts.append(f" | {s.article_marker}")
        parts.append("]")
        blocks.append("".join(parts) + "\n" + (s.text or ""))
    context = "\n\n---\n\n".join(blocks)
    return f"""Retrieved legal context:

{context}

---

User question: {query}

Respond using the required format (Answer, Why I answered this, Source excerpts used, What is missing / what needs lawyer review). If the context above does not support a safe answer, or is incomplete, ambiguous, or conflicting, you MUST say exactly: "I do not have enough verified legal context to answer this safely.\""""


def answer_with_llm(openai_client: OpenAI, query: str, sources: list[RAGSource]) -> str:
    """
    Build user message with context and metadata, call LM Studio chat, return assistant reply.
    Call only when has_sufficient_retrieval_context(sources) is True.
    """
    user_message = build_user_message(query, sources)
    try:
        resp = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        reply = (resp.choices[0].message.content or "").strip()
        return reply if reply else SAFETY_PHRASE
    except APIConnectionError:
        raise
    except Exception as e:
        raise RuntimeError(f"Chat model request failed: {e}") from e


def _dedupe_sources(sources: list[RAGSource]) -> list[RAGSource]:
    """Deduplicate retrieved sources while preserving order."""
    seen = set()
    out = []
    for s in sources:
        key = (s.filename, s.page, s.chunk_index, s.text)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def build_draft_user_message(document_type: str, user_facts: str, notes: str, sources: list[RAGSource]) -> str:
    """Build prompt for Draft mode with template + supporting context."""
    blocks = []
    for i, s in enumerate(sources, start=1):
        parts = [f"[Source {i} | {s.filename} | page {s.page}"]
        if s.source_type:
            parts.append(f" | type: {s.source_type}")
        if s.section_heading:
            parts.append(f" | section: {s.section_heading}")
        if s.article_marker:
            parts.append(f" | {s.article_marker}")
        parts.append("]")
        blocks.append("".join(parts) + "\n" + (s.text or ""))
    context = "\n\n---\n\n".join(blocks)

    return f"""Retrieved drafting context:

{context}

---

Requested document type: {document_type}
User facts: {user_facts}
Optional notes: {notes or "(none)"}

Create a first draft based only on the retrieved context and user facts. Include this warning in section 1: {DRAFT_WARNING}"""


def generate_draft(document_type: str, user_facts: str, notes: str = "") -> DraftResponse:
    """Draft mode: retrieve template + legal/firm chunks, then generate a first draft."""
    openai_client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)
    collection = get_chroma_collection()
    query = f"{document_type}\n{user_facts}\n{notes}".strip()

    try:
        template_sources = retrieve_with_filter(
            openai_client=openai_client,
            collection=collection,
            query=query,
            top_k=4,
            where={"source_type": "template"},
        )
        supporting_sources = retrieve_with_filter(
            openai_client=openai_client,
            collection=collection,
            query=query,
            top_k=4,
            where={"source_type": {"$in": ["law", "regulation", "firm_note", "unknown"]}},
        )
    except APIConnectionError as e:
        raise RuntimeError(
            "Cannot reach LM Studio for embeddings. Is it running at "
            f"{LM_STUDIO_BASE_URL}? Start LM Studio and load the embedding model."
        ) from e

    sources = _dedupe_sources(template_sources + supporting_sources)
    if not sources:
        return DraftResponse(
            draft=(
                "1. Draft only notice\n"
                f"{DRAFT_WARNING}\n\n"
                "2. Proposed first draft\n"
                "Insufficient local drafting context was retrieved.\n\n"
                "3. Supporting context used\n"
                "No sources retrieved.\n\n"
                "4. Missing points / lawyer review required\n"
                "Please ingest relevant templates, laws, regulations, or firm notes and retry."
            ),
            sources=[],
            warning=DRAFT_WARNING,
        )

    user_message = build_draft_user_message(document_type, user_facts, notes, sources)
    try:
        resp = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": DRAFT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        draft_text = (resp.choices[0].message.content or "").strip()
    except APIConnectionError as e:
        raise RuntimeError(
            "Cannot reach LM Studio. Is it running at "
            f"{LM_STUDIO_BASE_URL}? Start LM Studio and load the chat model."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Chat model request failed: {e}") from e

    if not draft_text:
        draft_text = (
            "1. Draft only notice\n"
            f"{DRAFT_WARNING}\n\n"
            "2. Proposed first draft\nNo draft could be generated.\n\n"
            "3. Supporting context used\nSee listed sources.\n\n"
            "4. Missing points / lawyer review required\nLawyer review required."
        )

    return DraftResponse(draft=draft_text, sources=sources, warning=DRAFT_WARNING)


def is_sufficient_context(reply: str) -> bool:
    """
    True if the model gave a real answer rather than refusing with the safety phrase.
    Uses normalized equality (not substring) so the phrase inside a larger answer
    does not count as refusal.
    """
    norm_reply = _normalize_reply(reply)
    norm_safety = _normalize_reply(SAFETY_PHRASE)
    return norm_reply != norm_safety


def query_rag(user_query: str) -> RAGResponse:
    """
    Full RAG pipeline: embed query → retrieve top-k → gate check → build prompt → get LLM answer.
    If retrieval is insufficient, returns safety phrase immediately without calling the model.
    Raises RuntimeError if LM Studio is not reachable for embedding or chat.
    """
    openai_client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)
    collection = get_chroma_collection()

    try:
        sources = retrieve(openai_client, collection, user_query)
    except APIConnectionError as e:
        raise RuntimeError(
            "Cannot reach LM Studio. Is it running at "
            f"{LM_STUDIO_BASE_URL}? Start LM Studio and load the embedding model."
        ) from e

    if not has_sufficient_retrieval_context(sources):
        confidence_note = build_confidence_note(sources, retrieval_sufficient=False, model_was_called=False)
        return RAGResponse(answer=SAFETY_PHRASE, sources=sources, confidence_note=confidence_note)

    try:
        answer = answer_with_llm(openai_client, user_query, sources)
    except APIConnectionError as e:
        raise RuntimeError(
            "Cannot reach LM Studio. Is it running at "
            f"{LM_STUDIO_BASE_URL}? Start LM Studio and load the chat model."
        ) from e

    model_refused = not is_sufficient_context(answer)
    confidence_note = build_confidence_note(
        sources, retrieval_sufficient=True, model_was_called=True, model_refused=model_refused
    )
    return RAGResponse(answer=answer, sources=sources, confidence_note=confidence_note)
