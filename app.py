"""
FastAPI app for the local legal hub: health, ingest, search, ask, draft, and a simple HTML UI.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ingest import run_ingest
from rag import DraftResponse, RAGResponse, RAGSource, generate_draft, query_rag, search_chunks

app = FastAPI(title="Internal Legal Hub API", description="Local legal hub with Search, Ask, and Draft modes")

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


class AskRequest(BaseModel):
    question: str


class SearchRequest(BaseModel):
    query: str


class DraftRequest(BaseModel):
    document_type: str
    user_facts: str
    notes: str = ""


class SourceInResponse(BaseModel):
    """One retrieved source returned by Search, Ask, or Draft."""

    text: str
    filename: str
    page: int
    chunk_index: int
    distance: float
    source_type: str = ""
    section_heading: str = ""
    article_marker: str = ""


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceInResponse]
    confidence_note: str


class SearchResponse(BaseModel):
    query: str
    results: list[SourceInResponse]


class DraftAPIResponse(BaseModel):
    draft: str
    sources: list[SourceInResponse]
    warning: str


def source_to_dict(s: RAGSource) -> dict:
    return {
        "text": s.text,
        "filename": s.filename,
        "page": s.page,
        "chunk_index": s.chunk_index,
        "distance": s.distance,
        "source_type": s.source_type or "",
        "section_heading": s.section_heading or "",
        "article_marker": s.article_marker or "",
    }


@app.get("/health")
def health():
    """Health check for the API and optional dependency checks."""
    return {"status": "ok"}


@app.post("/ingest")
def ingest():
    """
    Run ingestion: scan data dir, chunk, embed via LM Studio, store in Chroma.
    Returns summary (files_processed, chunks_added, chunks_skipped, errors).
    """
    try:
        result = run_ingest()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """Retrieve indexed chunks only (no generation)."""
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    try:
        results = search_chunks(query)
        return SearchResponse(query=query, results=[SourceInResponse(**source_to_dict(s)) for s in results])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Answer a question using RAG: embed query, retrieve top chunks, generate answer from context.
    Returns answer, sources used, and a confidence note.
    """
    if not (req.question or req.question.strip()):
        raise HTTPException(status_code=400, detail="question is required")
    try:
        rag_result: RAGResponse = query_rag(req.question.strip())
        return AskResponse(
            answer=rag_result.answer,
            sources=[SourceInResponse(**source_to_dict(s)) for s in rag_result.sources],
            confidence_note=rag_result.confidence_note,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/draft", response_model=DraftAPIResponse)
def draft(req: DraftRequest):
    """Generate first legal draft from template + supporting retrieved context."""
    document_type = (req.document_type or "").strip()
    user_facts = (req.user_facts or "").strip()
    notes = (req.notes or "").strip()

    if not document_type:
        raise HTTPException(status_code=400, detail="document_type is required")
    if not user_facts:
        raise HTTPException(status_code=400, detail="user_facts is required")

    try:
        draft_result: DraftResponse = generate_draft(document_type=document_type, user_facts=user_facts, notes=notes)
        return DraftAPIResponse(
            draft=draft_result.draft,
            sources=[SourceInResponse(**source_to_dict(s)) for s in draft_result.sources],
            warning=draft_result.warning,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
@app.get("/search", response_class=HTMLResponse)
def index():
    """Serve the HTML legal hub UI with Search, Ask, and Draft tabs."""
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            content="<p>Template not found. Create templates/index.html.</p>",
            status_code=404,
        )
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
