"""
Ingest legal documents: recursively scan data dir, extract text (preserving line breaks for structure),
chunk (structure-aware when possible), embed via LM Studio, store in Chroma. Dedup by chunk id
(filename|page|text) so identical text in different files keeps separate provenance.
"""

import hashlib
import re
from pathlib import Path

from pypdf import PdfReader

import chromadb
from chromadb.config import Settings
from openai import APIConnectionError, OpenAI

from config import (
    CHROMA_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_MODEL,
    LM_STUDIO_API_KEY,
    LM_STUDIO_BASE_URL,
    SUPPORTED_EXTENSIONS,
)

# Structure-aware splitting: split on article/section/markdown headings (start of string or after newline)
ARTICLE_SECTION_PATTERN = re.compile(
    r"(?:^|\n)(?=\s*(?:Article\s+\d+|ARTICLE\s+\d+|Section\s+\d+|المادة\s+\d+|\#+\s+))",
    re.IGNORECASE | re.MULTILINE,
)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single space and strip. Use after structural segmentation for chunk cleanup."""
    if not text or not text.strip():
        return ""
    return re.sub(r"\s+", " ", text).strip()


def normalize_text_preserve_lines(text: str) -> str:
    """
    Clean text for extraction while preserving line breaks so structure-aware chunking can detect
    article/section/markdown headings. Collapses horizontal whitespace (spaces/tabs) per line only.
    """
    if not text or not text.strip():
        return ""
    # Normalize line endings, then collapse runs of spaces/tabs on each line; leave newlines intact
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines)


def get_source_type(file_path: Path) -> str:
    """
    Derive source_type from file path for filtering. Uses simple path heuristics.
    """
    path_lower = str(file_path).lower()
    if "laws" in path_lower:
        return "law"
    if "regulations" in path_lower:
        return "regulation"
    if "templates" in path_lower:
        return "template"
    if "firm_notes" in path_lower:
        return "firm_note"
    return "unknown"


def extract_first_line_marker(line: str) -> tuple[str, str]:
    """
    If the line looks like an article/section heading or markdown heading, return (section_heading, article_marker).
    Otherwise ("", ""). Used to tag chunks from structure-aware split.
    """
    line = (line or "").strip()
    if not line:
        return "", ""
    # Markdown heading
    if line.startswith("#"):
        return line[:80], ""
    # Article / Section
    if re.match(r"^(?:Article|ARTICLE|Section)\s+\d+", line, re.IGNORECASE):
        return "", line[:80]
    if re.match(r"^المادة\s+\d+", line):
        return "", line[:80]
    return "", ""


def split_by_structure(text: str) -> list[tuple[str, str, str]]:
    """
    Try to split text on article/section/markdown markers. Returns list of (segment_text, section_heading, article_marker).
    If no markers found, returns [(text, "", "")].
    """
    if not text or not text.strip():
        return []
    parts = ARTICLE_SECTION_PATTERN.split(text)
    result = []
    for i, seg in enumerate(parts):
        seg = seg.strip()
        if not seg:
            continue
        section_heading = ""
        article_marker = ""
        # First line of segment may be the marker (including at start of document)
        first_line, rest = seg.split("\n", 1) if "\n" in seg else (seg, "")
        if first_line:
            section_heading, article_marker = extract_first_line_marker(first_line)
            if section_heading or article_marker:
                seg = (first_line + "\n" + rest).strip() if rest else first_line
            else:
                section_heading = ""
                article_marker = ""
        result.append((seg, section_heading, article_marker))
    if not result:
        return [(text.strip(), "", "")]
    return result


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into chunks of roughly chunk_size characters with overlap.
    Tries to break at sentence or word boundaries when possible. Fallback when no structure found.
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # Prefer break at sentence end, then space
            search = text.rfind(". ", start, end + 1)
            if search > start:
                end = search + 1
            else:
                search = text.rfind(" ", start, end + 1)
                if search > start:
                    end = search + 1
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
        if start >= len(text):
            break

    return chunks


def chunk_one_block(
    text: str,
    chunk_size: int,
    overlap: int,
    section_heading: str,
    article_marker: str,
) -> list[tuple[str, str, str]]:
    """
    Chunk a single segment: use structure-aware split first; if segment still too long, use chunk_text.
    Returns list of (chunk_text, section_heading, article_marker).
    """
    if not text or not text.strip():
        return []
    # If segment is small enough, one chunk
    if len(text) <= chunk_size:
        return [(text.strip(), section_heading, article_marker)]
    # Otherwise sub-chunk with overlap; same metadata for all sub-chunks (or only first has heading)
    sub_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    return [(c, section_heading, article_marker) for c in sub_chunks]


def chunk_text_with_structure(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[tuple[str, str, str]]:
    """
    First try structure-aware split (articles, sections, markdown headings); then sub-chunk large segments.
    Returns list of (chunk_text, section_heading, article_marker).
    """
    if not text or not text.strip():
        return []
    segments = split_by_structure(text)
    out = []
    for seg, seg_heading, seg_marker in segments:
        out.extend(chunk_one_block(seg, chunk_size, overlap, seg_heading, seg_marker))
    # If no structure was found, split_by_structure returns one segment; chunk_one_block will use chunk_text
    return out if out else [(text.strip(), "", "")]


def extract_text_txt_or_md(file_path: Path) -> list[dict]:
    """
    Extract text from .txt or .md. Returns list of dicts with keys: text, filename, page.
    Page is 1-based; for single-file formats we use page 1.
    Preserves line breaks for structure-aware chunking.
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")
    cleaned = normalize_text_preserve_lines(text)
    if not cleaned:
        return []
    return [{"text": cleaned, "filename": file_path.name, "page": 1}]


def extract_text_pdf(file_path: Path) -> list[dict]:
    """
    Extract text per page from PDF. Returns list of dicts with keys: text, filename, page.
    Preserves line breaks for structure-aware chunking.
    """
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = normalize_text_preserve_lines(text)
        if cleaned:
            pages.append({"text": cleaned, "filename": file_path.name, "page": i})
    return pages


def extract_text(file_path: Path) -> list[dict]:
    """
    Dispatch by extension. Returns list of {text, filename, page}.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_pdf(file_path)
    if suffix in (".txt", ".md"):
        return extract_text_txt_or_md(file_path)
    return []


def chunk_document_pages(
    pages: list[dict],
    source_type: str = "unknown",
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    For each page (or single block), produce chunks with metadata using structure-aware chunking when possible.
    Each chunk dict: text, filename, page, chunk_index, source_type, section_heading, article_marker.
    """
    result = []
    for block in pages:
        text = block["text"]
        filename = block["filename"]
        page = block["page"]
        raw_chunks = chunk_text_with_structure(text, chunk_size=chunk_size, overlap=overlap)
        for idx, (c_text, section_heading, article_marker) in enumerate(raw_chunks):
            # Normalize chunk text after structure-aware split for consistent storage/embedding
            result.append({
                "text": normalize_whitespace(c_text),
                "filename": filename,
                "page": page,
                "chunk_index": idx,
                "source_type": source_type,
                "section_heading": section_heading,
                "article_marker": article_marker,
            })
    return result


def chunk_id(filename: str, page: int, text: str) -> str:
    """
    Stable id for a chunk that includes provenance (filename, page). Same text in different
    files or pages gets different ids so we do not lose source information when deduplicating.
    """
    return hashlib.sha256(f"{filename}|{page}|{text}".encode("utf-8")).hexdigest()


def get_embedding(client: OpenAI, text: str) -> list[float]:
    """Get embedding for a single string from LM Studio."""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def get_embeddings_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Get embeddings for multiple texts. Loops one-by-one for LM Studio compatibility."""
    if not texts:
        return []
    embeddings = []
    for t in texts:
        embeddings.append(get_embedding(client, t))
    return embeddings


def run_ingest() -> dict:
    """
    Recursively scan DATA_DIR, extract and chunk documents (structure-aware), embed, store in Chroma.
    Uses chunk id (filename|page|text) so duplicate text in different files keeps separate provenance.
    Returns summary: files_processed, chunks_added, chunks_skipped, errors (list of strings).
    """
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        return {"files_processed": 0, "chunks_added": 0, "chunks_skipped": 0, "errors": ["DATA_DIR did not exist; created empty."]}

    # Recursively collect supported files under DATA_DIR (including nested folders)
    supported_files = [
        f for f in sorted(data_path.rglob("*"))
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    files_processed = len(supported_files)

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
    collection = chroma_client.get_or_create_collection(name="legal_docs", metadata={"description": "Legal document chunks"})
    client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)

    all_chunks: list[dict] = []
    errors: list[str] = []

    for file_path in supported_files:
        try:
            pages = extract_text(file_path)
            source_type = get_source_type(file_path)
            for block in chunk_document_pages(pages, source_type=source_type):
                all_chunks.append(block)
        except Exception as e:
            errors.append(f"{file_path.name}: {e}")

    chunks_to_embed = all_chunks

    if not chunks_to_embed:
        return {
            "files_processed": files_processed,
            "chunks_added": 0,
            "chunks_skipped": 0,
            "errors": errors,
        }

    # Deduplicate by chunk id (filename|page|text): same text in different files keeps separate provenance
    existing_ids = set(collection.get()["ids"]) if collection.count() else set()
    new_chunks = []
    seen_ids = set()
    for c in chunks_to_embed:
        cid = chunk_id(c["filename"], c["page"], c["text"])
        if cid in existing_ids or cid in seen_ids:
            continue
        seen_ids.add(cid)
        c["id"] = cid
        new_chunks.append(c)

    chunks_skipped = len(chunks_to_embed) - len(new_chunks)
    if not new_chunks:
        return {
            "files_processed": files_processed,
            "chunks_added": 0,
            "chunks_skipped": chunks_skipped,
            "errors": errors,
        }

    # Embed and add to Chroma
    texts = [c["text"] for c in new_chunks]
    try:
        embeddings = get_embeddings_batch(client, texts)
    except APIConnectionError as e:
        errors.append("Cannot reach LM Studio. Is it running? Start LM Studio and load the embedding model.")
        return {
            "files_processed": files_processed,
            "chunks_added": 0,
            "chunks_skipped": chunks_skipped,
            "errors": errors,
        }
    except Exception as e:
        errors.append(f"Embedding failed: {e}")
        return {
            "files_processed": files_processed,
            "chunks_added": 0,
            "chunks_skipped": chunks_skipped,
            "errors": errors,
        }

    ids = [c["id"] for c in new_chunks]
    metadatas = [
        {
            "filename": c["filename"],
            "page": c["page"],
            "chunk_index": c["chunk_index"],
            "source_type": c.get("source_type", "unknown"),
            "section_heading": c.get("section_heading", ""),
            "article_marker": c.get("article_marker", ""),
        }
        for c in new_chunks
    ]
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    return {
        "files_processed": files_processed,
        "chunks_added": len(new_chunks),
        "chunks_skipped": chunks_skipped,
        "errors": errors,
    }


if __name__ == "__main__":
    result = run_ingest()
    print("Ingest result:", result)
