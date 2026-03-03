# Legal RAG Prototype

A small local RAG (Retrieval-Augmented Generation) prototype for legal documents. It uses **FastAPI**, **ChromaDB**, and **LM Studio** (OpenAI-compatible API) to ingest documents, embed chunks, and answer questions only from retrieved context—with clear prompts to avoid inventing legal conclusions.

## Requirements

- Python 3.11+
- [LM Studio](https://lmstudio.ai/) running locally with a chat model and an embedding model loaded

## Install dependencies

```bash
pip install -r requirements.txt
```

Copy the example env file and set your model names (as shown in LM Studio):

```bash
copy .env.example .env
```

Edit `.env` and set:

- `CHAT_MODEL` – name of the chat model you load in LM Studio (e.g. `llama-3.2-3b` or whatever appears in the UI).
- `EMBEDDING_MODEL` – name of the embedding model you load in LM Studio (e.g. `nomic-embed-text` or similar).

## Start LM Studio

1. Open LM Studio.
2. Download (or use) a **chat model** (e.g. Llama, Mistral) and an **embedding model** (e.g. `nomic-embed-text`, `all-MiniLM-L6-v2` if available).
3. Start the **Local Server** (e.g. port **1234**).
4. In the server view, **load** both the chat model and the embedding model so both are available under the same server.

The app expects the API at `http://localhost:1234/v1`. If you use another port, set `LM_STUDIO_BASE_URL` in `.env`.

## Add legal documents

Place your documents in:

```text
./data/legal_docs/
```

You can use **nested folders** (e.g. `laws/`, `templates/`, `regulations/`, `firm_notes/`); ingestion scans recursively. Supported formats: **.txt**, **.md**, **.pdf**. Extraction preserves line breaks so structure-aware chunking can detect headings and articles.

## Run ingestion

From the project root:

```bash
python ingest.py
```

This will:

- **Recursively** scan all files under `./data/legal_docs/` (including subfolders)
- Extract text (line breaks preserved for structure detection)
- Chunk using **structure-aware splitting** (Article/Section/markdown headings) first, then fallback character chunking (~700–1000 characters with overlap)
- Generate embeddings via LM Studio’s embeddings endpoint
- Store chunks in ChromaDB under `./chroma_db` (persistent)
- Skip duplicates by **chunk id** (`filename|page|text`), so the same text in different files keeps separate provenance

Chunk metadata includes `source_type`, `section_heading`, and `article_marker` when detected. You can run `python ingest.py` again after adding or changing files; new chunks are added, duplicates by id are skipped.

## Start the FastAPI server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- API: `http://localhost:8000`
- Simple UI: `http://localhost:8000/` (type a question, see answer and sources)

## API endpoints

| Method | Path      | Description |
|--------|-----------|-------------|
| GET    | `/health` | Health check |
| POST   | `/ingest` | Run ingestion (scan, chunk, embed, store) |
| POST   | `/ask`    | Ask a question; returns answer, sources, confidence note |

## Test `/ask` with curl

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"What does the contract say about termination?\"}"
```

Response includes `answer`, `sources` (each with `text`, `filename`, `page`, `chunk_index`, `distance`, and optionally `source_type`, `section_heading`, `article_marker`), and `confidence_note`.

## Troubleshooting

- **“Cannot reach LM Studio”**  
  - Ensure LM Studio is running and the local server is started (e.g. port 1234).  
  - Check `LM_STUDIO_BASE_URL` in `.env` (e.g. `http://localhost:1234/v1`).  
  - In LM Studio, ensure both the **chat** and **embedding** models are loaded in the server tab.

- **Empty or no relevant answers**  
  - Run ingestion after adding documents: `python ingest.py`.  
  - Confirm files are in `./data/legal_docs/` (or in nested folders under it) with extensions `.txt`, `.md`, or `.pdf`.  
  - Check that ChromaDB has data (e.g. no errors during `ingest.py`).  
  - Try a question that clearly relates to the content of your documents.

- **Wrong or missing model**  
  - In LM Studio, the model names must match `CHAT_MODEL` and `EMBEDDING_MODEL` in `.env`.  
  - Names are case-sensitive and must match exactly what LM Studio shows when the model is loaded.

- **ChromaDB errors**  
  - If the DB is corrupted or you change embedding model, delete the folder `./chroma_db` and run `python ingest.py` again to rebuild.

- **PDF not extracting text**  
  - Some PDFs are image-only (scans). Use OCR externally and add the result as `.txt` or `.md` in `./data/legal_docs/`.

## Design notes

- **Local only:** No auth, no cloud; suitable for testing on a laptop.
- **Verifiability:** Answers are tied to retrieved chunks; the UI and API show which sources were used.
- **Legal safety:** The system prompt instructs the model to answer only from retrieved context and to state when it lacks sufficient verified legal context or when lawyer review is needed. It does not replace a licensed lawyer.

## Output format

Every answer is structured as:

1. **Answer**  
2. **Why I answered this** (which retrieved context supports the answer)  
3. **Source excerpts used** (short supporting chunks with filename/page)  
4. **What is missing / what needs lawyer review**

If context is insufficient, the model is instructed to respond: *“I do not have enough verified legal context to answer this safely.”*
