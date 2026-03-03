"""
Configuration for the legal RAG prototype.
Loads settings from environment variables (see .env.example).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# LM Studio API
LM_STUDIO_BASE_URL: str = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_API_KEY: str = os.getenv("LM_STUDIO_API_KEY", "lm-studio")

# Model names (must match models loaded in LM Studio)
CHAT_MODEL: str = os.getenv("CHAT_MODEL", "your-chat-model-name")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "your-embedding-model-name")

# ChromaDB persistent storage path
CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./chroma_db")

# Data directory for legal documents
DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data/legal_docs"))

# Chunking defaults (characters)
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

# Retrieval
TOP_K: int = int(os.getenv("TOP_K", "5"))

# Supported file extensions for ingestion
SUPPORTED_EXTENSIONS: set[str] = {".txt", ".md", ".pdf"}
