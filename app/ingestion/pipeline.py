"""
Document Ingestion Pipeline
────────────────────────────
Supports: PDF, CSV, HTML, TXT, MD

Steps:
  1. Load raw file with the appropriate LangChain loader
  2. Clean and normalise text
  3. Split into chunks with overlap
  4. Embed + store in ChromaDB
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    BSHTMLLoader,
    TextLoader,
)

from app.core.vector_store import get_vector_store
from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

SupportedType = Literal["pdf", "csv", "html", "txt", "md"]

SUPPORTED_EXTENSIONS: dict[str, SupportedType] = {
    ".pdf": "pdf",
    ".csv": "csv",
    ".html": "html",
    ".htm": "html",
    ".txt": "txt",
    ".md": "txt",
}


def _detect_type(path: Path) -> SupportedType:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")
    return SUPPORTED_EXTENSIONS[ext]


def _load_documents(path: Path, doc_type: SupportedType) -> list[Document]:
    file_str = str(path)

    if doc_type == "pdf":
        loader = PyPDFLoader(file_str)
    elif doc_type == "csv":
        loader = CSVLoader(file_str, encoding="utf-8")
    elif doc_type == "html":
        loader = BSHTMLLoader(file_str, open_encoding="utf-8")
    else:  # txt / md
        loader = TextLoader(file_str, encoding="utf-8")

    docs = loader.load()
    logger.info("loader_done", file=path.name, doc_type=doc_type, pages=len(docs))
    return docs


def _clean_text(text: str) -> str:
    """Remove excessive whitespace and control characters."""
    text = re.sub(r"\x00", "", text)                  # null bytes
    text = re.sub(r"[ \t]+", " ", text)               # collapse spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)            # max 2 newlines
    return text.strip()


def _chunk_documents(
    documents: list[Document],
    source_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[Document] = []
    for doc in documents:
        cleaned = _clean_text(doc.page_content)
        if not cleaned:
            continue

        sub_chunks = splitter.create_documents(
            texts=[cleaned],
            metadatas=[{
                **doc.metadata,
                "source": source_name,
                "chunk_size": chunk_size,
            }],
        )
        chunks.extend(sub_chunks)

    logger.info("chunking_done", source=source_name, num_chunks=len(chunks))
    return chunks


def ingest_file(file_path: str | Path) -> dict:
    """
    Full ingestion pipeline for a single file.
    Returns a summary dict.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    settings = get_settings()
    doc_type = _detect_type(path)

    logger.info("ingestion_start", file=path.name, type=doc_type)

    # 1. Load
    raw_docs = _load_documents(path, doc_type)

    # 2. Chunk
    chunks = _chunk_documents(
        raw_docs,
        source_name=path.name,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if not chunks:
        raise ValueError(f"No text could be extracted from {path.name}")

    # 3. Embed + store
    store = get_vector_store()
    ids = store.add_documents(chunks)

    result = {
        "file": path.name,
        "type": doc_type,
        "pages_loaded": len(raw_docs),
        "chunks_stored": len(chunks),
        "chunk_ids": ids[:5],  # return sample only
    }
    logger.info("ingestion_complete", **result)
    return result


def get_store_stats() -> dict:
    store = get_vector_store()
    return store.get_collection_stats()
