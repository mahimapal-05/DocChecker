"""
Singleton wrapper around ChromaDB + Ollama embeddings.
Provides semantic search and document storage.
"""

from __future__ import annotations

import os
from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
_store: Optional[VectorStore] = None


class VectorStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        os.makedirs(self.settings.chroma_path, exist_ok=True)

        self.embeddings = OllamaEmbeddings(
            base_url=self.settings.ollama_base_url,
            model=self.settings.embed_model,
        )

        self._client = chromadb.PersistentClient(
            path=self.settings.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._vectorstore = Chroma(
            client=self._client,
            collection_name=self.settings.chroma_collection,
            embedding_function=self.embeddings,
        )
        logger.info(
            "vector_store_ready",
            collection=self.settings.chroma_collection,
            path=self.settings.chroma_path,
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents and return their IDs."""
        ids = self._vectorstore.add_documents(documents)
        logger.info("documents_added", count=len(documents))
        return ids

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        filter: dict | None = None,
    ) -> list[Document]:
        k = k or self.settings.top_k_results
        return self._vectorstore.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(
        self,
        query: str,
        k: int | None = None,
        filter: dict | None = None,
    ) -> list[tuple[Document, float]]:
        k = k or self.settings.retrieval_expansion_k
        return self._vectorstore.similarity_search_with_relevance_scores(
            query, k=k, filter=filter
        )

    def get_collection_stats(self) -> dict:
        col = self._client.get_collection(self.settings.chroma_collection)
        return {"document_count": col.count(), "collection": self.settings.chroma_collection}

    def delete_collection(self) -> None:
        self._client.delete_collection(self.settings.chroma_collection)
        logger.warning("collection_deleted", collection=self.settings.chroma_collection)
        # Re-init
        self._vectorstore = Chroma(
            client=self._client,
            collection_name=self.settings.chroma_collection,
            embedding_function=self.embeddings,
        )


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
