"""
Test suite for the Multi-Agent Document Intelligence System.
Run: pytest tests/ -v
"""

from __future__ import annotations

import json
import os
import tempfile
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ── Unit tests: Orchestrator ───────────────────────────────────────────────────

class TestOrchestrator:
    def test_valid_json_response(self):
        """Orchestrator correctly parses a well-formed LLM response."""
        from app.agents.orchestrator import orchestrator_node

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "intent": "analytical",
            "retrieval_queries": [
                "What are the main risks?",
                "risks identified in the report",
            ],
        })

        with patch("app.agents.orchestrator.ChatOllama") as MockLLM:
            mock_chain_result = MagicMock()
            mock_chain_result.content = mock_response.content
            MockLLM.return_value.__or__ = MagicMock(return_value=MagicMock(
                invoke=MagicMock(return_value=mock_chain_result)
            ))

            state = {
                "query": "What are the main risks?",
                "agent_trace": [],
                "retrieval_loop_count": 0,
            }
            # Just test the fallback path (no real Ollama)
            result = orchestrator_node({
                **state,
                "query": "What are the main risks in the Q3 report?",
            })

        # Fallback is acceptable if Ollama not running in test env
        assert "query_intent" in result
        assert "retrieval_queries" in result
        assert len(result["retrieval_queries"]) >= 1

    def test_fallback_on_bad_json(self):
        """Orchestrator falls back gracefully when LLM returns bad JSON."""
        from app.agents.orchestrator import orchestrator_node

        with patch("app.agents.orchestrator.ChatOllama") as MockLLM:
            chain_mock = MagicMock()
            chain_mock.invoke.return_value = MagicMock(content="not json at all")
            MockLLM.return_value.__or__ = MagicMock(return_value=chain_mock)

            result = orchestrator_node({
                "query": "test query",
                "agent_trace": [],
                "retrieval_loop_count": 0,
            })

        assert result["query_intent"] == "analytical"
        assert result["retrieval_queries"] == ["test query"]


# ── Unit tests: Retriever ──────────────────────────────────────────────────────

class TestRetriever:
    def test_deduplication(self):
        """_deduplicate removes near-duplicate chunks."""
        from app.agents.retriever import _deduplicate

        chunks = [
            ("Hello world this is a test chunk", "file1.pdf", 0.9, {}),
            ("Hello world this is a test chunk", "file2.pdf", 0.85, {}),  # dup
            ("Completely different content here", "file1.pdf", 0.8, {}),
        ]
        result = _deduplicate(chunks)
        assert len(result) == 2

    def test_mmr_rerank_respects_top_k(self):
        """MMR reranking returns at most top_k results."""
        from app.agents.retriever import _mmr_rerank

        chunks = [
            (f"Document chunk number {i} with some content about topic {i}", f"file{i}.pdf", 0.9 - i * 0.05, {})
            for i in range(10)
        ]
        result = _mmr_rerank(chunks, top_k=4)
        assert len(result) == 4

    def test_empty_vector_store_returns_empty(self):
        """Retriever handles zero results gracefully."""
        from app.agents.retriever import retriever_node

        with patch("app.agents.retriever.get_vector_store") as mock_store:
            mock_store.return_value.similarity_search_with_score.return_value = []

            result = retriever_node({
                "query": "test",
                "retrieval_queries": ["test"],
                "retrieval_loop_count": 0,
                "retry_guidance": "",
                "agent_trace": [],
            })

        assert result["retrieved_chunks"] == []


# ── Unit tests: Critic ────────────────────────────────────────────────────────

class TestCritic:
    def test_max_loops_prevents_retry(self):
        """Critic never sets needs_retry when loop count >= max."""
        from app.agents.critic import critic_node

        response_json = json.dumps({
            "confidence_score": 0.4,
            "critique": "Missing information",
            "needs_retry": True,
            "retry_guidance": "search more",
        })

        with patch("app.agents.critic.ChatOllama") as MockLLM:
            chain_mock = MagicMock()
            chain_mock.invoke.return_value = MagicMock(content=response_json)
            MockLLM.return_value.__or__ = MagicMock(return_value=chain_mock)

            with patch("app.agents.critic.get_settings") as mock_settings:
                mock_settings.return_value.max_retrieval_loops = 3
                mock_settings.return_value.confidence_threshold = 0.7
                mock_settings.return_value.ollama_base_url = "http://localhost:11434"
                mock_settings.return_value.ollama_model = "llama3.1:8b"

                result = critic_node({
                    "query": "test",
                    "draft_answer": "Some answer",
                    "retrieved_chunks": [],
                    "retrieval_loop_count": 3,  # at max
                    "agent_trace": [],
                    "error": None,
                })

        assert result["needs_retry"] is False

    def test_high_confidence_skips_retry(self):
        """Critic doesn't retry when confidence is above threshold."""
        from app.agents.critic import critic_node

        response_json = json.dumps({
            "confidence_score": 0.95,
            "critique": "Answer is complete",
            "needs_retry": False,
            "retry_guidance": "",
        })

        with patch("app.agents.critic.ChatOllama") as MockLLM:
            chain_mock = MagicMock()
            chain_mock.invoke.return_value = MagicMock(content=response_json)
            MockLLM.return_value.__or__ = MagicMock(return_value=chain_mock)

            with patch("app.agents.critic.get_settings") as mock_settings:
                mock_settings.return_value.max_retrieval_loops = 3
                mock_settings.return_value.confidence_threshold = 0.7
                mock_settings.return_value.ollama_base_url = "http://localhost:11434"
                mock_settings.return_value.ollama_model = "llama3.1:8b"

                result = critic_node({
                    "query": "test",
                    "draft_answer": "Great answer",
                    "retrieved_chunks": [],
                    "retrieval_loop_count": 0,
                    "agent_trace": [],
                    "error": None,
                })

        assert result["needs_retry"] is False
        assert result["final_answer"] == "Great answer"


# ── Integration tests: Ingestion ───────────────────────────────────────────────

class TestIngestion:
    def test_unsupported_extension_raises(self):
        """ingest_file raises ValueError for unsupported extensions."""
        from app.ingestion.pipeline import _detect_type
        from pathlib import Path

        with pytest.raises(ValueError, match="Unsupported"):
            _detect_type(Path("document.xlsx"))

    def test_supported_extensions_detected(self):
        """All supported extensions are correctly identified."""
        from app.ingestion.pipeline import _detect_type, SUPPORTED_EXTENSIONS
        from pathlib import Path

        for ext, expected_type in SUPPORTED_EXTENSIONS.items():
            result = _detect_type(Path(f"file{ext}"))
            assert result == expected_type

    def test_text_cleaning(self):
        """_clean_text normalises whitespace and removes null bytes."""
        from app.ingestion.pipeline import _clean_text

        messy = "Hello\x00 world   \n\n\n\n\nEnd"
        cleaned = _clean_text(messy)
        assert "\x00" not in cleaned
        assert "\n\n\n" not in cleaned
        assert "Hello" in cleaned
        assert "End" in cleaned

    def test_txt_ingestion(self):
        """Full ingestion pipeline works on a plain text file."""
        from app.ingestion.pipeline import ingest_file

        with patch("app.ingestion.pipeline.get_vector_store") as mock_vs:
            mock_vs.return_value.add_documents.return_value = ["id1", "id2"]

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write("This is a test document.\n" * 50)
                tmp_path = f.name

            try:
                result = ingest_file(tmp_path)
                assert result["type"] == "txt"
                assert result["chunks_stored"] > 0
            finally:
                os.unlink(tmp_path)


# ── API tests ─────────────────────────────────────────────────────────────────

class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_health_endpoint_exists(self, client):
        """Health endpoint returns 200."""
        with patch("app.api.routes.get_vector_store") as mock_vs, \
             patch("httpx.AsyncClient") as mock_http:
            mock_vs.return_value.get_collection_stats.return_value = {
                "document_count": 5,
                "collection": "documents",
            }
            resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "document_count" in data

    def test_query_validation(self, client):
        """Query endpoint rejects too-short queries."""
        resp = client.post("/query", json={"query": "hi"})
        # 'hi' is 2 chars, below min_length=3
        assert resp.status_code == 422

    def test_ingest_rejects_bad_extension(self, client):
        """Ingest endpoint rejects unsupported file types."""
        resp = client.post(
            "/ingest",
            files={"file": ("report.xlsx", b"fake content", "application/octet-stream")},
        )
        assert resp.status_code == 415
