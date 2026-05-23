#!/usr/bin/env python3
"""
CLI tool for the Multi-Agent Document Intelligence System.

Usage:
  python scripts/cli.py ingest path/to/document.pdf
  python scripts/cli.py query "What are the key findings?"
  python scripts/cli.py stats
  python scripts/cli.py health
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def cmd_ingest(args):
    from app.ingestion.pipeline import ingest_file

    print(f"Ingesting: {args.file}")
    try:
        result = ingest_file(args.file)
        print(f"\n✓ Ingestion complete")
        print(f"  File:           {result['file']}")
        print(f"  Type:           {result['type']}")
        print(f"  Pages loaded:   {result['pages_loaded']}")
        print(f"  Chunks stored:  {result['chunks_stored']}")
    except Exception as e:
        print(f"\n✗ Ingestion failed: {e}")
        sys.exit(1)


def cmd_query(args):
    from app.agents.pipeline import run_query

    print(f"Query: {args.query}\n")
    print("Running agent pipeline...\n")

    state = asyncio.run(run_query(args.query))

    answer = state.get("final_answer") or state.get("draft_answer") or "No answer generated."
    confidence = state.get("confidence_score", 0.0)
    sources = state.get("sources_used", [])
    loops = state.get("retrieval_loop_count", 0)
    critique = state.get("critique", "")

    print("─" * 60)
    print("ANSWER")
    print("─" * 60)
    print(answer)
    print()
    print(f"Confidence: {confidence:.0%}  |  Loops: {loops}  |  Sources: {len(sources)}")
    if sources:
        print(f"Sources: {', '.join(sources)}")
    if critique:
        print(f"Critic: {critique}")

    if args.trace:
        print("\nAgent trace:")
        for step in state.get("agent_trace", []):
            print(f"  {step}")


def cmd_stats(args):
    from app.ingestion.pipeline import get_store_stats

    stats = get_store_stats()
    print(f"Collection:     {stats['collection']}")
    print(f"Document chunks: {stats['document_count']}")


def cmd_health(args):
    import httpx
    from app.config import get_settings

    settings = get_settings()
    print(f"Ollama URL:  {settings.ollama_base_url}")
    print(f"Model:       {settings.ollama_model}")
    print(f"Embed model: {settings.embed_model}")

    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        print(f"✓ Ollama reachable. Available models: {models}")
    except Exception as e:
        print(f"✗ Ollama unreachable: {e}")

    try:
        stats = __import__("app.ingestion.pipeline", fromlist=["get_store_stats"]).get_store_stats()
        print(f"✓ ChromaDB OK. {stats['document_count']} chunks stored.")
    except Exception as e:
        print(f"✗ ChromaDB error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Document Intelligence CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest a document file")
    p_ingest.add_argument("file", help="Path to file (pdf, csv, html, txt, md)")
    p_ingest.set_defaults(func=cmd_ingest)

    # query
    p_query = sub.add_parser("query", help="Run a query through the agent pipeline")
    p_query.add_argument("query", help="Question to answer")
    p_query.add_argument("--trace", action="store_true", help="Show agent execution trace")
    p_query.set_defaults(func=cmd_query)

    # stats
    p_stats = sub.add_parser("stats", help="Show document store statistics")
    p_stats.set_defaults(func=cmd_stats)

    # health
    p_health = sub.add_parser("health", help="Check system health")
    p_health.set_defaults(func=cmd_health)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
