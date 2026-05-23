"""
FastAPI Application Entry Point
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.core.logging import get_logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    logger = get_logger("startup")

    # Ensure data directories exist
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.chroma_path, exist_ok=True)

    # Warm up vector store connection
    from app.core.vector_store import get_vector_store
    get_vector_store()

    # Warm up agent graph
    from app.agents.pipeline import get_graph
    get_graph()

    logger.info(
        "app_started",
        model=settings.ollama_model,
        embed_model=settings.embed_model,
        chroma_path=settings.chroma_path,
    )
    yield
    logger.info("app_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Multi-Agent Document Intelligence API",
        description=(
            "RAG pipeline with 4 specialized LangGraph agents: "
            "Orchestrator, Retriever, Analyst, and Critic."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="")
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
