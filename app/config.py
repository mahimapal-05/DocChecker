from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    embed_model: str = "nomic-embed-text"

    # ChromaDB
    chroma_path: str = "./data/chroma_db"
    chroma_collection: str = "documents"

    # Agent behaviour
    confidence_threshold: float = 0.7
    max_retrieval_loops: int = 3
    top_k_results: int = 6
    retrieval_expansion_k: int = 10

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # Uploads
    max_upload_size_mb: int = 50
    upload_dir: str = "./data/uploads"

    # GCP (optional)
    gcp_project_id: str = ""
    gcp_region: str = "us-central1"
    cloud_run_service: str = "doc-intel-api"


@lru_cache
def get_settings() -> Settings:
    return Settings()
