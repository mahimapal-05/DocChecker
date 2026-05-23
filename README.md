# Multi-Agent Document Intelligence System

A production-grade RAG pipeline using LangGraph with 4 specialized agents, FastAPI, ChromaDB, and Ollama (local LLMs).

## Architecture

```
User Query → Orchestrator Agent
                 ↓
           Retriever Agent  ←──── ChromaDB Vector Store
                 ↓
           Analyst Agent   (synthesizes answer)
                 ↓
           Critic Agent    (validates quality & confidence)
                 ↓
         [confidence < threshold?] → loop back to Retriever
                 ↓
           Final Response
```

### Agents
| Agent | Role |
|-------|------|
| **Orchestrator** | Parses query intent, routes to retriever, manages state |
| **Retriever** | Hybrid search (semantic + keyword), re-ranks results |
| **Analyst** | Synthesizes retrieved chunks into a coherent answer |
| **Critic** | Scores answer quality/confidence; triggers retry loop if needed |

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- [Ollama](https://ollama.ai) installed and running

### Pull required models
```bash
ollama pull llama3.1:8b          # Main agent model
ollama pull nomic-embed-text     # Embeddings
```

## Quick Start

### Option A: Docker Compose (recommended)
```bash
cp .env.example .env
docker compose up --build
```

### Option B: Local dev
```bash
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

## API Usage

### Ingest documents
```bash
# Single file
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/document.pdf"

# Check ingestion status
curl http://localhost:8000/ingest/status
```

### Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main risks identified in the Q3 report?"}'
```

### Stream query (SSE)
```bash
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the key findings"}'
```

### Health check
```bash
curl http://localhost:8000/health
```

## Configuration

Edit `.env` to configure:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
EMBED_MODEL=nomic-embed-text
CHROMA_PATH=./data/chroma_db
CONFIDENCE_THRESHOLD=0.7        # Below this, Critic triggers retry
MAX_RETRIEVAL_LOOPS=3           # Max retry iterations
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K_RESULTS=6
```

## Supported Document Types
- PDF (`.pdf`)
- CSV (`.csv`)
- HTML (`.html`, `.htm`)
- Plain text (`.txt`)
- Markdown (`.md`)

## Running Tests
```bash
pytest tests/ -v
```

## GCP Cloud Run Deployment
See `docker/cloudbuild.yaml` and `scripts/deploy_gcp.sh`.

Requirements:
- GCP project with Cloud Run and Artifact Registry enabled
- `gcloud` CLI authenticated
- Update `GCP_PROJECT_ID` in `.env`
