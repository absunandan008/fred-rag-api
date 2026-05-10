# fred-rag-api

A Retrieval-Augmented Generation (RAG) API built on top of FRED (Federal Reserve Economic Data), designed to answer natural language questions about U.S. macroeconomic data.

## What This Is

This project is part of a structured learning path in applied ML engineering — progressing from foundational LLM concepts (character-level language models, tokenizers) through vector databases, RAG pipelines, and production-ready API design on AWS.

The goal is not just to build a working system, but to understand every layer of it.

## Architecture

```
User Query
    │
    ▼
FastAPI (REST API)
    │
    ├── Embedder (sentence-transformers / Bedrock Titan)
    │       │
    │       ▼
    ├── ChromaDB (vector store — Docker volume)
    │       │
    │       ▼ retrieved context
    │
    └── LLM
            ├── Ollama / llama3.2  (local dev default)
            └── AWS Bedrock / Nova Micro  (production swap)
```

**Key design principle:** the LLM provider is swappable via a single environment variable. Local development runs on Ollama with no cloud costs. Production targets AWS Bedrock (Amazon Nova Micro).

## Tech Stack

| Layer | Tool |
|---|---|
| API | FastAPI |
| Vector Store | ChromaDB (Docker volume) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (local) |
| LLM — dev | Ollama (`llama3.2`) |
| LLM — prod | AWS Bedrock (`us.amazon.nova-micro-v1:0`) |
| Data Source | FRED API (Federal Reserve Economic Data) |
| Infra | Docker + Docker Compose |
| Package Manager | uv |

## Project Structure

```
fred-rag-api/
├── app/
│   ├── main.py           # FastAPI entrypoint
│   ├── retriever.py      # ChromaDB query logic
│   ├── embedder.py       # Embedding abstraction
│   ├── llm.py            # LLM abstraction (Ollama / Bedrock)
│   └── ingest.py         # FRED data ingestion pipeline
├── vectorstore/          # Mounted ChromaDB volume
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── pyproject.toml        # uv project config & dependencies
└── uv.lock
```

## Getting Started

### Prerequisites

- Docker + Docker Compose
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- Ollama running locally (`ollama pull llama3.2`)
- FRED API key — free at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)

### Run locally

```bash
# Clone
git clone https://github.com/absunandan008/fred-rag-api.git
cd fred-rag-api

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env — set FRED_API_KEY and confirm LLM_PROVIDER=ollama

# Start
docker compose up --build

# Ingest FRED data
docker compose exec api uv run python app/ingest.py

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What has the unemployment rate been doing over the last year?"}'
```

### Switch to AWS Bedrock

```bash
# In .env
LLM_PROVIDER=bedrock
BEDROCK_MODEL_ID=us.amazon.nova-micro-v1:0
AWS_REGION=us-east-1
```

No code changes needed.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` or `bedrock` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `BEDROCK_MODEL_ID` | `us.amazon.nova-micro-v1:0` | Bedrock model ID |
| `AWS_REGION` | `us-east-1` | AWS region |
| `CHROMA_PATH` | `/app/vectorstore` | ChromaDB persistence path |
| `FRED_API_KEY` | — | Required for data ingestion |

## Learning Context

This project sits at step 4 of a deliberate ML engineering curriculum:

1. ✅ Makemore → GPT (Karpathy series)
2. ✅ Vector DB foundations
3. ✅ RAG from scratch (ChromaDB + sentence-transformers + Ollama + FastAPI)
4. 🔨 **This project** — production-shaped RAG with real data, evals, and cloud LLM swap
5. 🔜 Tokenizer deep dive
6. 🔜 Fine-tuning

The emphasis is on understanding the internals — not just wiring together libraries.

## Evaluation

The API includes an eval suite measuring:

- **Faithfulness** — does the answer stay grounded in retrieved context?
- **Relevance** — does the retrieved context match the query?
- **Groundedness** — are claims traceable to source documents?

---

*Data provided by [FRED, Federal Reserve Bank of St. Louis](https://fred.stlouisfed.org).*
