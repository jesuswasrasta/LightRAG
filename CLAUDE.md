# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**LightRAG** is a production-ready Retrieval-Augmented Generation (RAG) system that uses knowledge graphs for enhanced document retrieval and question answering. The system extracts entities and relationships from documents using LLMs, stores them in a knowledge graph, and uses this graph alongside vector embeddings for intelligent retrieval.

**Key Capabilities:**
- Multi-modal document processing (text, images, tables via RAG-Anything integration)
- Graph-based RAG with entity-relationship extraction
- Multiple query modes: local, global, hybrid, naive, mix, bypass
- Web UI and REST API with Ollama-compatible endpoints
- Multiple LLM providers: OpenAI, Ollama, Gemini, Anthropic, Azure, Bedrock, and more
- Multiple storage backends: In-memory (default), PostgreSQL, Neo4j, MongoDB, Redis, Milvus, Qdrant
- RAGAS evaluation and Langfuse tracing support

## Development Setup

### Python Environment

This project uses **uv** for fast Python package management (recommended) or pip as an alternative.

```bash
# Install uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh  # Unix/macOS
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install LightRAG Core (basic functionality)
uv sync  # Creates .venv/ automatically
source .venv/bin/activate  # Linux/macOS
# Windows: .venv\Scripts\activate

# Install with API/WebUI support
uv sync --extra api

# Install with all offline dependencies (storage + LLM providers)
uv sync --extra offline

# Or using pip
pip install -e .  # Core only
pip install -e ".[api]"  # With API support
```

**Python Requirements:** Python 3.10+

### Frontend Development

**CRITICAL: Always use Bun for frontend operations** (not npm or yarn)

```bash
cd lightrag_webui

# Install dependencies
bun install --frozen-lockfile

# Development server
bun run dev

# Production build (required before running lightrag-server)
bun run build

# Other commands
bun run lint      # Linting
bun test          # Run tests
bun run preview   # Preview production build
```

The frontend is a React + TypeScript + Vite application using Tailwind CSS and Sigma.js for graph visualization.

## Running LightRAG

### LightRAG Server (Web UI + API)

The server provides a web interface for document management, knowledge graph visualization, and query interface.

```bash
# Setup environment
cp env.example .env
# Edit .env with your LLM and embedding configurations

# Build frontend first
cd lightrag_webui && bun install --frozen-lockfile && bun run build && cd ..

# Run server
lightrag-server  # Uvicorn development server
# or
lightrag-gunicorn  # Production with Gunicorn
```

Default: http://localhost:9621

### Docker Deployment

```bash
cp env.example .env
# Edit .env configuration
docker compose up
```

### LightRAG Core (Programmatic Usage)

```python
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

async def main():
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
    )
    # REQUIRED: Initialize storage before use
    await rag.initialize_storages()

    # Insert documents
    await rag.ainsert("Your document text here")

    # Query
    result = await rag.aquery(
        "Your question",
        param=QueryParam(mode="hybrid")
    )
    print(result)

    # REQUIRED: Finalize storage on shutdown
    await rag.finalize_storages()

asyncio.run(main())
```

## Testing

The project uses pytest with markers for CI/CD optimization.

```bash
# Run offline tests only (fast, ~3 seconds, runs in CI)
pytest tests/ -m offline -v

# Run all tests including integration tests (requires external services)
pytest tests/ --run-integration -v

# Run specific test file
pytest tests/test_chunking.py -v

# Run with coverage
pytest tests/ -m offline --cov=lightrag --cov-report=html
```

**Test Categories:**
- `@pytest.mark.offline` - No external dependencies, uses mocks (CI default)
- `@pytest.mark.integration` - Requires databases/APIs (skipped by default)

**Configuration:**
- `pyproject.toml` - pytest settings and markers
- `tests/conftest.py` - fixtures and custom options

## Architecture

### Core Components

**LightRAG Core (`lightrag/lightrag.py`):**
- Main entry point: `LightRAG` class
- Orchestrates document processing pipeline and query execution
- Manages storage backends (KV, Vector, Graph, DocStatus)
- **CRITICAL**: Requires explicit `await rag.initialize_storages()` and `await rag.finalize_storages()`

**Document Processing (`lightrag/operate.py`):**
- `chunking_by_token_size()` - Splits documents into chunks
- `extract_entities()` - LLM-based entity and relationship extraction
- `merge_nodes_and_edges()` - Merges and summarizes duplicate entities/relations
- `rebuild_knowledge_from_chunks()` - Reconstructs knowledge graph

**Query Processing (`lightrag/operate.py`):**
- `kg_query()` - Knowledge graph-based retrieval (local/global/hybrid modes)
- `naive_query()` - Simple vector similarity search
- Supports multiple retrieval modes and optional reranking

**Storage Layer (`lightrag/kg/`):**
- Abstraction: `BaseKVStorage`, `BaseVectorStorage`, `BaseGraphStorage`, `DocStatusStorage`
- Implementations:
  - In-memory: `JsonKVStorage`, `NetworkXStorage`, `NanoVectorDBStorage`
  - PostgreSQL: `PGKVStorage`, `PGVectorStorage`, `PGGraphStorage`, `PGDocStatusStorage`
  - Neo4j: `Neo4JStorage`
  - MongoDB: `MongoKVStorage`, `MongoVectorDBStorage`, `MongoGraphStorage`
  - Redis: `RedisKVStorage`, `RedisDocStatusStorage`
  - Vector-specific: `MilvusVectorDBStorage`, `QdrantVectorDBStorage`, `FaissVectorDBStorage`

**LLM Integration (`lightrag/llm/`):**
- Provider implementations with consistent interface
- Supports: OpenAI, Azure OpenAI, Ollama, Gemini, Anthropic, Bedrock, and OpenAI-compatible APIs
- Embedding and completion functions follow `EmbeddingFunc` and LLM callable protocols
- **CRITICAL Pattern**: Always handle both base64 and raw array embedding formats from custom endpoints

**API Server (`lightrag/api/`):**
- FastAPI-based REST API with JWT authentication
- Ollama-compatible endpoints for easy integration with AI chat bots
- Routers: `document_routes.py`, `query_routes.py`, `graph_routes.py`, `ollama_api.py`
- Gunicorn support for production deployment

**Web UI (`lightrag_webui/`):**
- React + TypeScript frontend
- Features: document upload, knowledge graph visualization (Sigma.js), query interface
- Internationalization: i18next for multi-language support
- Build tool: Vite with Bun runtime

### Data Flow

**Document Indexing:**
1. Document uploaded → chunked by token size
2. Each chunk → LLM extracts entities and relationships
3. Entities/relations → deduplicated and merged with LLM summaries
4. Entities/relations → embedded and stored in vector DB
5. Graph structure → stored in graph DB
6. Original chunks → stored in KV storage

**Query Execution:**
1. User question → embedded
2. Retrieve relevant entities/relations from vector DB
3. For KG modes: expand via graph traversal (local/global/hybrid strategies)
4. Retrieve associated document chunks
5. Optional: rerank retrieved context
6. Send context + question → LLM for final answer

### Storage Backends

**Default (In-Memory with File Persistence):**
- `JsonKVStorage` - Key-value storage (documents, chunks, LLM cache)
- `NetworkXStorage` - Graph structure (entities, relationships)
- `NanoVectorDBStorage` - Vector embeddings
- `JsonDocStatusStorage` - Document processing status

**Production Recommendations:**
- PostgreSQL: All-in-one solution (`PGKVStorage`, `PGVectorStorage`, `PGGraphStorage`)
- Neo4j: Advanced graph operations (`Neo4JStorage`)
- Milvus/Qdrant: High-performance vector search
- Redis: Fast KV and doc status storage

## Configuration

All configuration via `.env` file (see `env.example` for complete reference).

**Critical Settings:**

```bash
# LLM (required)
LLM_BINDING=openai  # openai, ollama, gemini, azure_openai, anthropic, aws_bedrock
LLM_MODEL=gpt-4o
LLM_BINDING_API_KEY=your_key
LLM_BINDING_HOST=https://api.openai.com/v1

# Embedding (required, must be set before first document)
EMBEDDING_BINDING=openai  # openai, ollama, gemini, jina
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIM=3072
EMBEDDING_BINDING_API_KEY=your_key

# Workspace (data isolation)
WORKSPACE=space1  # Isolate data between LightRAG instances

# Storage backends
LIGHTRAG_KV_STORAGE=JsonKVStorage
LIGHTRAG_GRAPH_STORAGE=NetworkXStorage
LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage
LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage

# Query behavior
TOP_K=40  # Number of entities/relations retrieved from KG
MAX_TOTAL_TOKENS=30000  # Max tokens sent to LLM
ENABLE_LLM_CACHE=true

# Reranking (optional but recommended)
RERANK_BINDING=null  # cohere, jina, aliyun
RERANK_MODEL=BAAI/bge-reranker-v2-m3
```

**IMPORTANT**: Embedding model cannot be changed after first document is processed (vector dimensions must match).

## Critical Implementation Patterns

### 1. Async/Await Pattern
Always await coroutines before accessing their results:
```python
# WRONG: calling methods on coroutine
result = await storage._data.list_indexes().to_list()

# CORRECT: await first, then call methods
cursor = await storage._data.list_indexes()
result = await cursor.to_list()
```

### 2. Embedding Format Compatibility
Custom OpenAI-compatible endpoints may return raw arrays instead of base64:
```python
# Handle both formats
np.array(dp.embedding) if isinstance(dp.embedding, list) \
    else np.frombuffer(base64.b64decode(dp.embedding), dtype=np.float32)
```

### 3. Storage Data Compatibility
Filter deprecated fields during deserialization to maintain backward compatibility:
```python
# Remove fields before creating dataclass
data.pop('content', None)
data.pop('_id', None)
entity = EntityNode(**data)
```

### 4. Lock Key Generation
Always sort relationship pairs for deterministic lock ordering (prevents deadlocks):
```python
sorted_key_parts = sorted([src, tgt])
lock_key = f"{sorted_key_parts[0]}-{sorted_key_parts[1]}"
```

### 5. Async Generator Lock Management
Never hold locks across async generator yields - create snapshots instead:
```python
# WRONG: deadlock prone
async with storage._lock:
    for item in storage._data.items():
        yield item  # Lock still held!

# CORRECT: snapshot approach
async with storage._lock:
    snapshot = list(storage._data.items())
# Lock released
for item in snapshot:
    yield item
```

### 6. LightRAG Initialization
Always initialize and finalize storage:
```python
rag = LightRAG(...)
await rag.initialize_storages()  # REQUIRED
try:
    # Use rag
    pass
finally:
    await rag.finalize_storages()  # REQUIRED
```

## LLM Requirements

**Model Selection:**
- **Minimum**: 32B parameters recommended for entity-relationship extraction
- **Context Length**: 32KB minimum, 64KB recommended
- **Avoid**: Reasoning models during indexing (slow)
- **Query Stage**: Use stronger models than indexing for better results

**Recommended Models:**
- OpenAI: gpt-4o, gpt-4o-mini
- Gemini: gemini-2.5-flash, gemini-2.5-pro
- Open Source: Qwen3-30B-A3B (requires temperature tuning)

**Embedding Models:**
- OpenAI: text-embedding-3-large
- Open Source: BAAI/bge-m3
- Must be consistent after first document is indexed

**Reranker (Optional but Recommended):**
- BAAI/bge-reranker-v2-m3
- Jina reranker models
- Significantly improves retrieval quality, especially for "mix" mode

## Utilities and Tools

```bash
# Download cache files for offline deployment
lightrag-download-cache

# Clean LLM query cache
lightrag-clean-llmqc --working-dir ./rag_storage

# Migration tools (see lightrag/tools/)
python -m lightrag.tools.migrate_llm_cache  # Migrate LLM cache between storages
```

## Project Structure

```
lightrag/
├── lightrag.py          # Main LightRAG class
├── operate.py           # Document processing and query logic
├── base.py              # Abstract base classes for storage
├── prompt.py            # LLM prompts for entity extraction
├── utils.py             # Utilities (tokenization, embedding)
├── rerank.py            # Reranking implementations
├── kg/                  # Storage implementations
│   ├── json_kv_impl.py
│   ├── networkx_impl.py
│   ├── postgres_impl.py
│   ├── neo4j_impl.py
│   ├── mongo_impl.py
│   └── ...
├── llm/                 # LLM provider integrations
│   ├── openai.py
│   ├── gemini.py
│   ├── ollama.py
│   ├── anthropic.py
│   └── ...
├── api/                 # FastAPI server
│   ├── lightrag_server.py
│   └── routers/
└── tools/               # CLI utilities

lightrag_webui/          # React frontend
tests/                   # Test suite
examples/                # Example scripts
```

## Common Issues

**1. Dimension Mismatch Error**
- Caused by changing embedding model after indexing
- Solution: Clear working directory or use different workspace

**2. Initialization Error**
- Forgot to call `await rag.initialize_storages()`
- Always call before using LightRAG instance

**3. Frontend Build Required**
- Web UI won't load if frontend not built
- Solution: `cd lightrag_webui && bun run build`

**4. Docker Localhost Issues**
- Use `host.docker.internal` instead of `localhost` in `.env` when LightRAG runs in Docker

**5. Async Event Loop Errors**
- Mixing sync and async code
- Always use `asyncio.run()` for entry point or properly manage event loops

## Development Guidelines

**Code Comments:** Always use English for code comments and documentation

**Testing:** Write tests with appropriate markers (`@pytest.mark.offline` or `@pytest.mark.integration`)

**Storage Implementation:** Follow the abstract base class interfaces in `lightrag/base.py`

**LLM Integration:** Follow existing patterns in `lightrag/llm/` for consistency

**Dependency Injection:** Pass configuration through constructors, not global imports

**Error Handling:** Use comprehensive logging with appropriate levels (DEBUG, INFO, WARNING, ERROR)

## Additional Documentation

- **LightRAG Server API**: `lightrag/api/README.md`
- **Offline Deployment**: `docs/OfflineDeployment.md`
- **Docker Deployment**: `docs/DockerDeployment.md`
- **Frontend Build Guide**: `docs/FrontendBuildGuide.md`
- **Testing Guidelines**: `.clinerules/01-basic.md`
- **Algorithm Details**: `docs/Algorithm.md`
