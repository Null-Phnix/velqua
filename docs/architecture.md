# Architecture

## Overview

Velqua has three main components: the **API server**, the **proxy**, and the **Anamnesis memory engine**. They run in the same process on two ports.

```
Port 8765 — FastAPI server (UI, API, WebSocket)
Port 11435 — Proxy (Ollama-compatible, forwards to your LLM backend)
```

## Request flow

```
LLM Client
    │
    │ POST /api/chat  (or /api/generate, /v1/chat/completions, /v1/messages)
    ▼
Proxy (port 11435)
    │
    ├─ Detect agent identity (X-Velqua-Agent header, User-Agent heuristics)
    ├─ Mesh heartbeat (update agent registry)
    ├─ Retrieve unread Mesh notes for this agent
    │
    ├─ Extract last 3 user messages → build retrieval query
    ├─ Hybrid retrieval: FTS5 (30%) + vector (70%) → top-k facts
    ├─ Topic detection → boost facts matching topic (+30%)
    ├─ Freshness scoring → AdaptiveDecay re-rank
    ├─ Budget-aware selection (token budget from VRAM tier)
    │
    ├─ Inject memory context + Mesh notes into system message
    │
    ▼
Provider (Ollama / OpenAI-compat / Anthropic)
    │
    ▼
LLM Response
    │
    ├─ Auto-learn: extract self-disclosure facts from user messages
    │   ├─ Quality score (0.0–1.0)
    │   ├─ High (>0.7): store immediately
    │   ├─ Medium (0.4–0.7): queue for review
    │   └─ Low (<0.4): discard
    │
    └─ Broadcast Mesh event (WebSocket to dashboard)
```

## Memory engine (Anamnesis)

Velqua vendors a subset of the Anamnesis memory library at `backend/anamnesis/`. Key components:

### Storage
- **SQLite backend**: all facts stored in `data/velqua.db`
- **FTS5 table**: full-text search index, standalone mode (not external-content)
- **In-memory vector store**: sentence-transformer embeddings (all-MiniLM-L6-v2, 384d)

### Retrieval
- **Hybrid search**: combines FTS5 keyword score (30%) and vector cosine similarity (70%)
- **Topic weighting**: TopicDetector classifies each query → 30% boost for matching-topic facts
- **Freshness decay**: AdaptiveDecay with 4-week half-life — recent/confirmed facts rank higher

### Fact lifecycle
```
raw text
  → 26 self-disclosure markers (I am, I work, I live, ...)
  → fiction filter (FANTASY_KEYWORDS word-boundary matching)
  → quality scorer (0.0–1.0)
  → TF-IDF dedup (cosine >= 0.75 = duplicate, increments confirmation_count)
  → topic detection + sentiment analysis
  → stored as Fact(id, content, confidence, topic, metadata...)
  → auto-linked via MemoryGraph
```

## Provider abstraction

```python
BaseProvider (abstract)
├── OllamaProvider       # /api/chat, /api/generate — no auth
├── OpenAICompatProvider # /v1/chat/completions — Bearer token
│   (used for: OpenAI, Groq, LM Studio, llama.cpp, vLLM, LocalAI)
└── AnthropicProvider    # /v1/messages — x-api-key header, system as top-level param
```

The proxy routes all chat requests through `_handle_chat_request()` which:
1. Selects the active provider from the registry
2. Calls the provider's `inject_memory()` method (handles system message placement per-API)
3. Streams or returns the response

## Mesh

The Mesh layer adds three SQLite tables to `data/mesh.db`:

- `mesh_agents`: agent_id, last_seen, task_hint, metadata
- `mesh_memory`: id, agent_id, content, tags, created_at
- `mesh_notes`: id, from_agent, to_agent, content, tags, read, created_at

On every proxied request:
1. `registry.heartbeat(agent_id)` — upserts agent record
2. `noteboard.get_for_agent(agent_id, unread_only=True)` — fetch pending notes
3. Notes are injected into system context, then marked read
4. `broadcast_event()` fires a WebSocket event to all dashboard clients

## Security model

- Binds to `127.0.0.1` by default (not `0.0.0.0`)
- Optional Bearer token auth via `VELQUA_AUTH_TOKEN` env var
- CORS restricted to configured origins
- API keys encrypted at rest (Fernet, PBKDF2 key derived from machine ID + salt)
- All user input validated before processing (file size, format, encoding)
- Frontend uses DOM API only (no innerHTML) to prevent XSS
