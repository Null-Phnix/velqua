# Velqua

**Give your local LLM persistent memory — no code changes required.**

Point any Ollama-compatible app at `localhost:11435` instead of `localhost:11434`. Every conversation automatically injects your personal context. Your AI remembers who you are across sessions, models, and tools — without touching a single line of application code.

![Tests](https://img.shields.io/badge/tests-645%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen)
![Version](https://img.shields.io/badge/version-3.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Why Velqua

Every other memory layer requires code changes. Mem0 needs API calls. Zep needs SDK integration. OpenMemory needs MCP configuration per-tool. Letta requires rewriting your agent around its framework.

Velqua is a **transparent proxy**. Your existing apps work unchanged.

```
Before: your app → localhost:11434 (Ollama)
After:  your app → localhost:11435 (Velqua) → localhost:11434 (Ollama)
```

That's the full integration. One port number.

---

## What it does

```
You: What do you know about me?

Without Velqua:
  LLM: I don't have any information about you.

With Velqua:
  LLM: You're a software engineer in Toronto, you have ADHD, you prefer
       Python, and you're building a sovereign AI infrastructure called The Hive.
```

1. **Import** your Claude or ChatGPT conversation history
2. **Extract** personal facts (filters fiction, scores quality, deduplicates)
3. **Inject** relevant facts into every LLM request via the transparent proxy
4. **Learn** new facts in real-time as you chat

---

## Comparison

| Feature | **Velqua** | Mem0 | Zep | OpenMemory |
|---------|-----------|------|-----|-----------|
| Zero code changes | **Yes** | No — API calls | No — SDK | No — MCP config per tool |
| Local-first, no cloud | **Yes** | Cloud product | Cloud / self-host | Yes (Docker stack) |
| Import Claude/ChatGPT history | **Yes** | No | No | No |
| Multi-agent mesh | **Yes** | No | No | No |
| Native desktop app | **Yes** | No | No | No |
| Multi-provider (OpenAI, Anthropic, Groq) | **Yes** | Yes | Yes | Via Mem0 |
| Setup complexity | Port change | API integration | SDK + graph DB | Docker + Postgres + Qdrant |

**Velqua's niche:** local-first personal memory with zero application integration cost. If you're running Ollama or any OpenAI-compatible backend, Velqua is the fastest path to an AI that knows you.

---

## Quick start

```bash
pip install -e .
python backend/server.py
```

Open http://localhost:8765. The setup wizard walks you through provider configuration on first launch.

Or as a native desktop window:

```bash
pip install -e ".[desktop]"
python backend/desktop.py
```

### Import your memory

1. Export from Claude (Settings > Export data) or ChatGPT (Settings > Data controls > Export)
2. Drag the JSON file into the Velqua UI
3. Velqua auto-detects the format and extracts facts

### Connect your LLM apps

Point any Ollama-compatible client at `localhost:11435` instead of `localhost:11434`. Done.

Works with Open WebUI, Continue.dev, any custom script using httpx/requests, or any tool that lets you configure an Ollama base URL.

For cloud providers, configure your API key in the Settings tab and point at Velqua's OpenAI-compatible endpoint:

```
OPENAI_BASE_URL=http://localhost:11435/v1
```

---

## Architecture

```
                    +-----------------------+
                    |  Web UI / Desktop     |
                    |  8 tabs: Facts,       |
                    |  Review, Timeline,    |
                    |  Insights, Status,    |
                    |  Settings, Import,    |
                    |  Mesh                 |
                    +----------+------------+
                               |
                    +----------v------------+
                    |  FastAPI Server       |
                    |  8 route modules      |
                    |  License middleware   |
                    |  Auth middleware      |
                    +----------+------------+
                               |
              +----------------+----------------+
              |                                 |
   +----------v------------+       +------------v-----------+
   |  Anamnesis Engine     |       |  Provider Registry     |
   |  SQLite + FTS5        |       |  Ollama                |
   |  TF-IDF dedup         |       |  OpenAI / compatible   |
   |  Vector embeddings    |       |  Anthropic             |
   |  Topic detection      |       |  Encrypted keystore    |
   |  Contradiction engine |       +-----------+------------+
   +----------+------------+                   |
              |                                |
User --> Proxy (:11435) -----------------------+---> LLM backend
         inject memory
         auto-learn facts
         Mesh coordination
```

---

## Features

### Zero-integration memory

- **Transparent proxy**: memory context injected as a system message, invisible to the user and app
- **Hybrid retrieval**: FTS5 full-text (30%) + sentence-transformer vector similarity (70%)
- **Topic-weighted**: 30% boost for facts matching the current conversation topic
- **Budget-aware**: configurable token budget scales with GPU VRAM (200–2000 tokens)
- **Auto-learning**: extracts facts from conversations in real-time (26 self-disclosure markers)

### Memory intelligence

- **Smart import**: auto-detects Claude memories/conversations/projects and ChatGPT exports
- **Deduplication**: TF-IDF cosine similarity prevents storing the same fact twice
- **Fiction filtering**: keyword-based detection filters creative writing from real facts
- **Quality scoring**: facts scored 0.0–1.0 — high auto-stores, medium goes to review, low discarded
- **Topic detection**: auto-classifies facts by topic for weighted retrieval
- **Sentiment analysis**: tracks emotional valence of stored facts
- **Contradiction detection**: flags conflicting facts with resolution workflow
- **Freshness decay**: AdaptiveDecay with 4-week half-life ranks recent facts higher

### Multi-provider support

- **Provider abstraction**: Ollama, OpenAI-compatible, and Anthropic via unified interface
- **Runtime switching**: add/remove/swap providers from the Settings UI without restart
- **Encrypted keystore**: API keys stored with Fernet encryption, machine-derived key
- **Connection testing**: verify provider connectivity before saving

### Mesh — multi-agent coordination

Velqua Mesh extends the proxy into a **local multi-agent coordination layer**. Multiple AI agents connect through the same proxy port and share memory, leave notes for each other, and appear on a live dashboard — without any of them knowing the others exist.

See [Mesh](#mesh) section below for full details.

### Operations

- **Review queue**: medium-quality facts held for approval with topic/emotion/category badges
- **Edit-approve**: modify facts during review before accepting
- **Database backup**: one-click backup/restore with safety snapshots
- **Import history**: track all imports with undo capability
- **Export**: download all facts as JSON
- **Compact memory**: deduplication pass removes near-duplicate facts (Jaccard >= 0.8)

### Desktop & packaging

- **Native desktop app**: pywebview window with JS bridge (no browser needed)
- **Cross-platform installers**: Windows (.exe via Inno Setup), macOS (.dmg), Linux (AppImage)
- **System tray**: optional tray icon with status and quick actions
- **Setup wizard**: 5-step guided first-run (provider selection, connection test, import)

### Security

- **Localhost-only by default**: binds to 127.0.0.1
- **Optional auth token**: set `VELQUA_AUTH_TOKEN` for remote access
- **CORS restricted**: only configured origins allowed
- **Input validation**: file size limits, JSON validation, filename sanitization
- **No XSS**: all dynamic content rendered via DOM API, never innerHTML
- **Encrypted keys**: API keys never stored in plaintext

---

## Mesh

Velqua Mesh is a local multi-agent coordination layer built into the proxy. Multiple AI agents — Blackreach, Claude sessions, Open WebUI, coding assistants — connect through the same proxy port. They share memory, leave notes for each other, and appear live on the Mesh dashboard without any of them needing to know the others exist.

### How it works

Every agent connecting through `localhost:11435` is automatically detected and registered:

1. `X-Velqua-Agent: <name>` request header (explicit declaration)
2. User-Agent string heuristics (Blackreach, Open WebUI, Continue.dev detected automatically)
3. Fallback: "unknown" (anonymous grouping)

On every proxied request, Velqua:
- Records the agent heartbeat (last seen, current task from conversation context)
- Injects any unread notes addressed to that agent into its system context
- Broadcasts the event to the Mesh dashboard via WebSocket

### Setup

Nothing to configure. Point multiple tools at the proxy:

```bash
# Terminal 1 — research agent
OLLAMA_HOST=localhost:11435 python my_agent.py

# Terminal 2 — another session (e.g. Open WebUI already pointing at :11435)

# Open Velqua UI → Mesh tab
# Both agents appear within seconds
```

Agents can declare their identity with a header:

```
X-Velqua-Agent: my-agent-name
```

### Noteboard — inter-agent notes

```bash
# Agent 1 finishes research, leaves a note for Agent 2
curl -X POST localhost:8765/mesh/notes \
  -H "Content-Type: application/json" \
  -d '{
    "from_agent": "researcher",
    "to_agent": "writer",
    "content": "Research complete. Found 47 papers on mechanistic interpretability.",
    "tags": ["research", "handoff"]
  }'

# On writer's next request through the proxy,
# this note is automatically injected into its system context.
# No code change needed in the writer.
```

### Shared memory

```bash
# Write a finding to the shared pool
curl -X POST localhost:8765/mesh/memory \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "researcher",
    "content": "Benchmark shows 23% improvement on MMLU with longer context",
    "tags": ["benchmark", "mmlu"]
  }'

# Read the pool
curl localhost:8765/mesh/memory
```

### Two agents, one memory — no code changes

```python
# Agent 1 (researcher) — unchanged from how it was before Velqua
import httpx
client = httpx.Client(base_url="http://localhost:11435")

# Agent 2 (writer) — also unchanged
# On its next proxied request, Velqua injects any notes
# addressed to it from the researcher.

# Post a note programmatically (or from the Mesh dashboard UI)
import requests
requests.post("http://localhost:8765/mesh/notes", json={
    "from_agent": "researcher",
    "to_agent": "writer",
    "content": "Summary ready at /data/summary.md",
    "tags": ["handoff"],
})
```

### Mesh endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/mesh/agents` | List active agents |
| GET | `/mesh/agents/{id}` | Agent details |
| GET | `/mesh/memory` | Read shared memory pool |
| POST | `/mesh/memory` | Write to shared memory |
| DELETE | `/mesh/memory/{id}` | Delete an entry |
| GET | `/mesh/notes` | Read noteboard |
| POST | `/mesh/notes` | Post a note |
| PUT | `/mesh/notes/{id}/read` | Mark note as read |
| DELETE | `/mesh/notes/{id}` | Delete a note |
| GET | `/mesh/status` | Overall mesh stats |
| WS | `/mesh/stream` | Real-time event stream (dashboard) |

---

## Configuration

All settings via environment variables (or `.env` file):

```bash
# Server
VELQUA_HOST=127.0.0.1       # Bind address (localhost for security)
VELQUA_PORT=8765             # API server port
VELQUA_PROXY_PORT=11435      # Ollama proxy port

# Security
VELQUA_AUTH_TOKEN=            # Set to require Bearer token on API calls
VELQUA_CORS_ORIGINS=http://localhost:8765,http://127.0.0.1:8765

# Limits
VELQUA_MAX_UPLOAD_MB=100
VELQUA_MAX_CONVERSATIONS=50
VELQUA_MEMORY_BUDGET=200     # Token budget for memory injection
VELQUA_PROXY_TIMEOUT=300     # Backend request timeout (seconds)

# Storage
VELQUA_DB_PATH=data/velqua.db
VELQUA_LOG_LEVEL=INFO
```

Provider configuration (API keys, URLs, active provider) is managed through the Settings UI and stored encrypted in `data/keys.enc`.

---

## Hardware scaling

| GPU VRAM | Memory Budget | Injected Tokens | Notes |
|----------|--------------|-----------------|-------|
| 8GB      | minimal      | 200             | RTX 4060 |
| 16GB     | standard     | 500             | RTX 4070/5070 |
| 24GB     | generous     | 1000            | RTX 4090/5090 |
| 128GB+   | generous     | 2000            | Rubin-class |

Override via API: `curl -X POST localhost:11435/proxy/config -d '{"gpu_vram_gb": 24}'`

---

## Project structure

```
velqua/
  backend/
    server.py             # App factory, middleware, startup
    proxy.py              # Multi-provider proxy with memory injection
    auto_learner.py       # Real-time fact extraction + quality scoring
    file_detector.py      # File type detection + format-specific extraction
    config.py             # Centralized configuration (env vars)
    validators.py         # Upload validation + filename sanitization
    logging_config.py     # Structured logging setup
    keystore.py           # Fernet-encrypted API key storage
    license.py            # LemonSqueezy license management
    desktop.py            # pywebview native window + JS bridge
    updater.py            # GitHub releases version checker
    tray.py               # System tray launcher (optional)
    mesh/
      db.py               # SQLite mesh DB (agents, memory, notes tables)
      registry.py         # Agent registry + identity detection
      shared_memory.py    # Cross-agent shared memory pool
      noteboard.py        # Inter-agent note delivery
    providers/
      __init__.py          # Provider registry (singleton)
      base.py              # BaseProvider abstract class
      ollama.py            # Ollama provider
      openai_compat.py     # OpenAI-compatible provider (OpenAI, Groq, local)
      anthropic.py         # Anthropic provider
    routes/
      __init__.py          # Route registration
      _shared.py           # Shared memory instance + import history store
      facts.py             # CRUD, search, merge, bulk ops, tags, types
      imports.py           # Smart import, ChatGPT import, JSON import
      review.py            # Pending fact approval/rejection + edit-approve
      backup.py            # Backup, restore, export
      system.py            # Health, analytics, graph, emotional/temporal recall
      settings.py          # Provider config, memory budget, connection test
      license.py           # License activation/deactivation/status
      mesh.py              # Mesh REST endpoints + WebSocket stream
    anamnesis/             # Vendored memory engine
  src/
    index.html             # Web UI structure (8 tabs + wizard + modals)
    styles.css             # Styling with micro-animations
    api.js                 # API base URL + fetch wrapper
    app.js                 # Entry point, tab routing, event wiring
    components/
      facts.js             # Facts tab (search, CRUD, bulk ops, tags)
      review.js            # Review queue (approve/reject/edit)
      timeline.js          # Timeline view
      insights.js          # Analytics dashboard
      status.js            # System status, backups, contradictions
      settings.js          # Provider config, memory settings
      license.js           # License management
      wizard.js            # First-run setup wizard
      import.js            # File import + drag-and-drop
      modal.js             # Modal system (confirm, prompt, toast)
      mesh.js              # Mesh dashboard (agents, memory feed, noteboard)
  packaging/
    build_windows.py       # PyInstaller + Inno Setup → .exe installer
    build_macos.py         # PyInstaller → .app + hdiutil → .dmg
    build_linux.py         # PyInstaller + appimagetool → AppImage
  tests/                   # 645 tests across 16 test files
  data/                    # Runtime data (gitignored)
  pyproject.toml
```

---

## Testing

```bash
python -m pytest tests/ -v
```

645 tests, ~15s runtime. Tests use FastAPI's TestClient with a temporary database — no server process needed.

---

## API reference

The server auto-generates OpenAPI docs at http://localhost:8765/docs when running.

Key endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | System health + fact counts |
| POST | `/import/smart` | Auto-detect and import any supported format |
| POST | `/import/smart/stream` | SSE progress stream for large imports |
| GET | `/facts/list` | Paginated fact listing |
| GET | `/facts/search?q=` | Full-text search |
| PATCH | `/facts/{id}` | Edit a fact |
| DELETE | `/facts/{id}` | Delete a fact |
| POST | `/facts/merge` | Merge multiple facts into one |
| POST | `/facts/compact` | Dedup near-duplicate facts |
| GET | `/facts/types` | List all fact types |
| GET | `/review/pending` | List facts awaiting review (with badges) |
| POST | `/review/approve/{id}` | Approve a pending fact |
| POST | `/review/edit-approve/{id}` | Edit and approve in one step |
| POST | `/proxy/summarize-session` | Ingest full conversation, extract all facts |
| POST | `/backup/create` | Create database backup |
| GET | `/backup/list` | List available backups |
| POST | `/backup/restore/{name}` | Restore from backup |
| GET | `/export/facts` | Export all facts as JSON |
| GET | `/import/history` | View import log |
| POST | `/import/undo/{batch_id}` | Undo an import |
| GET | `/analytics/report` | Memory health + topic analysis |
| GET | `/analytics/quality` | Per-fact quality scoring |
| GET | `/graph/stats` | Memory graph statistics |
| GET | `/graph/links/{id}` | Get related facts for a fact |
| GET | `/settings` | Current provider + memory config |
| PUT | `/settings` | Update memory settings |
| POST | `/settings/provider` | Add/update a provider |
| POST | `/settings/test-connection` | Test provider connectivity |
| GET | `/settings/models` | List available models |
| POST | `/license/activate` | Activate license key |
| GET | `/license/status` | License status |
| GET | `/update/check` | Check for new versions |

---

## License

MIT
