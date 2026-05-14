# Velqua v3.1.0 — Production Ready

**Release date:** 2026-03-18

Velqua is a transparent memory proxy that gives local LLMs persistent personal memory. Point your AI apps at `localhost:11435` instead of your LLM — Velqua enriches every request with relevant context from your knowledge base, then forwards it. No code changes required.

---

## What's New in v3.1.0

### Production Hardening
- **849 tests, 99% coverage, 0 warnings** — comprehensive test suite across all modules
- **XSS fixes** — all API response data rendered via `textContent` (no innerHTML with user data)
- **Security audit passed** — localhost-only binding, restrictive CORS, no wildcard origins
- **Performance validated** — 64ms hot-path memory injection, 1.5ms search over 1000+ facts

### TypeScript Frontend
- Full TypeScript migration (13 modules, strict mode)
- Compiled JS served via FastAPI static files
- Type-safe API response handling throughout

### Mesh Networking (Beta)
- Multi-agent coordination via shared memory and noteboard
- Agent registry with heartbeat tracking
- WebSocket stream for real-time agent events
- 10 mesh API endpoints

### UI Polish
- 9 CSS bugs fixed (tab overflow, fact cards, first-run banner, pagination)
- Cold start UX: embedding model cache detection in status tab
- Fact stats bar shows total count
- Beta chip on Mesh tab

### Packaging
- `pip install velqua` — `velqua` command starts the server
- AppImage for Linux desktop
- Desktop mode via `velqua-desktop` (pywebview)
- PyInstaller-safe data directory (persistent across launches)

---

## Features

- **Zero-integration memory** — hybrid retrieval (FTS5 + vector cosine), budget-aware injection, auto-learning
- **Memory intelligence** — 5 import formats, deduplication, fiction filtering, quality scoring, topic detection, sentiment analysis, contradiction detection, temporal decay
- **Multi-provider** — Ollama, OpenAI, Anthropic, xAI (any OpenAI-compatible)
- **64 API endpoints** across 8 route modules
- **8-tab web UI** — Facts, Review, Timeline, Insights, Status, Settings, Import, Mesh
- **Desktop app** — native window via pywebview, system tray support
- **License system** — trial mode, LemonSqueezy activation, grace period
- **Encrypted keystore** — PBKDF2 key derivation, Fernet encryption for API keys
- **E2E tested** — 64 Playwright tests across all UI tabs

## Install

```bash
# PyPI
pip install velqua
velqua

# From source
git clone https://github.com/your-org/velqua.git
cd velqua
pip install -e .
velqua

# AppImage (Linux)
chmod +x Velqua-x86_64.AppImage
./Velqua-x86_64.AppImage
```

## Configuration

Point your LLM client to `http://localhost:11435` instead of your Ollama/OpenAI endpoint. Velqua forwards all requests while injecting relevant memory context.

Web UI at `http://localhost:8765` — manage facts, review pending learnings, configure providers.

## Requirements

- Python 3.10+
- ~90MB disk for sentence-transformers model (downloaded on first use)
- SQLite (bundled with Python)

## Full Changelog

See [CHANGELOG.md](CHANGELOG.md) for the complete version history.
