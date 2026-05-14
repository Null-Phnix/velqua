# Changelog

All notable changes to Velqua will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - v3.0: Velqua Mesh + Launch (2026-03-07)

### Added — Velqua Mesh (multi-agent coordination layer)
- **Agent registry**: automatic identity detection via `X-Velqua-Agent` header, User-Agent heuristics (Blackreach, Open WebUI, Continue.dev), and fallback anonymous grouping
- **Shared memory pool**: cross-agent fact store at `/mesh/memory` — agents write findings, others receive them transparently via proxy context injection
- **Noteboard**: structured inter-agent notes (`/mesh/notes`) — agents leave notes for each other, received on next proxied request without polling
- **Real-time dashboard**: WebSocket `/mesh/stream` — Mesh tab shows live agent cards, shared memory feed, noteboard updates
- **Transparent proxy integration**: agent heartbeat + unread note injection on every proxied request, zero code changes required in agents
- **58 new tests** covering registry, shared memory, noteboard, REST endpoints

### Added — Proxy improvements
- `POST /proxy/summarize-session` — ingest full conversation in one shot, extract facts from all turns
- Multi-turn retrieval: context built from last 3 user messages (not just the last one)
- Assistant echo learning: extract second-person confirmations from model responses
- Memory Preview UI: Status tab → preview which facts would be injected for any query

### Added — UX
- First-run banner when no facts exist (import CTA, dismissable)
- Compact Memory button in Status tab (dedup pass)
- Real SSE progress stream for large imports (replaces fake progress timer)
- Empty-state illustration in Facts tab with import shortcut
- 8th UI tab: Mesh (agent cards, shared memory feed, noteboard, note composer)

### Changed
- Version bumped to 3.0.0
- README rewritten for competitive positioning
- QUICKSTART.md added

### Tests
- 627 tests total (up from 404 in v1.3)
- 85%+ coverage

---

## [1.1.0] - v1.1: Intelligence & Polish (2026-02-20)

### Auto-Learning Intelligence
- **Quality scoring** for extracted facts (0.0-1.0)
  - High-value markers (name, location, job) boost score
  - Transient markers (debugging, testing, fixing) reduce score
  - Code content, questions, generic requests penalized
  - Minimum quality threshold: 0.4 (below = discarded)
- **Review queue** for medium-quality facts (0.4-0.7)
  - Pending facts held for user approval before storage
  - Approve/reject individual or bulk
  - High-quality facts (>0.7) auto-accepted
  - Review tab with badge showing pending count
- **Contradiction detection** wired into auto-learning
  - New facts checked against existing knowledge
  - High-confidence contradictions auto-supersede old facts
  - API endpoint to find contradictions across all facts

### Fact Freshness & Decay
- **AdaptiveDecay** scoring for retrieval ranking
  - Recent/confirmed facts rank higher in context injection
  - 4-week half-life for personal facts (slower than default)
  - Confirmation count and access history boost freshness
  - Applied to both hybrid and FTS retrieval paths

### Backup & Data Management
- **Database backup** (POST /backup/create, GET /backup/list)
- **Backup restore** with safety backup of current DB
- **Fact export** as downloadable JSON (GET /export/facts)
- **Fact import** from exported JSON (POST /import/facts-json)
- **Import history** tracking (file type, facts count, timestamp)
- **Import undo** — delete all facts from a specific import batch

### Testing
- **38 API endpoint tests** (FastAPI TestClient)
  - Health, import (4 formats), list, delete, search, edit, merge
  - Bulk delete, stats, timeline, tags, types, legacy endpoints
  - Review queue, backup/export, import history, contradictions
- **8 quality scoring tests** (transient, code, questions, specific)
- **8 pending store tests** (add, approve, reject, persistence)
- **Total: 148 tests** (was 73), all passing in ~1.3s

### UI Enhancements
- **Loading spinners** on all tabs during API calls
- **Error boundaries** with retry buttons on all data views
- **Review tab** with approve/reject per fact and bulk actions
- **Proxy health dashboard** — real-time status of:
  - Proxy running status, backend URL
  - Memory budget, vector retrieval status
  - Auto-learning stats (learned, pending, duplicates)
- **Backup & Export panel** in Status tab
- **Import history** with undo capability
- **Fact export** as downloadable JSON file

### New Files
- tests/conftest.py — Shared test configuration
- tests/test_server_api.py — 54 API endpoint tests

### Modified Files
- backend/auto_learner.py — Quality scoring, PendingFactStore, contradiction check
- backend/proxy.py — AdaptiveDecay for freshness, re-ranking by freshness
- backend/server.py — Review queue, backup, export, import history, undo, contradictions
- src/index.html — Review tab, health dashboard, loading states, import history

## [1.0.0] - v1.0: Full Memory System (2026-02-20)

### Core Features
- **Auto-learning from proxy conversations**
  - Extracts self-disclosure facts in real-time from user messages
  - 26 fact markers (I am, I work, I live, I prefer, I built, etc.)
  - Fiction filtering applied to live extraction
  - Fire-and-forget async (zero latency impact on proxy)
- **Fact deduplication via TF-IDF cosine similarity**
  - Threshold: 0.75 similarity = duplicate
  - Duplicates increment confirmation_count + boost confidence
  - Applied to all import paths and auto-learning
- **Vector retrieval (SentenceTransformer + InMemoryVectorStore)**
  - all-MiniLM-L6-v2 embeddings (384 dimensions)
  - Hybrid search: FTS 30% + vector 70%
  - Existing facts indexed on proxy startup
  - New facts indexed immediately on learn
  - ~300MB overhead, works on 8GB VRAM
- **Smart context window**
  - Budget-aware fact injection (counts tokens per fact)
  - VRAM tiers: minimal (200), standard (500), generous (1000+)
  - Both /api/generate and /api/chat respect token budget

### Fact Management
- **API endpoints:** search, edit (PATCH), bulk delete, merge, stats, timeline
- **UI:** search bar, inline edit, select + bulk delete, merge selected
- **Pagination:** 50 facts per page with prev/next navigation
- **Tags:** add/remove tags via API, stored in fact metadata
- **Categories:** filter by FactType (personal, preference, professional, project, etc.)
- **Timeline view:** facts grouped by date, visual timeline with dots

### Multi-Model Support
- **Ollama:** /api/generate, /api/chat (existing)
- **OpenAI-compatible:** /v1/chat/completions (new)
  - Works with llama.cpp, vLLM, LocalAI, LM Studio, text-generation-webui
  - Set VELQUA_OPENAI_BASE_URL to configure backend
  - Full memory injection + auto-learning on all endpoints

### Security
- **Localhost-only by default** (127.0.0.1, not 0.0.0.0)
- **Optional auth token** (VELQUA_AUTH_TOKEN for remote access)
- **CORS restricted** to configured origins (not wildcard *)

### Infrastructure
- **System tray** (pystray) with background mode and menu
- **PyInstaller build script** for single-file executable
- **Structured logging** (replaced all print statements)
- **73 unit tests** (auto_learner: 18, file_detector: 33, validators: 22)

### New Files
- backend/auto_learner.py - Real-time fact extraction from conversations
- backend/tray.py - System tray with background mode
- build_exe.py - PyInstaller build script
- tests/test_auto_learner.py - 18 tests for auto-learning

## [0.2.0] - Sprint 2: Production Foundation (2026-02-11)

### Security
- **CRITICAL:** Fixed hardcoded /tmp path vulnerability → now uses secure tempfile
- **CRITICAL:** Added input validation (file size, format, encoding) before processing
- **CRITICAL:** Fixed 7 XSS vulnerabilities → replaced innerHTML with DOM manipulation
- Added filename sanitization (prevents path traversal attacks)
- Added ValidationError exception class for better error handling

### Features
- **ChatGPT import now works** (was stub returning "coming soon")
  - Full conversation parsing with mapping structure
  - User message extraction with self-disclosure patterns
  - Fiction filtering same as Claude imports
- **Windows support added**
  - run_velqua.bat (Command Prompt)
  - run_velqua.ps1 (PowerShell with colors)
  - Port checking and cleanup on Windows
- **User-friendly error messages**
  - Replaced raw Python errors with friendly messages
  - Added retry buttons on upload failures
  - Server not running detection
  - File too large detection (413 errors)
  - Invalid JSON detection with helpful hints

### Testing
- Added 22 unit tests for validators.py (validate_upload, sanitize_filename)
- Added 33 unit tests for file_detector.py (detect, extract functions)
- **Total: 55 passing tests** (was 0)
- Test coverage: 80%+ on critical paths (validators, file_detector)
- Fixed edge cases: path traversal, null bytes, unicode, empty files

### Code Quality
- **Configuration management:** Created backend/config.py with VelquaConfig class
  - Externalized all magic numbers (ports, file sizes, limits, thresholds)
  - Added environment variable support (.env)
  - Created .env.example template
- **Dead code cleanup:**
  - Removed server_old.py and server_backup.py
  - Created .gitignore for __pycache__, logs, data files
- **Improved file handling:**
  - File size validated BEFORE reading into memory
  - Proper temp file cleanup in all error paths
  - Cross-platform temp directories (not hardcoded /tmp)

### Documentation
- Updated README with Windows instructions
- Added Configuration section (environment variables)
- Added Troubleshooting section (common issues + fixes)
- Updated roadmap with Sprint 2 completion
- Created CHANGELOG.md (this file)
- Updated badges (tests: 55/55, coverage: 80%+, status: Production Ready)

### Performance
- No performance regressions
- Test suite runs in <1 second (55 tests)

## [0.1.0] - Sprint 1: Working MVP (2026-02-10)

### Initial Release
- Claude memories.json import (validated with real user data)
- Smart file detection (3 formats: memories, conversations, projects)
- Fiction filtering (context detection for creative writing)
- Ollama proxy with memory injection
- Web UI with drag-drop upload
- E2E test suite (5/5 passing)
- Hardware-aware memory budgets

### Known Issues
- ChatGPT import returns "coming soon" (stub)
- No Windows support (bash scripts only)
- Security: Hardcoded /tmp path
- Security: XSS vulnerabilities (innerHTML usage)
- No unit tests (only E2E)
- Print statements instead of structured logging
- Magic numbers hardcoded throughout

---

## Upgrade Guide

### 0.1.0 → 0.2.0

**Breaking changes:** None - fully backward compatible

**Database:** No migration needed (schema unchanged)

**Configuration (optional):**
1. Copy `.env.example` to `.env`
2. Customize as needed (defaults match 0.1.0 behavior)

**Windows users:**
- Can now use `run_velqua.bat` or `run_velqua.ps1`
- No more WSL required!

**ChatGPT users:**
- Import now works! Export from ChatGPT Settings → Data controls → Export
- Upload conversations.json to Velqua
