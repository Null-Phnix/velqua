# Velqua Mesh — Full Project Brief
**Version:** Concept v1.0
**Date:** March 2026
**Author:** Josii
**For:** Velqua Claude Agent

---

## The Problem

Every multi-agent AI setup today has three unsolved problems:

1. **Agents are isolated.** Blackreach finishes a research task and that knowledge dies with the session. The next agent starts blind.
2. **Coordination requires code changes.** Every framework (LangGraph, CrewAI, AutoGen) forces you to rewrite your entire workflow around their SDK. Nothing is drop-in.
3. **There is no visibility.** When three agents are running simultaneously you have no idea what any of them are doing, what they found, or where they are in their task. You are flying blind.

These three problems are widely acknowledged across the developer community and remain unsolved for **local, consumer hardware setups**. All existing solutions are either cloud-dependent, SDK-dependent, or both.

---

## The Solution: Velqua Mesh

Extend Velqua from a **single-app memory proxy** into a **local agent coordination layer**.

Velqua already does the hard part: it sits transparently between any app and its LLM provider, intercepting calls with zero code changes required. The proxy architecture is built. The memory engine (Anamnesis) is built. The FastAPI server is built.

Velqua Mesh adds three layers on top of the existing architecture:

### Layer 1 — Shared Agent Memory
Currently Velqua injects one user's personal facts into conversations. Mesh extends this so that **multiple agents share a common memory pool**.

- Blackreach completes a research task → its findings are written to the shared pool
- A coding agent starts a new session → it can read what Blackreach found
- A planning agent → sees everything both previous agents produced
- No agent needs to know the others exist. They just call Ollama as normal.

The shared memory is namespaced. Each agent has its own private memory AND access to the shared pool. Agents can write to shared, read from shared, or stay isolated — configurable per agent identity.

### Layer 2 — Agent Identity & Registry
Currently Velqua has no concept of multiple agents. Mesh introduces **agent identities**.

Each agent connecting to the proxy is assigned or declares an identity:
```
Blackreach → agent_id: "blackreach"
Coding session → agent_id: "coder"
Planning session → agent_id: "planner"
```

Identity is detected automatically by user-agent string, port, or declared in the request header. No code change required in the agent — Velqua infers it.

The registry tracks:
- Which agents are currently active
- When each agent last made a request
- What each agent's current task/goal is (extracted from conversation context)
- What each agent has written to shared memory

### Layer 3 — Real-Time Dashboard
Currently Velqua's UI has 7 tabs: Facts, Review, Timeline, Insights, Status, Settings, Import.

Mesh adds a **Mesh tab** — a live dashboard showing:

- All currently active agents (live, updating in real-time via websocket)
- Each agent's current task (extracted from its recent prompts)
- What each agent has found / written to shared memory recently
- A shared memory feed — a timeline of everything all agents have produced
- Inter-agent notes — agents can leave structured notes for other agents to pick up

The dashboard requires **zero configuration**. It populates automatically as agents connect and make calls through the proxy.

---

## What Does NOT Change

- **The proxy architecture** — unchanged. Still localhost:11435, still transparent, still zero code changes for existing tools
- **Anamnesis** — unchanged. Still the memory engine underneath. Mesh adds a shared namespace on top of existing personal namespace
- **Provider support** — unchanged. Ollama, OpenAI, Anthropic, all still work
- **Single-agent usage** — unchanged. Someone using Velqua for personal memory with one app sees no difference
- **Test suite** — all 520 existing tests must still pass

---

## Technical Scope

### New Backend Components

**`backend/mesh/`** — new module

```
mesh/
├── registry.py        # agent identity detection + active agent tracking
├── shared_memory.py   # shared memory pool (namespace layer on top of Anamnesis)
├── noteboard.py       # inter-agent structured notes (agents leave/read notes)
├── websocket.py       # real-time event stream for dashboard
└── __init__.py
```

**`backend/routes/mesh.py`** — new route module
- `GET /mesh/agents` — list all active agents + status
- `GET /mesh/memory` — read shared memory pool
- `POST /mesh/memory` — write to shared memory (agents can call this directly or Velqua auto-extracts)
- `GET /mesh/notes` — read noteboard
- `POST /mesh/notes` — leave a note for another agent
- `WS /mesh/stream` — websocket for real-time dashboard updates

**`backend/proxy.py`** — extend existing proxy
- On each intercepted request: detect agent identity, log to registry, check if shared memory should be injected alongside personal memory
- On each intercepted response: auto-extract any findings worth writing to shared pool

### New Frontend Components

**Mesh tab** in existing UI (`src/`)
- Agent cards — one per active agent, live status, current task, recent output
- Shared memory feed — chronological timeline of all cross-agent knowledge
- Noteboard — simple read/write interface for inter-agent notes
- All updates via websocket, no polling

### Database Changes

Extend existing SQLite schema (currently used by Anamnesis):
- `mesh_agents` table — agent registry (id, name, last_seen, current_task, status)
- `mesh_memory` table — shared memory pool (id, agent_id, content, timestamp, tags)
- `mesh_notes` table — inter-agent notes (id, from_agent, to_agent, content, read, timestamp)

---

## Agent Identity Detection (No Code Changes Required)

Velqua Mesh detects agent identity automatically using a priority chain:

1. **Request header** — `X-Velqua-Agent: blackreach` (optional, agents can declare themselves)
2. **User-agent string** — Blackreach sends a recognizable UA, detected automatically
3. **Port** — different agents can connect on different sub-ports (11435, 11436, 11437...)
4. **Process detection** — if on localhost, Velqua can check which process owns the connection
5. **Fallback** — assigned an anonymous ID, grouped as "unknown agent"

---

## Inter-Agent Communication: The Noteboard

The simplest useful coordination primitive. An agent can:

```python
# Blackreach finishes research, leaves a note
POST /mesh/notes
{
  "from": "blackreach",
  "to": "planner",  # or "any" for broadcast
  "content": "Found 47 arXiv papers on mechanistic interpretability. Saved to /results/mech_interp/. Key finding: attention head 4 in layer 8 consistently handles factual recall.",
  "tags": ["research", "complete", "mech_interp"]
}
```

The planner agent, on its next call through the proxy, has this note automatically injected into its context. It doesn't need to poll. It doesn't need to know the noteboard exists. Velqua injects it transparently.

This is the same transparent injection Velqua already does for personal memory — extended to cross-agent notes.

---

## Demo Scenario (What Gets You Seen)

**Setup:** Two agents running simultaneously — Blackreach researching mechanistic interpretability papers, a second Claude session writing a summary document.

**What the user sees in the Mesh dashboard:**
- Left card: "Blackreach — Active — Downloading arXiv papers — 23 found so far"
- Right card: "Claude Writer — Active — Writing summary — Waiting for research"
- Shared memory feed updating live as Blackreach finds papers
- When Blackreach finishes, it leaves a note — Claude Writer's next message automatically knows what was found

**The demo gif:** Split screen. Blackreach terminal on left, Claude session on right, Mesh dashboard in the middle. Blackreach finds papers in real time. Dashboard updates. Claude Writer's response incorporates the findings without being told.

**The tweet:** "Changed one port number. Now all my AI agents share memory and I can watch them work in real time. No SDK. No cloud. Open source."

That's the post that goes viral on r/LocalLLaMA.

---

## Phased Build Plan

### Phase 1 — Agent Registry + Dashboard Skeleton (Week 1)
- `mesh/registry.py` — agent identity detection
- `backend/routes/mesh.py` — basic agent listing endpoint
- Mesh tab in UI — agent cards, static first
- Websocket connection — agents appear/disappear live
- Goal: open dashboard, see Blackreach appear when it starts

### Phase 2 — Shared Memory (Week 1-2)
- `mesh/shared_memory.py` — shared pool on top of Anamnesis
- Proxy extension — auto-inject shared memory alongside personal memory
- Mesh tab — shared memory feed
- Goal: Blackreach findings visible in dashboard and injected into other agents

### Phase 3 — Noteboard (Week 2)
- `mesh/noteboard.py` — structured inter-agent notes
- Proxy extension — auto-inject relevant notes into agent context
- Mesh tab — noteboard UI
- Goal: Blackreach leaves note, Claude Writer receives it transparently

### Phase 4 — Polish + Demo (Week 2-3)
- Demo script — runs both agents simultaneously, records the dashboard
- README updates — Mesh section with architecture diagram
- All existing 520 tests still passing
- New Mesh tests added

---

## Constraints

- **Must run on RTX 4060 (8GB VRAM)** — no cloud dependency, no heavy compute requirement for the coordination layer itself
- **Zero breaking changes** to existing Velqua single-agent usage
- **Zero code changes required** in any agent connecting through the proxy
- **All existing 520 tests must still pass**
- **Python 3.10+**, FastAPI, SQLite — same stack as current Velqua
- **MIT license** — fully open source

---

## Why This Gets Attention

**From developers:**
Every person running multiple local AI tools (Open WebUI + Blackreach + Continue.dev etc.) wants this. The pain is real and daily. One port change to fix it is the lowest possible friction.

**From companies:**
Steel.dev, Skyvern, Browserbase, Mem0 — all building multi-agent systems. A transparent coordination layer that requires no SDK integration is directly useful to their products. This is a cold email attachment.

**From researchers:**
The transparent proxy architecture for agent coordination is publishable. "Zero-modification multi-agent coordination via transparent LLM proxy" — that's an arXiv paper with a novel systems contribution.

---

## Existing Codebase Reference

- **Proxy:** `backend/proxy.py` — intercepts all LLM calls, already handles memory injection
- **Memory engine:** `backend/anamnesis/` — SQLite + FTS5 + TF-IDF + vector embeddings
- **Routes:** `backend/routes/` — modular FastAPI route structure, add `mesh.py` here
- **Frontend:** `src/` — existing tab structure, add Mesh tab alongside existing 7 tabs
- **Server:** `backend/server.py` — mounts all routes, add websocket endpoint here
- **Tests:** `tests/` — 520 passing, add `tests/test_mesh.py`

---

## Success Criteria

1. Open Velqua dashboard → Mesh tab shows all connected agents live
2. Start Blackreach → appears in dashboard within 2 seconds, no config required
3. Blackreach finds something → appears in shared memory feed automatically
4. Second agent connects → receives Blackreach's findings in its next prompt, transparently
5. Record a 30-second demo gif of this working
6. All 520 existing tests still pass + new Mesh tests added
7. README updated with Mesh section

That's the full scope. Build it.
