# Velqua Mesh

Mesh turns Velqua's proxy into a local multi-agent coordination layer. Multiple AI agents share memory, exchange notes, and appear on a live dashboard — with zero code changes in the agents themselves.

## Concepts

### Agent registry

Every process that sends requests through the proxy is tracked as an agent. Velqua identifies agents by:

1. `X-Velqua-Agent: <name>` request header (highest priority, explicit)
2. User-Agent string heuristics:
   - `blackreach` → "blackreach"
   - `open-webui` → "open-webui"
   - `continue` → "continue"
3. Fallback: "unknown" (all unidentified requests grouped together)

Agents are registered on first request. The registry records:
- `agent_id`: string identifier
- `last_seen`: timestamp
- `task_hint`: first 200 chars of the last user message (extracted from conversation)

Agents inactive for more than 10 minutes are marked inactive in the dashboard.

### Shared memory pool

A cross-agent fact store at `/mesh/memory`. Any agent (or you) can write a finding; other agents can retrieve relevant entries.

Unlike personal memory (which is per-user), shared memory is explicitly written — it doesn't auto-populate from conversations.

```bash
# Write
POST /mesh/memory
{"agent_id": "researcher", "content": "MMLU score improved 23% with longer context", "tags": ["benchmark"]}

# Read (all entries)
GET /mesh/memory

# Read (filter by agent)
GET /mesh/memory?agent_id=researcher

# Delete
DELETE /mesh/memory/{id}
```

### Noteboard

Structured inter-agent messages. An agent leaves a note addressed to another; that agent receives it injected into its system context on its next proxied request.

```bash
# Post a note
POST /mesh/notes
{
  "from_agent": "researcher",
  "to_agent": "writer",
  "content": "Research phase done. Key finding: attention head 4 in layer 8 handles factual recall.",
  "tags": ["handoff", "complete"]
}

# Note is injected into "writer"'s next request automatically.
# No polling. No code changes in writer.
```

Notes are marked read after injection. They don't accumulate across requests.

## Setup

Nothing to configure. Start Velqua and point agents at the proxy:

```bash
python backend/server.py

# In your agents:
OLLAMA_HOST=localhost:11435 python agent.py

# Or declare identity explicitly:
# Add header: X-Velqua-Agent: my-agent
```

## Dashboard

Open http://localhost:8765 → Mesh tab.

The dashboard shows:
- **Agent cards**: all active agents, last seen time, current task
- **Shared memory feed**: all entries in the shared pool, with tags and author
- **Noteboard**: all notes, read/unread status, source and target agents
- **Note composer**: write and send notes directly from the UI
- **WebSocket connection status**: live updates or polling fallback

## API reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/mesh/agents` | List all registered agents |
| GET | `/mesh/agents/{id}` | Single agent details |
| GET | `/mesh/memory` | Read shared memory pool (optional `?agent_id=` filter) |
| POST | `/mesh/memory` | Write an entry to shared memory |
| DELETE | `/mesh/memory/{id}` | Delete a shared memory entry |
| GET | `/mesh/notes` | Read all notes (optional `?agent_id=` and `?unread=true` filters) |
| POST | `/mesh/notes` | Post a note |
| PUT | `/mesh/notes/{id}/read` | Mark a note as read |
| DELETE | `/mesh/notes/{id}` | Delete a note |
| GET | `/mesh/status` | Aggregate stats (agent count, memory count, note count) |
| WS | `/mesh/stream` | WebSocket real-time event stream |

## WebSocket events

Connect to `ws://localhost:8765/mesh/stream` to receive live events:

```json
{"type": "snapshot", "agents": [...], "memory": [...], "notes": [...]}
{"type": "agent_heartbeat", "agent": {...}}
{"type": "memory_written", "entry": {...}}
{"type": "note_posted", "note": {...}}
{"type": "ping"}
```

The dashboard connects on tab load and falls back to 30-second polling if WebSocket fails.

## Example: pipeline handoff

```python
# step1_researcher.py — runs as "researcher" agent
import httpx

client = httpx.Client(
    base_url="http://localhost:11435",
    headers={"X-Velqua-Agent": "researcher"}
)

# Do research via LLM
response = client.post("/api/chat", json={
    "model": "llama3",
    "messages": [{"role": "user", "content": "Summarize mechanistic interpretability."}],
    "stream": False,
})
summary = response.json()["message"]["content"]

# Leave findings for the writer
httpx.post("http://localhost:8765/mesh/notes", json={
    "from_agent": "researcher",
    "to_agent": "writer",
    "content": f"Research complete: {summary[:500]}",
    "tags": ["handoff"],
})
```

```python
# step2_writer.py — runs as "writer" agent
# Velqua automatically injects the researcher's note on first request
import httpx

client = httpx.Client(
    base_url="http://localhost:11435",
    headers={"X-Velqua-Agent": "writer"}
)

# The note from the researcher is injected into this request's system context
response = client.post("/api/chat", json={
    "model": "llama3",
    "messages": [{"role": "user", "content": "Write a blog post intro."}],
    "stream": False,
})
print(response.json()["message"]["content"])
# Will reference the researcher's findings because Velqua injected them
```
