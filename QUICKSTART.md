# Velqua Quick Start

Five minutes from zero to an AI that knows you.

---

## 1. Install

```bash
git clone <repo>
cd velqua
pip install -e .
```

Python 3.10+ required. No GPU required (embeddings run on CPU, ~300MB RAM).

---

## 2. Start

```bash
python backend/server.py
```

Open http://localhost:8765. The setup wizard runs on first launch.

---

## 3. Choose a provider

**Using Ollama (local, free):**

Make sure Ollama is running at `localhost:11434`. Velqua detects it automatically.

**Using OpenAI/Anthropic/Groq:**

1. Go to the Settings tab
2. Click the provider card
3. Paste your API key
4. Click "Test Connection"

---

## 4. Import your memory

1. **From Claude**: Settings > Privacy & Data > Export Data → download the zip, extract `conversations.json`
2. **From ChatGPT**: Settings > Data Controls > Export Data → download, extract `conversations.json`
3. Drag the file into the Velqua Import tab

Velqua auto-detects the format. For 1000 conversations, expect ~30 seconds and ~50–200 extracted facts.

---

## 5. Connect your apps

Change one line in any app that uses Ollama:

```
Before: http://localhost:11434
After:  http://localhost:11435
```

Examples by tool:

| Tool | Where to change |
|------|----------------|
| Open WebUI | Admin > Settings > Ollama Base URL |
| Continue.dev | `.continue/config.json` → `apiBase` |
| AnythingLLM | Settings > LLM Provider > Ollama URL |
| Custom script | Change base URL in httpx/requests client |

That's it. Your next conversation will have memory injected automatically.

---

## 6. Verify it's working

```bash
curl http://localhost:11435/api/chat -d '{
  "model": "llama3",
  "messages": [{"role": "user", "content": "What do you know about me?"}],
  "stream": false
}'
```

The response should reference facts from your import.

Check what's being injected via the Status tab → Memory Preview.

---

## Multi-agent Mesh (optional)

If you run multiple AI agents or tools simultaneously, Velqua Mesh coordinates them:

```bash
# Terminal 1
OLLAMA_HOST=localhost:11435 python agent_one.py

# Terminal 2
OLLAMA_HOST=localhost:11435 python agent_two.py

# Both agents now appear in the Velqua UI → Mesh tab
```

Agents can share notes without knowing about each other. See [README.md#mesh](README.md#mesh) for details.

---

## Troubleshooting

**No facts after import:**
- Check the Review tab — facts with medium quality score (0.4–0.7) wait for approval
- Low-quality facts (<0.4) are discarded. Import a larger conversation file.

**Memory not injecting:**
- Confirm the proxy is running: `curl http://localhost:11435/health`
- Check you're hitting `:11435` not `:11434`
- Check the Status tab for proxy status

**Port 11435 already in use:**
```bash
VELQUA_PROXY_PORT=11436 python backend/server.py
```

**Slow startup:**
- First run downloads sentence-transformer model (~90MB). Subsequent starts are instant.
