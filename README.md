<p align="center">
  <img src="https://img.shields.io/badge/tests-645%20passing-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/coverage-85%25-brightgreen" alt="Coverage">
  <img src="https://img.shields.io/badge/version-3.2.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/local--first-privacy--native-purple" alt="Local First">
</p>

# Velqua

**The transparent memory proxy for local LLMs.**

Point any Ollama app at `:11435` instead of `:11434`. That's it. Your AI now remembers who you are across sessions, models, and tools.

### How it works

```
import chat history → extract personal facts → inject into every request
```

```
You: What do you know about me?

Without Velqua:   "I don't have any information about you."
With Velqua:      "You're a software engineer building AI infrastructure.  
                   You prefer Python. You run everything on local hardware."
```

### Why a proxy

Every other memory layer requires integration work. Mem0 needs API calls. Zep needs an SDK. OpenMemory needs Docker, Postgres, and Qdrant.

Velqua is a transparent proxy. Your apps don't change. The port number changes.

| Feature | Velqua | Mem0 | Zep | OpenMemory |
|---|---|---|---|---|
| Zero code changes | ✓ | No | No | No |
| Local-first | ✓ | Cloud | Self-host | Docker stack |
| Import Claude/GPT history | ✓ | No | No | No |
| Multi-agent mesh | ✓ | No | No | No |
| Desktop app | ✓ | No | No | No |

### Quickstart

```bash
pip install velqua
velqua-server
```

Open `http://localhost:8765`. The setup wizard walks you through provider configuration. Then point any Ollama app at `:11435`.

```bash
# Import your history
# Export from Claude (Settings → Export) or ChatGPT (Settings → Data Controls → Export)
# Drag the JSON into the Velqua UI

# Or use the CLI
velqua import ~/Downloads/conversations.json
```

### Pricing

| | Free | Pro | Teams |
|---|---|---|---|
| | $0 | $9/mo | $29/mo |
| Facts stored | 100 | 10,000 | 100,000 |
| Agents | 1 | 5 | 50 |
| Import sources | Claude, ChatGPT | + Obsidian, Notion | + Obsidian, Notion |
| Multi-agent mesh | — | ✓ | ✓ |
| Shared memory | — | — | ✓ |
| Agent dashboard | — | ✓ | ✓ |

### Architecture

```
Any Ollama App (Open WebUI, Continue.dev, etc.)
        │
        ▼
   localhost:11435  ← Velqua proxy (memory injection)
        │
        ▼
   localhost:11434  ← Ollama (your models, unchanged)
```

Local-first. Nothing phones home. API keys encrypted at rest. Binds to `127.0.0.1` by default.

---

Built for people who run their own AI. [velqua.dev](https://velqua.dev)
