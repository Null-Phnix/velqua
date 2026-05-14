# Velqua — Landing Page Copy

---

## Hero

**Your AI finally knows who you are.**

Velqua gives any local LLM persistent personal memory — with no code changes in your apps. Change one port number. That's it.

[Get Velqua — Free] [See pricing]

---

## The problem (above the fold)

Every time you open a new chat, your AI starts from zero.

It doesn't know your name. Your job. Your preferred frameworks. The project you've been building for six months. You repeat yourself constantly — to every model, in every tool, across every session.

Mem0, Zep, and other memory libraries fix this — if you're willing to rewrite your application to call their APIs. Most people aren't. Most people just want their AI to remember them.

---

## How Velqua is different

Every other memory solution requires integration work.

**Velqua doesn't.** It's a transparent proxy. Your apps don't change. The port number changes.

```
Before: your tool → localhost:11434 (Ollama)
After:  your tool → localhost:11435 (Velqua) → localhost:11434
```

Open WebUI, Continue.dev, AnythingLLM, your custom scripts — they all work unchanged. Velqua intercepts the request, injects your personal context, forwards to your LLM, and extracts new facts from the response. Invisible. Automatic.

---

## What it remembers

Import your conversation history from Claude or ChatGPT. Velqua reads through your messages, extracts what matters — who you are, what you do, what you prefer — and discards the noise.

From 1,000 conversations, it typically extracts 50–200 high-quality facts. Personal details. Professional context. Recurring preferences. Things worth knowing.

After that, it learns in real-time. Every conversation through the proxy is scanned for new facts. Quality scored. Deduplicated against what's already stored. Kept if it's worth keeping.

---

## Features

### Memory that sticks

- Import from Claude or ChatGPT — one drag-and-drop
- Real-time fact learning from every conversation
- Quality scoring: high facts auto-store, medium go to review, low discard
- Deduplication: same fact from different conversations merges cleanly
- Contradiction detection: conflicting facts flagged for resolution
- Freshness decay: recent facts rank higher than stale ones

### Works with everything

- Ollama (local, free)
- OpenAI, Anthropic, Groq (cloud)
- Any OpenAI-compatible backend: llama.cpp, vLLM, LM Studio, LocalAI
- API keys encrypted at rest, never leave your machine

### Mesh — for power users

Running multiple AI agents? Velqua Mesh coordinates them without any of them knowing the others exist.

Agents see each other on a live dashboard. They can leave notes for each other that get injected transparently. A research agent finishes work, leaves a handoff note — the writer agent receives it on its next request, automatically.

No frameworks. No message queues. Just a port.

### Native desktop app

Not just a server process — a native desktop window (pywebview). Manage your memory, review facts, check proxy status, watch the Mesh dashboard. Installs as a proper application on Windows, macOS, and Linux.

---

## Comparison

| | Velqua | Mem0 | Zep | OpenMemory |
|-|--------|------|-----|-----------|
| Zero code changes | **Yes** | No | No | No |
| Local-first | **Yes** | No (cloud) | Optional | Yes (Docker stack) |
| Import Claude/ChatGPT | **Yes** | No | No | No |
| Multi-agent mesh | **Yes** | No | No | No |
| Desktop app | **Yes** | No | No | No |
| Setup | Port change | API integration | SDK + graph DB | Docker + Postgres + Qdrant |

---

## Who it's for

**Local LLM users** running Ollama, LM Studio, or similar — who want their AI to know them without rebuilding their workflow.

**Power users** running multiple agents or tools simultaneously — who want coordination without infrastructure.

**Developers** who don't want to think about memory plumbing — they just want it to work.

---

## Pricing

**$49 one-time.** No subscription. No per-request fees. No cloud dependency.

The memory never leaves your machine. You pay once and it's yours.

[Get Velqua — $49]

*30-day offline tolerance. Data never deleted on license issues.*

---

## Quick start (from homepage)

```bash
pip install velqua
python -m velqua

# Change one setting in your LLM tool:
# Before: localhost:11434
# After:  localhost:11435
```

[Full quick start guide →]

---

## Footer tagline options

- "Memory for your local AI. No cloud. No code changes. No compromise."
- "The proxy that gives your AI a long-term memory."
- "One port change. Permanent memory."

---

## SEO / meta

**Title:** Velqua — Persistent Memory for Local LLMs

**Description:** Give any Ollama-compatible LLM persistent personal memory with no code changes. Import your Claude or ChatGPT history, auto-learn from conversations, coordinate multiple agents. $49 one-time.

**Keywords:** local LLM memory, Ollama memory, persistent AI memory, LLM proxy, personal AI memory, ChatGPT memory, Claude memory, Open WebUI memory

---

## Objection handling

**"I can just paste context manually."**
You do this every session, for every model, for every tool. Velqua does it automatically, with only the relevant facts (not your entire history), scoped to the current conversation topic.

**"Won't this slow down my requests?"**
Memory retrieval adds ~10–50ms. LLM inference takes seconds. The overhead is imperceptible.

**"Is my data safe?"**
Everything stays local. API keys are encrypted at rest. The proxy binds to `127.0.0.1` by default. Nothing phones home.

**"What if I change models or tools?"**
Velqua is model-agnostic and app-agnostic. Switch from llama3 to mistral to GPT-4 — same memory. Switch from Open WebUI to Continue.dev — same memory.

---

## Pricing

| | Free | Pro | Teams |
|---|---|---|---|
| Price | $0 | $9/mo | $29/mo |
| Facts stored | 100 | 10,000 | 100,000 |
| Agents | 1 | 5 | 50 |
| Import formats | Claude, ChatGPT | + Obsidian, Notion | + Obsidian, Notion |
| Multi-agent mesh | — | ✓ | ✓ |
| Shared memory pools | — | — | ✓ |
| Agent dashboard | — | ✓ | ✓ |

Free tier includes everything you need for personal use. Pro adds multi-agent mesh and larger memory. Teams adds shared pools for coordinating agents across machines. All tiers are local-first — your data never leaves your machine.
