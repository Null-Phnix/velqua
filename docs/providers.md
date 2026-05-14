# Provider Configuration

Velqua supports four provider types. Configure them in the Settings tab or via the API.

## Ollama (default)

No API key needed. Runs locally.

**Requirements:** Ollama installed and running at `localhost:11434`

```bash
# Pull a model
ollama pull llama3

# Start Velqua proxy
python backend/server.py

# Chat through proxy
curl http://localhost:11435/api/chat -d '{
  "model": "llama3",
  "messages": [{"role": "user", "content": "hello"}],
  "stream": false
}'
```

The Ollama provider exposes all three Ollama endpoints:
- `POST /api/generate` (prompt-based)
- `POST /api/chat` (message-based)
- `GET /api/tags` (model listing)

## OpenAI

```bash
# In Velqua Settings tab:
# Provider: OpenAI
# API Key: sk-...
# Model: gpt-4o

# Or via API:
curl -X POST http://localhost:8765/settings/provider \
  -H "Content-Type: application/json" \
  -d '{"name": "openai", "api_key": "sk-...", "default_model": "gpt-4o"}'
```

Point your client at Velqua's OpenAI-compatible endpoint:

```
OPENAI_BASE_URL=http://localhost:11435/v1
OPENAI_API_KEY=unused  # Velqua uses its own stored key
```

## Anthropic

```bash
# In Velqua Settings tab:
# Provider: Anthropic
# API Key: sk-ant-...
# Model: claude-sonnet-4-6
```

Point at Velqua's Anthropic-compatible endpoint:

```
ANTHROPIC_BASE_URL=http://localhost:11435
```

The Anthropic provider handles the format difference automatically:
- System prompt is passed as a top-level `system` parameter (not in the messages array)
- Auth uses `x-api-key` header (not `Authorization: Bearer`)

## OpenAI-compatible backends

Works with any backend that implements the OpenAI Chat Completions API:

| Backend | Base URL | Notes |
|---------|----------|-------|
| llama.cpp | `http://localhost:8080` | `--server` flag |
| vLLM | `http://localhost:8000` | |
| LM Studio | `http://localhost:1234` | |
| LocalAI | `http://localhost:8080` | |
| Groq | `https://api.groq.com/openai` | API key required |

```bash
# Add a custom local backend
curl -X POST http://localhost:8765/settings/provider \
  -H "Content-Type: application/json" \
  -d '{
    "name": "local_openai",
    "base_url": "http://localhost:8080",
    "default_model": "mistral"
  }'
```

## Switching providers at runtime

```bash
# List configured providers
curl http://localhost:8765/settings

# Switch active provider
curl -X PUT http://localhost:8765/settings \
  -H "Content-Type: application/json" \
  -d '{"active_provider": "openai"}'
```

No restart required.

## Testing connections

```bash
curl -X POST http://localhost:8765/settings/test-connection \
  -H "Content-Type: application/json" \
  -d '{"name": "openai"}'

# Returns: {"ok": true, "models": ["gpt-4o", "gpt-4o-mini", ...]}
```

## Encrypted key storage

API keys are stored in `data/keys.enc` using Fernet symmetric encryption. The encryption key is derived from your machine's hardware ID using PBKDF2. Keys are never written to logs or environment variables.

To remove a stored key:

```bash
curl -X DELETE http://localhost:8765/settings/provider/openai
```
