"""
basic_memory.py — Velqua memory injection via the proxy.

This script shows how Velqua works from the perspective of a client app.
No Velqua-specific code required — just point at the proxy port.

Prerequisites:
  - Velqua running: python backend/server.py
  - Ollama running with a model pulled: ollama pull llama3
  - Some facts imported in the Velqua UI

Usage:
  python examples/basic_memory.py
"""

import json
import httpx

PROXY_URL = "http://localhost:11435"
MODEL = "llama3"


def chat(messages: list[dict]) -> str:
    """Send a chat request through the Velqua proxy."""
    response = httpx.post(
        f"{PROXY_URL}/api/chat",
        json={"model": MODEL, "messages": messages, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def main():
    print("Velqua Basic Memory Demo")
    print("=" * 40)
    print(f"Proxy: {PROXY_URL}")
    print(f"Model: {MODEL}")
    print()

    # Check health
    health = httpx.get(f"{PROXY_URL}/health").json()
    print(f"Facts in memory: {health.get('facts_count', 0)}")
    print(f"Proxy status: {health.get('status', 'unknown')}")
    print()

    if health.get("facts_count", 0) == 0:
        print("No facts found. Import some in the Velqua UI first.")
        print("  http://localhost:8765 → Import tab")
        return

    # Ask the model about itself and the user
    messages = [{"role": "user", "content": "What do you know about me?"}]
    print(f"User: {messages[0]['content']}")
    print()

    reply = chat(messages)
    print(f"Assistant: {reply}")
    print()

    # Follow-up to show memory persists through multi-turn
    messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": "Based on what you know, what kind of projects would suit me?"})
    print(f"User: {messages[-1]['content']}")
    print()

    reply2 = chat(messages)
    print(f"Assistant: {reply2}")


if __name__ == "__main__":
    main()
