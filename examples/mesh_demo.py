"""
mesh_demo.py — Velqua Mesh multi-agent coordination demo.

Simulates two agents (researcher and writer) working through the same proxy.
Shows how agents can:
  1. Be detected and registered automatically
  2. Leave notes for each other via the Mesh API
  3. Have those notes injected into their next LLM request context

Prerequisites:
  - Velqua running: python backend/server.py
  - Ollama running with a model: ollama pull llama3

Usage:
  python examples/mesh_demo.py
"""

import json
import time
import httpx

PROXY_URL = "http://localhost:11435"
API_URL = "http://localhost:8765"
MODEL = "llama3"


def agent_chat(agent_name: str, message: str) -> str:
    """Send a chat through the proxy, declaring agent identity via header."""
    response = httpx.post(
        f"{PROXY_URL}/api/chat",
        headers={"X-Velqua-Agent": agent_name},
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": message}],
            "stream": False,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def post_note(from_agent: str, to_agent: str, content: str, tags: list[str] = None) -> dict:
    """Post a note from one agent to another via the Mesh API."""
    response = httpx.post(
        f"{API_URL}/mesh/notes",
        json={
            "from_agent": from_agent,
            "to_agent": to_agent,
            "content": content,
            "tags": tags or [],
        },
    )
    response.raise_for_status()
    return response.json()


def get_agents() -> list[dict]:
    """List all active agents in the Mesh."""
    response = httpx.get(f"{API_URL}/mesh/agents")
    response.raise_for_status()
    return response.json()["agents"]


def get_mesh_status() -> dict:
    """Get overall Mesh status."""
    response = httpx.get(f"{API_URL}/mesh/status")
    response.raise_for_status()
    return response.json()


def main():
    print("Velqua Mesh Demo — Two Agents, One Proxy")
    print("=" * 50)
    print()

    # Step 1: Researcher agent sends a request — gets registered in Mesh
    print("[researcher] Sending first request (triggers registration)...")
    researcher_reply = agent_chat(
        "researcher",
        "Summarize what you know about transformer architecture in 2 sentences."
    )
    print(f"[researcher] Got reply: {researcher_reply[:100]}...")
    print()

    # Step 2: Check Mesh sees the agent
    agents = get_agents()
    agent_names = [a["agent_id"] for a in agents]
    print(f"Active agents in Mesh: {agent_names}")
    print()

    # Step 3: Researcher leaves a note for the writer
    print("[researcher] Posting note to writer...")
    note = post_note(
        from_agent="researcher",
        to_agent="writer",
        content=(
            "Research phase complete. Key finding: attention mechanisms in transformers "
            "use scaled dot-product to weight value vectors. Please write a blog intro paragraph."
        ),
        tags=["research", "handoff"],
    )
    print(f"[researcher] Note posted (id: {note.get('id', 'ok')})")
    print()

    # Step 4: Writer agent sends its first request
    # Velqua automatically injects the unread note into its system context
    print("[writer] Sending request (Velqua injects researcher's note automatically)...")
    writer_reply = agent_chat(
        "writer",
        "Write a short blog introduction about transformer architecture."
    )
    print(f"[writer] Response (should reference the research finding):")
    print(f"  {writer_reply[:300]}...")
    print()

    # Step 5: Show Mesh status
    status = get_mesh_status()
    print("Mesh status:")
    print(f"  Active agents: {status.get('active_agents', 0)}")
    print(f"  Shared memory entries: {status.get('memory_entries', 0)}")
    print(f"  Notes total: {status.get('notes_total', 0)}")
    print()
    print("Open http://localhost:8765 → Mesh tab to see the live dashboard.")


if __name__ == "__main__":
    main()
