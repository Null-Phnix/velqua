"""
Velqua Mesh — Local multi-agent coordination layer.

Extends Velqua from a single-app memory proxy into a shared coordination
layer. Multiple AI agents connect through the same proxy port; Mesh tracks
their identities, shares memory between them, and provides a noteboard for
structured inter-agent communication.

All coordination is transparent — agents don't need to know Mesh exists.
"""
from backend.mesh.registry import AgentRegistry
from backend.mesh.shared_memory import SharedMemoryPool
from backend.mesh.noteboard import Noteboard

__all__ = ["AgentRegistry", "SharedMemoryPool", "Noteboard"]
