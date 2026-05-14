import time
from pathlib import Path

import pytest


@pytest.fixture
def mesh_modules(tmp_path, monkeypatch):
    """
    Import mesh modules against an isolated temp data dir so tests don't touch
    real user state.
    """
    from backend.config import VelquaConfig as Config

    data_dir = tmp_path / "velqua_test_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(Config, "DATA_DIR", data_dir, raising=False)
    monkeypatch.setattr(Config, "LOGS_DIR", data_dir / "logs", raising=False)

    # Import after config patch so module-level singletons initialize in temp dir
    import importlib
    import backend.mesh.registry as registry_module
    import backend.mesh.noteboard as noteboard_module
    import backend.mesh.shared_memory as shared_memory_module

    importlib.reload(registry_module)
    importlib.reload(noteboard_module)
    importlib.reload(shared_memory_module)

    return {
        "registry_module": registry_module,
        "noteboard_module": noteboard_module,
        "shared_memory_module": shared_memory_module,
        "registry": registry_module.registry,
        "noteboard": noteboard_module.noteboard,
        "pool": shared_memory_module.pool,
        "data_dir": data_dir,
    }


def _list_agents(registry):
    agents = registry.list_agents()
    assert isinstance(agents, list)
    return agents


def _find_agent(registry, agent_id):
    return next((a for a in _list_agents(registry) if a.get("id") == agent_id), None)


def test_agent_registration_and_heartbeat(mesh_modules):
    registry = mesh_modules["registry"]

    agent_id = "agent-alpha"
    registry.heartbeat(agent_id, task_hint="first task")
    first = _find_agent(registry, agent_id)

    assert first is not None, "Agent should be registered on first heartbeat"
    assert first["id"] == agent_id
    assert first.get("task_hint") == "first task"
    assert first.get("last_seen")

    first_seen = first["last_seen"]
    time.sleep(0.01)

    registry.heartbeat(agent_id, task_hint="updated task")
    second = _find_agent(registry, agent_id)

    assert second is not None
    assert second["id"] == agent_id
    assert second.get("task_hint") == "updated task"
    assert second["last_seen"] >= first_seen


def test_shared_memory_write_and_cross_agent_read(mesh_modules):
    registry = mesh_modules["registry"]
    pool = mesh_modules["pool"]

    writer = "agent-writer"
    reader = "agent-reader"

    registry.heartbeat(writer, task_hint="writing")
    registry.heartbeat(reader, task_hint="reading")

    key = f"shared-{int(time.time() * 1000)}"
    value = {
        "fact": "The launch window is 9am UTC",
        "source_agent": writer,
        "priority": "high",
    }

    write_result = pool.write(key, value, agent_id=writer)
    assert write_result is not None

    read_result = pool.read(key, agent_id=reader)
    assert read_result is not None
    assert read_result == value


def test_noteboard_delivery_to_specific_agent(mesh_modules):
    registry = mesh_modules["registry"]
    noteboard = mesh_modules["noteboard"]

    sender = "agent-sender"
    target = "agent-target"
    other = "agent-other"

    registry.heartbeat(sender, task_hint="send note")
    registry.heartbeat(target, task_hint="receive note")
    registry.heartbeat(other, task_hint="should not receive")

    note_id = noteboard.post(
        from_agent=sender,
        to_agent=target,
        content="Please summarize the latest memory cluster.",
    )
    assert note_id

    target_notes = noteboard.get_for_agent(target, unread_only=True, limit=10)
    other_notes = noteboard.get_for_agent(other, unread_only=True, limit=10)

    assert len(target_notes) == 1
    assert target_notes[0]["id"] == note_id
    assert target_notes[0]["from_agent"] == sender
    assert target_notes[0]["to_agent"] == target
    assert "summarize" in target_notes[0]["content"].lower()

    assert other_notes == []


def test_broadcast_notes_reach_all_agents(mesh_modules):
    registry = mesh_modules["registry"]
    noteboard = mesh_modules["noteboard"]

    sender = "agent-broadcast"
    recipients = ["agent-a", "agent-b", "agent-c"]

    registry.heartbeat(sender, task_hint="broadcasting")
    for agent_id in recipients:
        registry.heartbeat(agent_id, task_hint="listening")

    note_id = noteboard.post(
        from_agent=sender,
        to_agent="*",
        content="Global update: embedding index rebuilt successfully.",
    )
    assert note_id

    for agent_id in recipients:
        notes = noteboard.get_for_agent(agent_id, unread_only=True, limit=20)
        assert len(notes) >= 1, f"{agent_id} should receive the broadcast"
        assert any(
            note["id"] == note_id and "embedding index rebuilt" in note["content"].lower()
            for note in notes
        ), f"{agent_id} did not receive the expected broadcast note"


def test_stale_agent_cleanup_after_timeout(mesh_modules, monkeypatch):
    registry_module = mesh_modules["registry_module"]
    registry = mesh_modules["registry"]

    stale_id = "agent-stale"
    fresh_id = "agent-fresh"

    registry.heartbeat(stale_id, task_hint="old task")
    registry.heartbeat(fresh_id, task_hint="current task")

    stale = _find_agent(registry, stale_id)
    fresh = _find_agent(registry, fresh_id)

    assert stale is not None and fresh is not None

    # Force one agent to appear stale
    old_timestamp = time.time() - 3600
    stale["last_seen"] = old_timestamp

    if hasattr(registry, "_save"):
        registry._save(_list_agents(registry))
    elif hasattr(registry, "_write"):
        registry._write(_list_agents(registry))
    else:
        # Fallback: update persisted structure through known internal storage attrs
        agents = _list_agents(registry)
        for idx, agent in enumerate(agents):
            if agent.get("id") == stale_id:
                agents[idx] = stale
                break
        if hasattr(registry, "agents"):
            registry.agents = agents

    cleanup_methods = [
        getattr(registry, "cleanup_stale", None),
        getattr(registry, "cleanup_stale_agents", None),
        getattr(registry, "prune_stale", None),
    ]
    cleanup = next((m for m in cleanup_methods if callable(m)), None)

    if cleanup is not None:
        try:
            cleanup(timeout_s=1)
        except TypeError:
            cleanup(1)
    else:
        # Last-resort compatibility path for registry implementations that only
        # remove stale agents during heartbeat/list operations.
        monkeypatch.setattr(registry_module.time, "time", lambda: old_timestamp + 3605)
        registry.heartbeat(fresh_id, task_hint="still current")

    remaining_ids = {a.get("id") for a in _list_agents(registry)}

    assert fresh_id in remaining_ids
    assert stale_id not in remaining_ids
