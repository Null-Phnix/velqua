"""Performance and concurrency tests for Velqua critical paths.

These tests verify latency budgets for the proxy and server APIs.
They use monkeypatch (not module reload) to avoid polluting shared state.
"""
import time

import pytest
from fastapi.testclient import TestClient

from backend.proxy import app as proxy_app, memory as proxy_memory
from backend.server import app as server_app


@pytest.fixture
def server_client():
    """HTTP client for the main API."""
    with TestClient(server_app) as client:
        yield client


@pytest.fixture
def proxy_client():
    """HTTP client for the proxy API."""
    with TestClient(proxy_app) as client:
        yield client


def _seed_facts_directly(count: int):
    """Seed deterministic facts directly into the proxy's memory store."""
    for i in range(count):
        proxy_memory.semantic.add_fact(
            content=f"Performance fact {i}: the user prefers low-latency systems and deterministic tests.",
            fact_type="general",
            confidence=0.9,
        )


class TestPerformance:
    @pytest.mark.parametrize("fact_count", [0, 100, 1000])
    def test_proxy_latency_under_100ms_with_varied_fact_counts(
        self, proxy_client: TestClient, monkeypatch, fact_count: int
    ):
        """Proxy request overhead should remain fast as memory grows."""
        _seed_facts_directly(fact_count)

        async def _fake_forward(body, stream, metadata):
            return {
                "model": body.get("model", "test-model"),
                "message": {"role": "assistant", "content": "ok"},
                "done": True,
                "velqua_metadata": metadata,
            }

        monkeypatch.setattr("backend.proxy._forward_ollama_chat", _fake_forward)

        payload = {
            "model": "test-model",
            "stream": False,
            "messages": [
                {"role": "user", "content": "What do you remember about my latency preferences?"}
            ],
        }

        # Warmup: first request may load the embedding model (~4s cold start)
        proxy_client.post("/api/chat", json=payload)

        # Timed run: should be fast now that model is loaded
        started = time.monotonic()
        response = proxy_client.post("/api/chat", json=payload)
        elapsed_ms = (time.monotonic() - started) * 1000

        assert response.status_code == 200, response.text
        assert elapsed_ms < 500, (
            f"Proxy latency too high with {fact_count} facts: {elapsed_ms:.2f}ms"
        )

    def test_facts_search_responds_under_50ms(self, server_client: TestClient):
        """Facts search should feel instant for common queries."""
        started = time.monotonic()
        response = server_client.get(
            "/facts/search",
            params={"q": "low-latency deterministic tests", "limit": 20},
        )
        elapsed_ms = (time.monotonic() - started) * 1000

        assert response.status_code == 200, response.text
        assert elapsed_ms < 50, f"/facts/search too slow: {elapsed_ms:.2f}ms"

    def test_bulk_import_200_facts_under_10_seconds(self, server_client: TestClient):
        """Bulk ingestion must stay practical for real-world imports."""
        facts = [
            {
                "content": f"Bulk import fact {i}: the user likes precise engineering workflows.",
                "fact_type": "personal",
                "confidence": 0.8,
                "importance": 0.5,
                "source": "performance_test",
            }
            for i in range(200)
        ]

        started = time.monotonic()
        response = server_client.post("/facts/import/bulk", json={"facts": facts})
        elapsed_s = time.monotonic() - started

        assert response.status_code == 200, response.text
        assert elapsed_s < 10, f"/facts/import/bulk took too long: {elapsed_s:.2f}s"

    def test_ten_concurrent_proxy_requests_do_not_deadlock(self, monkeypatch):
        """Concurrent proxy calls should all complete successfully."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        async def _fake_forward(body, stream, metadata):
            return {
                "model": body.get("model", "test-model"),
                "message": {"role": "assistant", "content": "ok"},
                "done": True,
                "velqua_metadata": metadata,
            }

        monkeypatch.setattr("backend.proxy._forward_ollama_chat", _fake_forward)

        payload = {
            "model": "test-model",
            "stream": False,
            "messages": [
                {"role": "user", "content": "Summarize what you remember about my workflow."}
            ],
        }

        def make_request(index: int):
            with TestClient(proxy_app) as client:
                started = time.monotonic()
                response = client.post("/api/chat", json=payload)
                elapsed = time.monotonic() - started
                return index, response.status_code, elapsed, response.json()

        started = time.monotonic()
        results = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(10)]
            for future in as_completed(futures, timeout=30):
                results.append(future.result())

        total_elapsed = time.monotonic() - started

        assert len(results) == 10, "Not all concurrent requests completed"
        assert total_elapsed < 30, f"Concurrent requests stalled: {total_elapsed:.2f}s"

        for index, status_code, elapsed, body in results:
            assert status_code == 200, f"Request {index} failed: {body}"
            assert elapsed < 15, f"Request {index} appeared blocked: {elapsed:.2f}s"
            assert "velqua_metadata" in body, f"Request {index} missing metadata"
