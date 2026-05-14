"""
E2E smoke test for Velqua server — exercises the critical request path
without needing a running server process.

Usage: python tests/smoke_test.py
"""
import io
import json
import os
import sys
import tempfile

# Use a temp data directory so we don't touch the real database
_tmp = tempfile.mkdtemp(prefix="velqua_smoke_")
os.environ["VELQUA_DATA_DIR"] = _tmp
os.environ.setdefault("VELQUA_DB_PATH", os.path.join(_tmp, "smoke.db"))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from backend.server import app  # noqa: E402

client = TestClient(app)
passed = 0
failed = 0


def check(name: str, ok: bool, detail: str = ""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}: {detail}")


# ── 1. Health ──────────────────────────────────────────────────────
print("\n1. Health check")
r = client.get("/health")
check("GET /health returns 200", r.status_code == 200, f"got {r.status_code}")
data = r.json()
check("/health has status=ok", data.get("status") == "ok", f"got {data}")

# ── 2. Facts list (empty) ─────────────────────────────────────────
print("\n2. Facts list (empty database)")
r = client.get("/facts/list")
check("GET /facts/list returns 200", r.status_code == 200, f"got {r.status_code}")
data = r.json()
check("facts list has 'facts' key", "facts" in data, f"keys: {list(data.keys())}")
check("facts list is empty initially", data.get("total", -1) == 0, f"total={data.get('total')}")

# ── 3. Import a fact via JSON upload ──────────────────────────────
print("\n3. Create a fact via JSON import")
facts_payload = json.dumps({
    "facts": [
        {"content": "The user prefers dark mode for all applications", "confidence": 0.95},
        {"content": "The user lives in Saskatchewan, Canada", "confidence": 0.9},
    ]
})
r = client.post(
    "/import/facts-json",
    files={"file": ("facts.json", io.BytesIO(facts_payload.encode()), "application/json")},
)
check("POST /import/facts-json returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:200]}")
data = r.json()
check("imported 2 facts", data.get("facts_stored", 0) == 2, f"got {data}")

# ── 4. Verify facts persist ──────────────────────────────────────
print("\n4. Verify facts persist")
r = client.get("/facts/list")
data = r.json()
check("facts count is now 2", data.get("total") == 2, f"total={data.get('total')}")

# ── 5. Search for a fact ──────────────────────────────────────────
print("\n5. Search for a fact")
r = client.get("/facts/search", params={"q": "dark mode"})
check("GET /facts/search returns 200", r.status_code == 200, f"got {r.status_code}")
data = r.json()
results = data.get("results", [])
check("search found results", len(results) > 0, f"got {len(results)} results")
if results:
    check(
        "search result contains 'dark mode'",
        "dark mode" in results[0].get("content", "").lower(),
        f"got: {results[0].get('content', '')[:80]}",
    )

# ── 6. Delete a fact ─────────────────────────────────────────────
print("\n6. Delete a fact")
r = client.get("/facts/list")
facts = r.json().get("facts", [])
if facts:
    fact_id = facts[0]["id"]
    r = client.delete(f"/facts/{fact_id}")
    check(f"DELETE /facts/{fact_id} returns 200", r.status_code == 200, f"got {r.status_code}")
    # Verify count decreased
    r = client.get("/facts/list")
    new_total = r.json().get("total", -1)
    check("fact count decreased to 1", new_total == 1, f"total={new_total}")
else:
    check("had facts to delete", False, "no facts found")

# ── 7. Settings / providers ──────────────────────────────────────
print("\n7. Settings and providers")
r = client.get("/settings")
check("GET /settings returns 200", r.status_code == 200, f"got {r.status_code}")
data = r.json()
check("settings has memory config", "budget" in data, f"keys: {list(data.keys())}")

r = client.get("/settings/providers")
check("GET /settings/providers returns 200", r.status_code == 200, f"got {r.status_code}")

# ── 8. License ────────────────────────────────────────────────────
print("\n8. License status")
r = client.get("/license/status")
check("GET /license/status returns 200", r.status_code == 200, f"got {r.status_code}")
data = r.json()
check("license status is trial", data.get("status") == "trial", f"got {data.get('status')}")

# ── 9. Backup and restore cycle ──────────────────────────────────
print("\n9. Backup and restore")
r = client.get("/export/facts")
check("GET /export/facts returns 200", r.status_code == 200, f"got {r.status_code}")
backup_data = r.content
check("backup is non-empty", len(backup_data) > 50, f"got {len(backup_data)} bytes")

# ── 10. Fact stats ────────────────────────────────────────────────
print("\n10. Fact stats")
r = client.get("/facts/stats")
check("GET /facts/stats returns 200", r.status_code == 200, f"got {r.status_code}")
data = r.json()
check("stats has total", "total" in data, f"keys: {list(data.keys())}")

# ── Summary ───────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"SMOKE TEST: {passed} passed, {failed} failed")
print(f"{'='*50}")

sys.exit(1 if failed else 0)
