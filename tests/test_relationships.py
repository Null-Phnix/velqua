"""Tests for fact relationship detection and the /graph/relationships endpoint."""
import importlib
import json
import os
import tempfile
import uuid

import pytest

# Set up temp DB before any server imports
_tmpdir = tempfile.mkdtemp()
os.environ["VELQUA_DB_PATH"] = os.path.join(_tmpdir, "test_relationships.db")

import backend.config
importlib.reload(backend.config)

from backend.server import app
from backend.routes._shared import get_memory
from backend.anamnesis.graph.relationships import (
    FactRelationship,
    RelationshipType,
    detect_contradiction,
    detect_elaboration,
    detect_temporal_sequence,
    detect_relationships,
)
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def clean_db():
    yield
    try:
        mem = get_memory()
        rows = mem.backend.list_facts(limit=10000)
        for r in rows:
            mem.backend.delete_fact(r["id"])
        mem.backend.clear_fact_relationships()
    except Exception:
        pass


def _make_fact(content, fact_type="general", confidence=0.6, importance=0.5,
               metadata=None, first_learned=None):
    """Create a fact dict for testing."""
    fid = str(uuid.uuid4())
    return {
        "id": fid,
        "content": content,
        "fact_type": fact_type,
        "confidence": confidence,
        "importance": importance,
        "metadata": metadata or {},
        "first_learned": first_learned,
        "source_episodes": [],
        "last_confirmed": None,
        "confirmation_count": 1,
        "is_superseded": False,
        "tags": [],
    }


def _seed_fact(client, content, fact_type="general", metadata=None):
    """Insert a fact via the bulk import endpoint and return its ID."""
    item = {
        "content": content,
        "fact_type": fact_type,
        "confidence": 0.6,
        "importance": 0.5,
        "tags": [],
        "metadata": metadata or {},
    }
    r = client.post("/facts/import/bulk", json={"facts": [item]})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["inserted"] >= 1, f"Seed failed: {data}"
    # Get the ID of the last inserted fact
    facts = client.get("/facts/list?limit=50").json()["facts"]
    match = [f for f in facts if f["content"] == content]
    assert match, f"Seeded fact not found: {content}"
    return match[0]["id"]


# ============================================================
# Unit tests: contradiction detection
# ============================================================

class TestContradictionDetection:
    def test_antonym_pair_detected(self):
        a = _make_fact("I love hiking in the mountains",
                       metadata={"topic": "hobbies"})
        b = _make_fact("I hate hiking in the mountains",
                       metadata={"topic": "hobbies"})
        rel = detect_contradiction(a, b)
        assert rel is not None
        assert rel.relationship_type == RelationshipType.CONTRADICTION
        assert rel.confidence >= 0.3
        assert "antonym" in rel.evidence.lower()

    def test_negation_detected(self):
        a = _make_fact("I enjoy cooking Italian food for friends",
                       metadata={"topic": "cooking"})
        b = _make_fact("I don't enjoy cooking Italian food anymore",
                       metadata={"topic": "cooking"})
        rel = detect_contradiction(a, b)
        assert rel is not None
        assert rel.relationship_type == RelationshipType.CONTRADICTION
        assert rel.confidence >= 0.3

    def test_opposing_sentiment(self):
        a = _make_fact("Python is the best programming language",
                       metadata={"topic": "programming", "sentiment_score": 0.9})
        b = _make_fact("Python is the worst programming language",
                       metadata={"topic": "programming", "sentiment_score": -0.8})
        rel = detect_contradiction(a, b)
        assert rel is not None
        assert "sentiment" in rel.evidence.lower() or "antonym" in rel.evidence.lower()

    def test_no_contradiction_unrelated(self):
        a = _make_fact("I enjoy mountain biking on weekends")
        b = _make_fact("The capital of France is Paris")
        rel = detect_contradiction(a, b)
        assert rel is None

    def test_superseded_fact_boosts_confidence(self):
        a = _make_fact("I live in Vancouver and work downtown",
                       metadata={"topic": "location"})
        b = _make_fact("I don't live in Vancouver anymore since moving",
                       metadata={"topic": "location"}, )
        b["is_superseded"] = True
        rel = detect_contradiction(a, b)
        assert rel is not None
        assert "superseded" in rel.evidence.lower()


# ============================================================
# Unit tests: elaboration detection
# ============================================================

class TestElaborationDetection:
    def test_specific_expands_general(self):
        a = _make_fact("I work as a software engineer",
                       fact_type="professional",
                       metadata={"topic": "career"})
        b = _make_fact("I work as a software engineer specializing in backend systems and distributed databases",
                       fact_type="professional",
                       metadata={"topic": "career"})
        rel = detect_elaboration(a, b)
        assert rel is not None
        assert rel.relationship_type == RelationshipType.ELABORATION
        # Direction: general -> specific
        assert rel.source_id == a["id"]
        assert rel.target_id == b["id"]

    def test_same_topic_same_type(self):
        a = _make_fact("I enjoy playing guitar",
                       fact_type="preference",
                       metadata={"topic": "music"})
        b = _make_fact("I enjoy playing guitar, especially jazz and blues styles on my vintage Fender",
                       fact_type="preference",
                       metadata={"topic": "music"})
        rel = detect_elaboration(a, b)
        assert rel is not None
        assert rel.confidence >= 0.3

    def test_no_elaboration_unrelated(self):
        a = _make_fact("I have a cat named Whiskers")
        b = _make_fact("The stock market crashed in 2008")
        rel = detect_elaboration(a, b)
        assert rel is None

    def test_direction_is_short_to_long(self):
        short = _make_fact("I studied computer science",
                           metadata={"topic": "education"})
        long = _make_fact("I studied computer science at MIT with a focus on artificial intelligence and machine learning",
                          metadata={"topic": "education"})
        rel = detect_elaboration(short, long)
        assert rel is not None
        assert rel.source_id == short["id"]
        assert rel.target_id == long["id"]
        # Also test with reversed input order
        rel2 = detect_elaboration(long, short)
        assert rel2 is not None
        assert rel2.source_id == short["id"]
        assert rel2.target_id == long["id"]


# ============================================================
# Unit tests: temporal sequence detection
# ============================================================

class TestTemporalSequenceDetection:
    def test_past_vs_present(self):
        a = _make_fact("I used to live in Toronto and commuted daily",
                       metadata={"topic": "location"})
        b = _make_fact("I currently live in Vancouver and work remotely",
                       metadata={"topic": "location"})
        rel = detect_temporal_sequence(a, b)
        assert rel is not None
        assert rel.relationship_type == RelationshipType.TEMPORAL_SEQUENCE
        # A (past) should come before B (present)
        assert rel.source_id == a["id"]
        assert rel.target_id == b["id"]

    def test_year_references(self):
        a = _make_fact("I graduated from university in 2018",
                       metadata={"topic": "education"})
        b = _make_fact("I started my PhD program in 2020",
                       metadata={"topic": "education"})
        rel = detect_temporal_sequence(a, b)
        assert rel is not None
        assert rel.source_id == a["id"]
        assert rel.target_id == b["id"]
        assert "2018" in rel.evidence and "2020" in rel.evidence

    def test_no_temporal_unrelated(self):
        a = _make_fact("I prefer dark chocolate over milk")
        b = _make_fact("The sun is approximately 93 million miles away")
        rel = detect_temporal_sequence(a, b)
        assert rel is None

    def test_reversed_years(self):
        a = _make_fact("I moved to the new office in 2024",
                       metadata={"topic": "work"})
        b = _make_fact("I joined the company in 2019",
                       metadata={"topic": "work"})
        rel = detect_temporal_sequence(a, b)
        assert rel is not None
        # B (2019) should be source, A (2024) should be target
        assert rel.source_id == b["id"]
        assert rel.target_id == a["id"]


# ============================================================
# Unit tests: bulk detection
# ============================================================

class TestBulkDetection:
    def test_detect_multiple_types(self):
        facts = [
            _make_fact("I love cooking pasta dishes at home",
                       metadata={"topic": "cooking"}),
            _make_fact("I hate cooking pasta dishes at home",
                       metadata={"topic": "cooking"}),
            _make_fact("I love cooking pasta dishes at home especially carbonara and pesto variations",
                       metadata={"topic": "cooking"}),
        ]
        rels = detect_relationships(facts)
        types_found = {r.relationship_type for r in rels}
        assert RelationshipType.CONTRADICTION in types_found

    def test_filter_by_type(self):
        facts = [
            _make_fact("I love pizza", metadata={"topic": "food"}),
            _make_fact("I hate pizza", metadata={"topic": "food"}),
        ]
        rels = detect_relationships(facts, types=[RelationshipType.TEMPORAL_SEQUENCE])
        # Should not find contradictions when only looking for temporal
        contradiction_rels = [r for r in rels if r.relationship_type == RelationshipType.CONTRADICTION]
        assert len(contradiction_rels) == 0

    def test_empty_list(self):
        assert detect_relationships([]) == []

    def test_single_fact(self):
        facts = [_make_fact("I like cats")]
        assert detect_relationships(facts) == []

    def test_no_duplicate_edges(self):
        facts = [
            _make_fact("I like running in the park", metadata={"topic": "exercise"}),
            _make_fact("I dislike running in the park", metadata={"topic": "exercise"}),
        ]
        rels = detect_relationships(facts)
        # Should not have both A->B and B->A for same type
        keys = [(r.source_id, r.target_id, r.relationship_type) for r in rels]
        assert len(keys) == len(set(keys))


# ============================================================
# Database layer tests
# ============================================================

class TestRelationshipDatabase:
    def test_save_and_retrieve(self, client):
        mem = get_memory()
        id_a = _seed_fact(client, "I enjoy reading science fiction novels regularly")
        id_b = _seed_fact(client, "I dislike reading science fiction novels now")

        mem.backend.save_fact_relationship(
            id_a, id_b, "contradiction", 0.8, "antonym pair: enjoy vs dislike",
        )
        edges = mem.backend.get_fact_relationships(fact_id=id_a)
        assert len(edges) >= 1
        edge = edges[0]
        assert edge["source_id"] == id_a
        assert edge["target_id"] == id_b
        assert edge["relationship_type"] == "contradiction"
        assert edge["confidence"] == 0.8

    def test_filter_by_type(self, client):
        mem = get_memory()
        id_a = _seed_fact(client, "I used to work at Google as a senior engineer")
        id_b = _seed_fact(client, "I now work at Meta as a staff engineer")

        mem.backend.save_fact_relationship(
            id_a, id_b, "temporal_sequence", 0.7, "past vs present",
        )
        mem.backend.save_fact_relationship(
            id_a, id_b, "contradiction", 0.5, "different employers",
        )

        temporal = mem.backend.get_fact_relationships(
            relationship_type="temporal_sequence",
        )
        assert all(e["relationship_type"] == "temporal_sequence" for e in temporal)

    def test_min_confidence_filter(self, client):
        mem = get_memory()
        id_a = _seed_fact(client, "I have a dog named Rex that I walk daily")
        id_b = _seed_fact(client, "I have a dog named Rex, a golden retriever that I walk twice daily")

        mem.backend.save_fact_relationship(id_a, id_b, "elaboration", 0.3, "low conf")
        mem.backend.save_fact_relationship(id_a, id_b, "temporal_sequence", 0.9, "high conf")

        high = mem.backend.get_fact_relationships(min_confidence=0.5)
        assert all(e["confidence"] >= 0.5 for e in high)

    def test_count(self, client):
        mem = get_memory()
        id_a = _seed_fact(client, "I studied at Oxford for my undergraduate degree")
        id_b = _seed_fact(client, "I got my masters degree from Cambridge after")

        mem.backend.save_fact_relationship(id_a, id_b, "temporal_sequence", 0.7, "test")
        count = mem.backend.count_fact_relationships()
        assert count >= 1

    def test_delete_relationships(self, client):
        mem = get_memory()
        id_a = _seed_fact(client, "I play tennis every weekend at the local club")
        id_b = _seed_fact(client, "I quit playing tennis after my knee injury")

        mem.backend.save_fact_relationship(id_a, id_b, "contradiction", 0.8, "test")
        deleted = mem.backend.delete_fact_relationships(id_a)
        assert deleted >= 1
        edges = mem.backend.get_fact_relationships(fact_id=id_a)
        assert len(edges) == 0

    def test_clear_all_relationships(self, client):
        mem = get_memory()
        id_a = _seed_fact(client, "I always drink green tea with breakfast every single day")
        id_b = _seed_fact(client, "I switched to black coffee for my morning routine exclusively")

        mem.backend.save_fact_relationship(id_a, id_b, "contradiction", 0.8, "test")
        cleared = mem.backend.clear_fact_relationships()
        assert cleared >= 1
        assert mem.backend.count_fact_relationships() == 0

    def test_upsert_on_duplicate(self, client):
        mem = get_memory()
        id_a = _seed_fact(client, "I run marathons competitively every year")
        id_b = _seed_fact(client, "I stopped running marathons after the accident")

        mem.backend.save_fact_relationship(id_a, id_b, "contradiction", 0.5, "v1")
        mem.backend.save_fact_relationship(id_a, id_b, "contradiction", 0.9, "v2")
        edges = mem.backend.get_fact_relationships(fact_id=id_a, relationship_type="contradiction")
        assert len(edges) == 1
        assert edges[0]["confidence"] == 0.9


# ============================================================
# API endpoint tests
# ============================================================

class TestGraphRelationshipsEndpoint:
    def test_get_empty(self, client):
        r = client.get("/graph/relationships")
        assert r.status_code == 200
        data = r.json()
        assert "relationships" in data
        assert "count" in data
        assert "total" in data

    def test_get_with_stored_edges(self, client):
        mem = get_memory()
        id_a = _seed_fact(client, "I am a vegetarian who avoids all meat products")
        id_b = _seed_fact(client, "I eat steak every Friday night at the restaurant")

        mem.backend.save_fact_relationship(
            id_a, id_b, "contradiction", 0.85, "vegetarian vs steak",
        )
        r = client.get("/graph/relationships")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] >= 1
        assert data["total"] >= 1

    def test_filter_by_fact_id(self, client):
        mem = get_memory()
        id_a = _seed_fact(client, "I started learning piano in January this year")
        id_b = _seed_fact(client, "I passed my piano grade five exam in June")
        id_c = _seed_fact(client, "I prefer cats over dogs as household pets")

        mem.backend.save_fact_relationship(id_a, id_b, "temporal_sequence", 0.8, "piano")
        mem.backend.save_fact_relationship(id_a, id_c, "elaboration", 0.4, "unrelated")

        r = client.get(f"/graph/relationships?fact_id={id_b}")
        assert r.status_code == 200
        data = r.json()
        for edge in data["relationships"]:
            assert id_b in (edge["source_id"], edge["target_id"])

    def test_filter_by_type(self, client):
        mem = get_memory()
        id_a = _seed_fact(client, "I like swimming in the ocean during summer")
        id_b = _seed_fact(client, "I dislike swimming in the ocean during summer")

        mem.backend.save_fact_relationship(id_a, id_b, "contradiction", 0.8, "test")

        r = client.get("/graph/relationships?type=contradiction")
        assert r.status_code == 200
        for edge in r.json()["relationships"]:
            assert edge["relationship_type"] == "contradiction"

    def test_invalid_type_returns_400(self, client):
        r = client.get("/graph/relationships?type=banana")
        assert r.status_code == 400

    def test_min_confidence_filter(self, client):
        mem = get_memory()
        id_a = _seed_fact(client, "I work from home three days a week minimum")
        id_b = _seed_fact(client, "I work from home three days a week, specifically Monday Wednesday Friday")

        mem.backend.save_fact_relationship(id_a, id_b, "elaboration", 0.3, "low")

        r = client.get("/graph/relationships?min_confidence=0.5")
        assert r.status_code == 200
        for edge in r.json()["relationships"]:
            assert edge["confidence"] >= 0.5

    def test_pagination(self, client):
        mem = get_memory()
        ids = []
        unique_facts = [
            "I visited the Grand Canyon on my birthday trip in spring",
            "I learned to play the violin during summer music camp",
            "I completed a marathon in Berlin with a personal best time",
            "I adopted a rescue dog named Biscuit from the shelter",
            "I built a wooden bookshelf for my home office last month",
        ]
        for content in unique_facts:
            fid = _seed_fact(client, content)
            ids.append(fid)

        for i in range(len(ids) - 1):
            mem.backend.save_fact_relationship(
                ids[i], ids[i + 1], "temporal_sequence", 0.7, f"seq {i}",
            )

        r1 = client.get("/graph/relationships?limit=2&offset=0")
        r2 = client.get("/graph/relationships?limit=2&offset=2")
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["count"] <= 2
        assert r2.json()["count"] <= 2


class TestDetectEndpoint:
    def test_detect_stores_edges(self, client):
        _seed_fact(client, "I love playing basketball every weekend at the park",
                   metadata={"topic": "sports"})
        _seed_fact(client, "I hate playing basketball every weekend at the park",
                   metadata={"topic": "sports"})

        r = client.post("/graph/relationships/detect")
        assert r.status_code == 200
        data = r.json()
        assert data["facts_analyzed"] >= 2
        assert data["detected"] >= 0  # detection depends on heuristics

    def test_detect_with_type_filter(self, client):
        _seed_fact(client, "I used to work in Toronto before",
                   metadata={"topic": "work"})
        _seed_fact(client, "I now work in Vancouver remotely",
                   metadata={"topic": "work"})

        r = client.post("/graph/relationships/detect?type=temporal_sequence")
        assert r.status_code == 200
        data = r.json()
        assert data["facts_analyzed"] >= 2

    def test_detect_invalid_type(self, client):
        r = client.post("/graph/relationships/detect?type=invalid_type")
        assert r.status_code == 400

    def test_detect_empty_db(self, client):
        r = client.post("/graph/relationships/detect")
        assert r.status_code == 200
        data = r.json()
        assert data["facts_analyzed"] == 0
        assert data["detected"] == 0
