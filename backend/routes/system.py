"""
System routes: health check, contradiction detection, import history.
"""
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.config import VelquaConfig as Config
from backend.logging_config import get_logger
from backend.routes._shared import get_memory, import_history

logger = get_logger("routes.system")

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    facts_count: int
    episodes_count: int
    database_path: str
    database_size_mb: float


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Quick health probe — returns fact/episode counts and DB size."""
    memory = get_memory()
    db_size = Config.DB_PATH.stat().st_size / (1024 * 1024) if Config.DB_PATH.exists() else 0

    return HealthResponse(
        status="ok",
        facts_count=memory.semantic.count(),
        episodes_count=len(memory.episodic.list_all()),
        database_path=str(Config.DB_PATH),
        database_size_mb=round(db_size, 2),
    )


# --- Contradiction Detection ---

@router.get("/facts/contradictions")
async def find_contradictions():
    """
    Scan stored facts for potential contradictions.

    Compares each fact against the rest using the Anamnesis contradiction
    detector. Limited to 100 facts to avoid O(n^2) explosion on large stores.
    """
    memory = get_memory()
    try:
        from anamnesis.consolidation.contradiction import detect_contradictions

        all_facts = memory.semantic.list_all(limit=Config.MAX_FACTS_LIST)
        contradictions = []
        checked = set()

        for fact in all_facts[:Config.CONTRADICTION_CHECK_LIMIT]:
            results = detect_contradictions(fact, all_facts, threshold=0.5)
            for c in results:
                if c.is_contradiction and c.existing_fact:
                    # Deduplicate: (A vs B) and (B vs A) are the same contradiction
                    pair_key = tuple(sorted([fact.id, c.existing_fact.id]))
                    if pair_key not in checked:
                        checked.add(pair_key)
                        contradictions.append({
                            "fact_a": {"id": fact.id, "content": fact.content},
                            "fact_b": {"id": c.existing_fact.id, "content": c.existing_fact.content},
                            "type": c.contradiction_type,
                            "confidence": c.confidence,
                            "explanation": c.explanation,
                        })

        return {"contradictions": contradictions, "count": len(contradictions)}

    except ImportError:
        return {"contradictions": [], "count": 0, "error": "Contradiction module not available"}
    except Exception as e:
        logger.error("Contradiction scan failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Contradiction scan failed")


@router.post("/facts/{fact_id}/supersede")
async def supersede_fact(fact_id: str):
    """Mark a fact as superseded (outdated, replaced by a newer fact)."""
    memory = get_memory()
    try:
        fact = memory.semantic.get(fact_id)
        if not fact:
            raise HTTPException(status_code=404, detail="Fact not found")

        fact.is_superseded = True
        memory.semantic.save(fact)
        return {"success": True, "fact_id": fact_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to supersede fact %s: %s", fact_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# --- Memory Compaction ---

@router.post("/facts/compact")
async def compact_memory():
    """
    Deduplicate stored facts by finding and removing near-duplicates.

    For each fact, searches for similar facts using FTS and computes
    word-overlap (Jaccard >= 0.8). Keeps the higher-confidence fact and
    marks the other as superseded.
    """
    memory = get_memory()
    try:
        all_facts = memory.semantic.list_all(limit=10000)
        active_facts = [f for f in all_facts if not getattr(f, "is_superseded", False)]

        superseded = 0
        checked_ids = set()

        for fact in active_facts:
            if fact.id in checked_ids:
                continue

            similar = memory.semantic.search(query=fact.content, limit=10)
            for s in similar:
                if s.id == fact.id or s.id in checked_ids:
                    continue
                if getattr(s, "is_superseded", False):
                    continue

                words_a = set(fact.content.lower().split())
                words_b = set(s.content.lower().split())
                if not words_a or not words_b:
                    continue

                overlap = len(words_a & words_b) / len(words_a | words_b)
                if overlap >= 0.8:
                    to_supersede = s if fact.confidence >= s.confidence else fact
                    to_supersede.is_superseded = True
                    memory.semantic.save(to_supersede)
                    checked_ids.add(to_supersede.id)
                    superseded += 1

        return {
            "success": True,
            "scanned": len(active_facts),
            "superseded": superseded,
            "message": f"Compacted {superseded} near-duplicates from {len(active_facts)} facts",
        }

    except Exception as e:
        logger.error("Compact failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Compact failed")


# --- Import History ---

@router.get("/import/history")
async def get_import_history():
    """Return all import events, newest first."""
    return {
        "history": import_history.list_all(),
        "count": import_history.count(),
    }


@router.post("/import/undo/{batch_id}")
async def undo_import(batch_id: str):
    """
    Undo an import by deleting all facts from that batch.

    Only works if the batch recorded fact_ids during import.
    """
    memory = get_memory()
    batch = import_history.get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Import batch not found")

    fact_ids = batch.get("fact_ids", [])
    deleted = 0
    for fid in fact_ids:
        try:
            fact = memory.semantic.get(fid)
            if fact:
                memory.semantic.delete(fid)
                deleted += 1
        except Exception as e:
            logger.warning("Failed to delete fact %s during undo: %s", fid, e)

    import_history.mark_undone(batch_id, deleted)

    return {
        "success": True,
        "batch_id": batch_id,
        "facts_deleted": deleted,
    }


@router.get("/analytics/report")
async def analytics_report():
    """
    Generate a comprehensive memory analytics report.

    Uses the Anamnesis MemoryAnalyzer to compute health scores,
    topic distribution, emotional patterns, and temporal stats.
    """
    memory = get_memory()
    try:
        from anamnesis.analytics.analyzer import MemoryAnalyzer
        analyzer = MemoryAnalyzer(memory.episodic, memory.semantic, memory.backend)
        report = analyzer.generate_report()

        # Serialize dataclass → JSON-safe dict
        def _serialize_topic(t):
            return {
                "topic": t.topic,
                "count": t.count,
                "first_seen": t.first_seen.isoformat() if t.first_seen else None,
                "last_seen": t.last_seen.isoformat() if t.last_seen else None,
                "avg_importance": round(t.avg_importance, 3),
                "keywords": t.keywords,
            }

        def _serialize_emotion(e):
            return {
                "valence": e.valence.value if hasattr(e.valence, "value") else str(e.valence),
                "count": e.count,
                "percentage": round(e.percentage, 1),
                "trend": e.trend,
            }

        return {
            "generated_at": report.generated_at.isoformat(),
            "total_episodes": report.total_episodes,
            "total_facts": report.total_facts,
            "memory_span_days": report.memory_span_days,
            "health": {
                "healthy": report.healthy_memories,
                "aging": report.aging_memories,
                "at_risk": report.at_risk_memories,
                "forgotten": report.forgotten_memories,
            },
            "top_topics": [_serialize_topic(t) for t in report.top_topics],
            "topic_diversity": round(report.topic_diversity, 3),
            "emotion_distribution": [_serialize_emotion(e) for e in report.emotion_distribution],
            "emotional_balance": round(report.emotional_balance, 3),
            "temporal": {
                "period": report.temporal_stats.period,
                "peak_period": report.temporal_stats.peak_period,
                "activity_trend": report.temporal_stats.activity_trend,
            },
            "most_accessed": report.most_accessed[:5],
            "most_important": report.most_important[:5],
            "avg_episode_importance": round(report.avg_episode_importance, 3),
            "avg_fact_confidence": round(report.avg_fact_confidence, 3),
            "facts_by_type": report.facts_by_type,
        }

    except ImportError:
        return {"error": "Analytics module not available"}
    except Exception as e:
        logger.error("Analytics report failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Analytics report failed")


# --- Fact Quality Scoring ---

@router.get("/analytics/quality")
async def fact_quality_report():
    """
    Score all facts for quality using the Anamnesis QualityScorer.

    Returns per-fact quality reports and aggregate statistics.
    """
    memory = get_memory()
    try:
        from anamnesis.quality.scorer import QualityScorer

        scorer = QualityScorer()
        all_facts = memory.semantic.list_all(limit=Config.MAX_FACTS_LIST)
        all_episodes = memory.episodic.list_all()

        reports = scorer.score_batch_facts(all_facts)
        stats = scorer.get_stats(all_episodes, all_facts)

        return {
            "facts": [
                {
                    "id": r.memory_id,
                    "overall_score": round(r.overall_score, 3),
                    "quality_level": r.quality_level.value,
                    "completeness": round(r.completeness_score, 3),
                    "richness": round(r.richness_score, 3),
                    "reliability": round(r.reliability_score, 3),
                    "activity": round(r.activity_score, 3),
                    "suggestions": r.suggestions,
                }
                for r in reports
            ],
            "stats": {
                "total": stats.total_memories,
                "avg_quality": round(stats.avg_quality, 3),
                "distribution": stats.quality_distribution,
                "common_issues": stats.common_issues,
            },
        }

    except ImportError:
        return {"error": "Quality scorer module not available"}
    except Exception as e:
        logger.error("Quality report failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Quality report failed")


# --- Memory Graph Links ---

@router.get("/graph/links/{fact_id}")
async def get_fact_links(fact_id: str):
    """
    Get all graph links for a fact (related facts, contradictions, etc).

    Uses the Anamnesis MemoryGraph to find connected memories.
    """
    try:
        from anamnesis.graph.memory_graph import MemoryGraph

        memory = get_memory()
        graph = MemoryGraph(str(Config.DB_PATH))
        links = graph.get_links(fact_id)

        result = []
        for link in links:
            # Try to resolve the linked fact's content
            other_id = link.target_id if link.source_id == fact_id else link.source_id
            other_fact = memory.semantic.get(other_id)
            result.append({
                "linked_id": other_id,
                "linked_content": other_fact.content if other_fact else None,
                "link_type": link.link_type.value,
                "weight": round(link.weight, 3),
                "direction": "outgoing" if link.source_id == fact_id else "incoming",
            })

        return {"fact_id": fact_id, "links": result, "count": len(result)}

    except ImportError:
        return {"fact_id": fact_id, "links": [], "count": 0, "error": "Graph module not available"}
    except Exception as e:
        logger.error("Graph links failed for %s: %s", fact_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Graph query failed")


@router.get("/graph/stats")
async def graph_stats():
    """Return memory graph statistics (total links, links by type)."""
    try:
        from anamnesis.graph.memory_graph import MemoryGraph

        graph = MemoryGraph(str(Config.DB_PATH))
        stats = graph.get_stats()
        return stats

    except ImportError:
        return {"total_links": 0, "by_type": {}, "error": "Graph module not available"}
    except Exception as e:
        logger.error("Graph stats failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Graph stats failed")


# --- Emotional Recall ---

@router.get("/retrieval/emotional")
async def emotional_recall(valence: str = "positive", limit: int = 10):
    """
    Retrieve memories by emotional valence (positive/negative/neutral).

    Uses the Anamnesis EmotionalRecall engine for mood-aware retrieval.
    """
    memory = get_memory()
    try:
        from anamnesis.emotional.recall import EmotionalRecall
        from anamnesis.models import EmotionalValence

        recall = EmotionalRecall(memory.episodic, memory.semantic)

        valence_map = {
            "positive": EmotionalValence.POSITIVE,
            "very_positive": EmotionalValence.VERY_POSITIVE,
            "negative": EmotionalValence.NEGATIVE,
            "very_negative": EmotionalValence.VERY_NEGATIVE,
            "neutral": EmotionalValence.NEUTRAL,
        }

        target = valence_map.get(valence.lower())
        if not target:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid valence. Choose from: {', '.join(valence_map.keys())}",
            )

        episodes = recall.recall_by_emotion(target, limit=limit)
        return {
            "valence": valence,
            "episodes": [
                {
                    "id": ep.id,
                    "content": ep.content,
                    "importance": ep.importance,
                    "emotion": ep.metadata.get("emotion", "") if ep.metadata else "",
                }
                for ep in episodes
            ],
            "count": len(episodes),
        }

    except HTTPException:
        raise
    except ImportError:
        return {"valence": valence, "episodes": [], "count": 0, "error": "Emotional recall module not available"}
    except Exception as e:
        logger.error("Emotional recall failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Emotional recall failed")


@router.get("/retrieval/emotional/history")
async def emotional_history(days: int = 30):
    """Analyze emotional patterns over recent history."""
    memory = get_memory()
    try:
        from anamnesis.emotional.recall import EmotionalRecall

        recall = EmotionalRecall(memory.episodic, memory.semantic)
        analysis = recall.analyze_emotional_history(days_back=days)
        return analysis

    except ImportError:
        return {"error": "Emotional recall module not available"}
    except Exception as e:
        logger.error("Emotional history failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Emotional history failed")


@router.get("/retrieval/temporal")
async def temporal_recall(q: str = "", days: int = 30, limit: int = 10):
    """
    Time-aware memory search using the Anamnesis TemporalRecall engine.

    Understands natural language time expressions like:
    - "what we discussed last week"
    - "Python topics recently"
    - "conversations about music yesterday"
    """
    memory = get_memory()
    try:
        from anamnesis.temporal.recall import TemporalRecall

        recall = TemporalRecall(memory.episodic, memory.semantic)

        if q:
            result = recall.recall(q, max_episodes=limit, max_facts=limit)
        else:
            result = recall.recall("", max_episodes=limit, max_facts=limit)

        return {
            "query": result.query.original_query,
            "core_query": result.query.core_query,
            "has_time_reference": result.query.has_time_reference,
            "time_filtered": result.time_filtered,
            "episodes": [
                {
                    "id": ep.id,
                    "content": ep.content,
                    "importance": ep.importance,
                    "started_at": ep.started_at.isoformat() if ep.started_at else None,
                }
                for ep in result.episodes
            ],
            "facts": [
                {
                    "id": f.id,
                    "content": f.content,
                    "confidence": f.confidence,
                }
                for f in result.facts
            ],
            "total_matches": result.total_matches,
        }

    except ImportError:
        return {"query": q, "episodes": [], "facts": [], "total_matches": 0, "error": "Temporal recall module not available"}
    except Exception as e:
        logger.error("Temporal recall failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Temporal recall failed")


@router.get("/update/check")
async def check_updates():
    """
    Check for new Velqua versions.

    Queries the GitHub releases API and compares the latest tag
    against the running version. No auto-install.
    """
    try:
        from backend.updater import check_for_updates
        info = await check_for_updates()
        return {
            "update_available": info.update_available,
            "current_version": info.current_version,
            "latest_version": info.latest_version,
            "release_url": info.release_url,
            "release_notes": info.release_notes,
            "error": info.error,
        }
    except Exception as e:
        logger.error("Update check failed: %s", e)
        return {
            "update_available": False,
            "current_version": "unknown",
            "latest_version": "unknown",
            "error": str(e),
        }


@router.get("/proxy-status")
async def proxy_status():
    """
    Pass-through health check for the Ollama proxy.

    Avoids hardcoding the proxy port in the frontend — the server knows
    the configured port and proxies the request.
    """
    proxy_url = f"http://127.0.0.1:{Config.PROXY_PORT}/"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(proxy_url)
            return response.json()
    except httpx.ConnectError:
        return {"status": "offline", "error": "Proxy not running"}
    except Exception:
        return {"status": "offline", "error": "Proxy unreachable"}


@router.get("/proxy-metrics")
async def proxy_metrics():
    """
    Pass-through for proxy metrics endpoint.

    Proxies GET /metrics from the proxy process so the frontend
    doesn't need to know the proxy port.
    """
    proxy_url = f"http://127.0.0.1:{Config.PROXY_PORT}/metrics"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(proxy_url)
            return response.json()
    except httpx.ConnectError:
        return {"status": "offline", "error": "Proxy not running"}
    except Exception:
        return {"status": "offline", "error": "Proxy unreachable"}
