"""
Episode ingestion and retrieval routes.

Episodes are temporal memories — timestamped experiences with emotional
valence, source agent tracking, and configurable decay rates. Unlike
facts (context-independent knowledge), episodes capture "what happened"
and fade faster over time.

POST /import/episodes activates the cognitive features that make Velqua
different from vanilla RAG: recency weighting, emotional relevance, and
temporal decay.
"""
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from anamnesis.models import EmotionalValence, Episode
from backend.config import VelquaConfig as Config
from backend.logging_config import get_logger
from backend.routes._shared import get_memory

logger = get_logger("routes.episodes")

router = APIRouter()


# === Request/Response Models ===

class EpisodeChunk(BaseModel):
    """A single episode to ingest."""
    content: str
    timestamp: Optional[str] = None  # ISO 8601, defaults to now
    emotional_valence: str = "neutral"  # very_negative, negative, neutral, positive, very_positive
    source_agent: str = "unknown"
    decay_rate: Optional[float] = None  # Override default decay, 0.0-1.0
    importance: float = 0.5
    topic: Optional[str] = None
    tags: Optional[List[str]] = None

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError("Episode content must be at least 10 characters")
        return v.strip()

    @field_validator("emotional_valence")
    @classmethod
    def valid_valence(cls, v):
        valid = {"very_negative", "negative", "neutral", "positive", "very_positive"}
        if v not in valid:
            raise ValueError(f"emotional_valence must be one of: {', '.join(sorted(valid))}")
        return v


class EpisodeImportRequest(BaseModel):
    """Batch episode import request."""
    episodes: List[EpisodeChunk]
    source: str = "api"  # Import source tag


class EpisodeImportResult(BaseModel):
    """Result of episode import."""
    success: bool
    episodes_stored: int
    episodes_failed: int
    episode_ids: List[str]
    message: str


# === Valence mapping ===

_VALENCE_MAP = {
    "very_negative": EmotionalValence.VERY_NEGATIVE,
    "negative": EmotionalValence.NEGATIVE,
    "neutral": EmotionalValence.NEUTRAL,
    "positive": EmotionalValence.POSITIVE,
    "very_positive": EmotionalValence.VERY_POSITIVE,
}


def _serialize_episode(ep: Episode) -> dict:
    """Serialize an Episode to a response dict."""
    return {
        "id": ep.id,
        "summary": ep.summary,
        "topic": ep.topic,
        "started_at": ep.started_at.isoformat() if ep.started_at else None,
        "overall_valence": ep.overall_valence.name.lower() if isinstance(ep.overall_valence, EmotionalValence) else "neutral",
        "importance": ep.importance,
        "tags": ep.tags or [],
        "access_count": ep.access_count,
        "source_id": ep.source_id,
    }


# === Endpoints ===

@router.post("/import/episodes", response_model=EpisodeImportResult)
async def import_episodes(request: EpisodeImportRequest):
    """
    Ingest chunks as episodes — temporal memories with emotional context.

    Unlike fact ingestion (which creates context-independent knowledge),
    episode ingestion preserves temporal ordering, emotional valence,
    and agent provenance. Episodes are weighted by recency and emotional
    relevance during retrieval, activating Velqua's cognitive features.

    Body:
        {
            "episodes": [
                {
                    "content": "User expressed frustration with the deployment pipeline",
                    "timestamp": "2026-03-20T14:30:00",
                    "emotional_valence": "negative",
                    "source_agent": "dev-assistant",
                    "decay_rate": 0.15,
                    "importance": 0.7,
                    "topic": "devops",
                    "tags": ["deployment", "frustration"]
                }
            ],
            "source": "conversation_export"
        }
    """
    memory = get_memory()
    if not memory:
        raise HTTPException(status_code=503, detail="Memory system not initialized")

    stored = 0
    failed = 0
    episode_ids = []

    for chunk in request.episodes:
        try:
            # Parse timestamp
            if chunk.timestamp:
                try:
                    ts = datetime.fromisoformat(chunk.timestamp)
                except ValueError:
                    raise ValueError(f"Invalid timestamp format: {chunk.timestamp}")
            else:
                ts = datetime.now()

            # Map valence string to enum
            valence = _VALENCE_MAP.get(chunk.emotional_valence, EmotionalValence.NEUTRAL)

            # Build metadata with source agent and decay rate
            metadata = {
                "source_agent": chunk.source_agent,
                "import_source": request.source,
            }
            if chunk.decay_rate is not None:
                metadata["decay_rate"] = chunk.decay_rate

            # Create Episode object
            episode = Episode(
                id=str(uuid.uuid4()),
                summary=chunk.content,
                messages=[],  # Raw episodes don't have message pairs
                started_at=ts,
                ended_at=ts,  # Point-in-time for imported episodes
                topic=chunk.topic,
                overall_valence=valence,
                importance=chunk.importance,
                source_id=chunk.source_agent,
                metadata=metadata,
                tags=chunk.tags or [],
            )

            # Save to episodic store
            memory.episodic.save(episode)
            episode_ids.append(episode.id)
            stored += 1

            logger.debug(
                "Stored episode %s: valence=%s importance=%.2f topic=%s",
                episode.id, chunk.emotional_valence, chunk.importance, chunk.topic,
            )

        except Exception as e:
            failed += 1
            logger.warning("Failed to store episode: %s", e)

    logger.info(
        "Episode import: %d stored, %d failed (source=%s)",
        stored, failed, request.source,
    )

    return EpisodeImportResult(
        success=stored > 0 or failed == 0,
        episodes_stored=stored,
        episodes_failed=failed,
        episode_ids=episode_ids,
        message=f"Imported {stored} episodes" + (f" ({failed} failed)" if failed else ""),
    )


@router.get("/episodes/list")
async def list_episodes(limit: int = 50, offset: int = 0):
    """List stored episodes with pagination."""
    memory = get_memory()
    try:
        episodes = memory.episodic.list_all(limit=limit, offset=offset)
        total = memory.episodic.count()
        return {
            "episodes": [_serialize_episode(ep) for ep in episodes],
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        logger.error("Failed to list episodes: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/episodes/search")
async def search_episodes(q: str, limit: int = 20):
    """Full-text search across episodes."""
    memory = get_memory()
    try:
        results = memory.episodic.search(query=q, limit=limit)
        return {
            "query": q,
            "results": [_serialize_episode(ep) for ep in results],
            "count": len(results),
        }
    except Exception as e:
        logger.error("Failed to search episodes: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/episodes/by-valence/{valence}")
async def get_episodes_by_valence(valence: str, limit: int = 20):
    """Get episodes filtered by emotional valence."""
    memory = get_memory()
    valence_enum = _VALENCE_MAP.get(valence)
    if valence_enum is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid valence. Must be one of: {', '.join(sorted(_VALENCE_MAP.keys()))}",
        )
    try:
        episodes = memory.episodic.get_emotional(valence_enum, limit=limit)
        return {
            "valence": valence,
            "episodes": [_serialize_episode(ep) for ep in episodes],
            "count": len(episodes),
        }
    except Exception as e:
        logger.error("Failed to get episodes by valence: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/episodes/{episode_id}")
async def delete_episode(episode_id: str, hard: bool = False):
    """Delete an episode (soft by default, hard=true for permanent)."""
    memory = get_memory()
    try:
        episode = memory.episodic.get(episode_id)
        if not episode:
            raise HTTPException(status_code=404, detail="Episode not found")

        memory.episodic.delete(episode_id, hard=hard)
        return {"success": True, "message": f"Episode {'deleted' if hard else 'forgotten'}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete episode %s: %s", episode_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/episodes/stats")
async def episode_stats():
    """Episode statistics: counts by valence and importance distribution."""
    memory = get_memory()
    try:
        all_episodes = memory.episodic.list_all(limit=10000)

        valence_counts = {}
        importance_buckets = {"high": 0, "medium": 0, "low": 0}

        for ep in all_episodes:
            v_name = ep.overall_valence.name.lower() if isinstance(ep.overall_valence, EmotionalValence) else "neutral"
            valence_counts[v_name] = valence_counts.get(v_name, 0) + 1

            if ep.importance >= 0.7:
                importance_buckets["high"] += 1
            elif ep.importance >= 0.4:
                importance_buckets["medium"] += 1
            else:
                importance_buckets["low"] += 1

        return {
            "total": len(all_episodes),
            "by_valence": valence_counts,
            "by_importance": importance_buckets,
        }
    except Exception as e:
        logger.error("Failed to compute episode stats: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
