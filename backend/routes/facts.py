"""
Fact CRUD, search, merge, bulk delete, bulk import, tags, and type queries.
"""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from anamnesis.models import FactType
from backend.config import VelquaConfig as Config
from backend.logging_config import get_logger
from backend.routes._shared import get_memory

logger = get_logger("routes.facts")

router = APIRouter()


def _serialize_fact(f, include_tags=False, include_timestamp=False, feedback=None):
    """Serialize a Fact to dict. Central point to avoid field divergence."""
    meta = f.metadata if hasattr(f, "metadata") and f.metadata else {}
    d = {
        "id": f.id,
        "content": f.content,
        "type": str(f.fact_type),
        "confidence": f.confidence,
        "confirmation_count": getattr(f, "confirmation_count", 1),
        "topic": meta.get("topic", ""),
        "category": meta.get("category", ""),
        "emotion": meta.get("emotion", ""),
        "sentiment_score": meta.get("sentiment_score", 0.0),
    }
    if include_tags:
        d["tags"] = meta.get("tags", [])
    if include_timestamp:
        d["timestamp"] = f.created_at.isoformat() if hasattr(f, "created_at") else ""
    if feedback is not None:
        d["feedback"] = feedback
    return d


class FactUpdate(BaseModel):
    content: str = None
    confidence: float = None
    fact_type: str = None

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v):
        if v is not None and len(v.strip()) < Config.MIN_FACT_LENGTH:
            raise ValueError(f"Content must be at least {Config.MIN_FACT_LENGTH} characters")
        return v


class BulkDeleteRequest(BaseModel):
    fact_ids: List[str]


class MergeRequest(BaseModel):
    fact_ids: List[str]
    merged_content: str


class FeedbackRequest(BaseModel):
    is_positive: bool


class TagRequest(BaseModel):
    tags: List[str]


class BulkFactItem(BaseModel):
    content: str
    fact_type: str = "general"
    confidence: float = Config.DEFAULT_CONFIDENCE
    importance: float = 0.5
    tags: List[str] = []
    metadata: Dict[str, Any] = {}


class BulkImportRequest(BaseModel):
    facts: List[BulkFactItem]


BULK_IMPORT_LIMIT = 1000


@router.post("/facts/import/bulk")
async def bulk_import_facts(request: BulkImportRequest):
    """
    Bulk-import an array of facts in a single request.

    Validates all items upfront, then inserts in a batch using a single
    DB connection.  Deduplication runs per-fact via the existing
    SemanticStore.add_fact pipeline.

    Returns counts of inserted / skipped (duplicate) / failed facts.
    """
    if not request.facts:
        raise HTTPException(status_code=400, detail="No facts provided")

    if len(request.facts) > BULK_IMPORT_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {BULK_IMPORT_LIMIT} facts per request",
        )

    # --- Validate all items before touching the DB ---
    valid_types = {t.value for t in FactType}
    validation_errors = []

    for i, item in enumerate(request.facts):
        stripped = item.content.strip()
        if len(stripped) < Config.MIN_FACT_LENGTH:
            validation_errors.append({
                "index": i,
                "error": f"Content too short (min {Config.MIN_FACT_LENGTH} chars)",
            })
        elif len(stripped) > Config.MAX_FACT_LENGTH:
            validation_errors.append({
                "index": i,
                "error": f"Content too long (max {Config.MAX_FACT_LENGTH} chars)",
            })
        if item.fact_type not in valid_types:
            validation_errors.append({
                "index": i,
                "error": f"Invalid fact_type '{item.fact_type}'",
            })
        if not 0.0 <= item.confidence <= 1.0:
            validation_errors.append({
                "index": i,
                "error": "Confidence must be between 0.0 and 1.0",
            })
        if not 0.0 <= item.importance <= 1.0:
            validation_errors.append({
                "index": i,
                "error": "Importance must be between 0.0 and 1.0",
            })

    if validation_errors:
        raise HTTPException(status_code=422, detail={
            "message": f"{len(validation_errors)} validation error(s)",
            "errors": validation_errors,
        })

    # --- Insert facts using a persistent connection for the batch ---
    memory = get_memory()
    inserted = 0
    skipped = 0
    failed = 0
    errors: List[dict] = []
    fact_ids: List[str] = []

    memory.backend.connect()
    try:
        for i, item in enumerate(request.facts):
            try:
                meta = dict(item.metadata) if item.metadata else {}
                if item.tags:
                    meta["tags"] = item.tags

                result = memory.semantic.add_fact(
                    content=item.content.strip(),
                    fact_type=FactType(item.fact_type),
                    confidence=item.confidence,
                    importance=item.importance,
                    metadata=meta if meta else None,
                )
                if result.confirmation_count > 1:
                    skipped += 1
                else:
                    inserted += 1
                    fact_ids.append(result.id)
            except Exception as e:
                failed += 1
                errors.append({"index": i, "error": str(e)})
                logger.warning("Bulk import fact %d failed: %s", i, e)
    finally:
        memory.backend.close()

    return {
        "success": failed == 0,
        "inserted": inserted,
        "skipped": skipped,
        "failed": failed,
        "errors": errors,
        "fact_ids": fact_ids,
    }


@router.get("/facts/list")
async def list_facts(limit: int = 50, offset: int = 0):
    """List stored facts with pagination."""
    memory = get_memory()
    try:
        # Get the page of facts
        all_facts = memory.semantic.list_all(limit=limit + offset)
        facts = all_facts[offset:offset + limit]

        # Use count() — avoids fetching all facts just to count them
        total_count = memory.semantic.count()

        # Batch-fetch feedback summaries
        fact_ids = [f.id for f in facts]
        fb_map = memory.backend.get_fact_feedback_summaries(fact_ids) if fact_ids else {}

        return {
            "facts": [
                _serialize_fact(
                    f, include_tags=True, include_timestamp=True,
                    feedback=fb_map.get(f.id, {"thumbs_up": 0, "thumbs_down": 0}),
                )
                for f in facts
            ],
            "total": total_count,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error("Failed to list facts: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/facts/{fact_id}")
async def delete_fact(fact_id: str):
    """Delete a specific fact by ID."""
    memory = get_memory()
    try:
        fact = memory.semantic.get(fact_id)
        if not fact:
            raise HTTPException(status_code=404, detail="Fact not found")

        memory.semantic.delete(fact_id)
        return {"success": True, "message": "Fact deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete fact %s: %s", fact_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/facts/search")
async def search_facts(q: str, limit: int = 20):
    """Full-text search across all facts."""
    memory = get_memory()
    try:
        results = memory.semantic.search(query=q, limit=limit)
        fact_ids = [f.id for f in results]
        fb_map = memory.backend.get_fact_feedback_summaries(fact_ids) if fact_ids else {}
        return {
            "query": q,
            "results": [
                _serialize_fact(
                    f,
                    feedback=fb_map.get(f.id, {"thumbs_up": 0, "thumbs_down": 0}),
                )
                for f in results
            ],
            "count": len(results),
        }
    except Exception as e:
        logger.error("Failed to search facts for '%s': %s", q, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.patch("/facts/{fact_id}")
async def update_fact(fact_id: str, update: FactUpdate):
    """Edit a fact's content, confidence, or type."""
    memory = get_memory()
    try:
        fact = memory.semantic.get(fact_id)
        if not fact:
            raise HTTPException(status_code=404, detail="Fact not found")

        if update.content is not None:
            fact.content = update.content
        if update.confidence is not None:
            fact.confidence = max(0.0, min(1.0, update.confidence))
        if update.fact_type is not None:
            fact.fact_type = update.fact_type

        memory.semantic.save(fact)
        return {
            "success": True,
            "fact": {
                "id": fact.id,
                "content": fact.content,
                "type": str(fact.fact_type),
                "confidence": fact.confidence,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update fact %s: %s", fact_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/facts/bulk-delete")
async def bulk_delete_facts(request: BulkDeleteRequest):
    """Delete multiple facts in one call. Reports per-fact errors."""
    try:
        memory = get_memory()
        deleted = 0
        not_found = 0
        errors = []
        for fact_id in request.fact_ids:
            try:
                fact = memory.semantic.get(fact_id)
                if fact:
                    memory.semantic.delete(fact_id)
                    deleted += 1
                else:
                    not_found += 1
            except Exception as e:
                errors.append({"fact_id": fact_id, "error": str(e)})

        result = {"success": True, "deleted": deleted, "not_found": not_found}
        if errors:
            result["errors"] = errors
            result["success"] = deleted > 0  # Partial success if some worked
        return result
    except Exception as e:
        logger.error("Failed to bulk delete: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/facts/merge")
async def merge_facts(request: MergeRequest):
    """Merge multiple facts into one, keeping the highest confidence."""
    memory = get_memory()
    try:
        if len(request.fact_ids) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 facts to merge")

        facts = []
        for fid in request.fact_ids:
            fact = memory.semantic.get(fid)
            if not fact:
                raise HTTPException(status_code=404, detail=f"Fact {fid} not found")
            facts.append(fact)

        max_confidence = max(f.confidence for f in facts)
        merged = memory.semantic.add_fact(
            content=request.merged_content,
            fact_type=facts[0].fact_type,
            confidence=max_confidence,
        )

        # Delete originals — if any delete fails, roll back the merged fact
        try:
            for fid in request.fact_ids:
                memory.semantic.delete(fid)
        except Exception as del_err:
            logger.error("Failed to delete originals during merge: %s", del_err)
            # Rollback: remove the newly created merge to avoid duplicates
            try:
                memory.semantic.delete(merged.id)
            except Exception as rb_err:
                logger.error("Merge rollback also failed for %s: %s", merged.id, rb_err)
            raise

        return {
            "success": True,
            "merged_fact": {
                "id": merged.id,
                "content": merged.content,
                "confidence": merged.confidence,
            },
            "deleted_count": len(request.fact_ids),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to merge facts: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/facts/stats")
async def fact_stats():
    """Aggregate statistics: type counts and confidence distribution."""
    memory = get_memory()
    try:
        all_facts = memory.semantic.list_all(limit=Config.MAX_FACTS_LIST)

        type_counts = {}
        confidence_buckets = {"high": 0, "medium": 0, "low": 0}

        for f in all_facts:
            ft = str(f.fact_type)
            type_counts[ft] = type_counts.get(ft, 0) + 1

            if f.confidence >= 0.8:
                confidence_buckets["high"] += 1
            elif f.confidence >= 0.5:
                confidence_buckets["medium"] += 1
            else:
                confidence_buckets["low"] += 1

        return {
            "total": len(all_facts),
            "by_type": type_counts,
            "by_confidence": confidence_buckets,
        }
    except Exception as e:
        logger.error("Failed to compute fact stats: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/facts/timeline")
async def fact_timeline():
    """Facts grouped by date for the timeline view."""
    memory = get_memory()
    try:
        all_facts = memory.semantic.list_all(limit=Config.MAX_FACTS_LIST)

        timeline = {}
        for f in all_facts:
            date_key = ""
            try:
                if hasattr(f, 'first_learned') and f.first_learned:
                    date_key = f.first_learned.strftime("%Y-%m-%d")
                elif hasattr(f, 'created_at') and f.created_at:
                    date_key = f.created_at.strftime("%Y-%m-%d")
            except (AttributeError, TypeError):
                pass  # Non-datetime value — fall through to "unknown"

            if not date_key:
                date_key = "unknown"

            if date_key not in timeline:
                timeline[date_key] = []

            timeline[date_key].append(_serialize_fact(f))

        sorted_dates = sorted(
            [d for d in timeline.keys() if d != "unknown"],
            reverse=True,
        )
        if "unknown" in timeline:
            sorted_dates.append("unknown")

        return {
            "dates": sorted_dates,
            "groups": timeline,
            "total_facts": len(all_facts),
            "total_days": len([d for d in sorted_dates if d != "unknown"]),
        }
    except Exception as e:
        logger.error("Failed to build timeline: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/facts/{fact_id}/tags")
async def add_tags(fact_id: str, request: TagRequest):
    """Add tags to a fact (deduplicates against existing tags)."""
    memory = get_memory()
    try:
        fact = memory.semantic.get(fact_id)
        if not fact:
            raise HTTPException(status_code=404, detail="Fact not found")

        if not hasattr(fact, "metadata") or fact.metadata is None:
            fact.metadata = {}
        existing_tags = fact.metadata.get("tags", [])
        new_tags = list(dict.fromkeys(existing_tags + request.tags))
        fact.metadata["tags"] = new_tags
        memory.semantic.save(fact)

        return {"success": True, "tags": new_tags}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to add tags to fact %s: %s", fact_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/facts/{fact_id}/tags/{tag}")
async def remove_tag(fact_id: str, tag: str):
    """Remove a single tag from a fact."""
    memory = get_memory()
    try:
        fact = memory.semantic.get(fact_id)
        if not fact:
            raise HTTPException(status_code=404, detail="Fact not found")

        if not hasattr(fact, "metadata") or fact.metadata is None:
            fact.metadata = {}
        tags = fact.metadata.get("tags", [])
        if tag in tags:
            tags.remove(tag)
            fact.metadata["tags"] = tags
            memory.semantic.save(fact)

        return {"success": True, "tags": tags}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to remove tag from fact %s: %s", fact_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/facts/by-type/{fact_type}")
async def get_facts_by_type(fact_type: str, limit: int = 50):
    """Get facts filtered by their FactType value."""
    memory = get_memory()
    try:
        facts = memory.semantic.get_by_type(fact_type, limit=limit)
        return {
            "type": fact_type,
            "facts": [_serialize_fact(f, include_tags=True) for f in facts],
            "count": len(facts),
        }
    except Exception as e:
        logger.error("Failed to get facts by type %s: %s", fact_type, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/facts/types")
async def get_fact_types():
    """List all FactType enum values."""
    from anamnesis.models import FactType
    return {
        "types": [
            {"value": t.value, "label": t.name.title()}
            for t in FactType
        ]
    }


# === Feedback endpoints ===

@router.post("/facts/{fact_id}/feedback")
async def submit_feedback(fact_id: str, request: FeedbackRequest):
    """Submit thumbs-up or thumbs-down feedback for a fact."""
    memory = get_memory()
    try:
        fact = memory.semantic.get(fact_id)
        if not fact:
            raise HTTPException(status_code=404, detail="Fact not found")

        # Record feedback
        memory.backend.save_fact_feedback(fact_id, request.is_positive)

        # Adjust confidence
        delta = 0.05 if request.is_positive else -0.05
        fact.confidence = max(0.0, min(1.0, fact.confidence + delta))
        memory.semantic.save(fact)

        summary = memory.backend.get_fact_feedback_summary(fact_id)
        return {
            "success": True,
            "fact_id": fact_id,
            "feedback": summary,
            "new_confidence": fact.confidence,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to submit feedback for %s: %s", fact_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/facts/{fact_id}/feedback")
async def get_feedback(fact_id: str):
    """Get feedback summary for a fact."""
    memory = get_memory()
    try:
        fact = memory.semantic.get(fact_id)
        if not fact:
            raise HTTPException(status_code=404, detail="Fact not found")

        summary = memory.backend.get_fact_feedback_summary(fact_id)
        return {"feedback": summary}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get feedback for %s: %s", fact_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
