"""
Review queue routes for fact approval/rejection.

Facts extracted by the auto-learner with medium quality scores (0.4-0.7)
land in a pending queue. Users approve or reject them here before they
enter the permanent knowledge base.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from anamnesis.models import FactType
from backend.auto_learner import PendingFactStore
from backend.config import VelquaConfig as Config
from backend.routes._shared import get_memory

router = APIRouter()

# Shared pending store instance (same JSON file the auto-learner writes to)
_pending_store = PendingFactStore()


def get_pending_store() -> PendingFactStore:
    """Accessor for tests that need to swap the store."""
    return _pending_store


@router.get("/review/pending")
async def list_pending_facts():
    """List facts waiting for user review, enriched with contradiction warnings."""
    pending = _pending_store.list_all()

    # Enrich with contradiction checks against stored facts
    try:
        from anamnesis.consolidation.contradiction import detect_contradictions
        from anamnesis.models import Fact

        memory = get_memory()
        existing_facts = memory.semantic.list_all(limit=100)

        for item in pending:
            if existing_facts:
                temp_fact = Fact(content=item["content"])
                results = detect_contradictions(temp_fact, existing_facts, threshold=0.5)
                item["contradictions"] = [
                    {
                        "content": r.existing_fact.content,
                        "type": r.contradiction_type,
                        "confidence": r.confidence,
                        "explanation": r.explanation,
                    }
                    for r in results if r.is_contradiction and r.existing_fact
                ]
            else:
                item["contradictions"] = []
    except (ImportError, Exception):
        # Contradiction enrichment is optional — don't fail the list
        for item in pending:
            if "contradictions" not in item:
                item["contradictions"] = []

    return {
        "pending": pending,
        "count": _pending_store.count(),
    }


@router.post("/review/approve/{pending_id}")
async def approve_pending_fact(pending_id: str):
    """Approve a pending fact and commit it to the knowledge base."""
    entry = _pending_store.approve(pending_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Pending fact not found")

    memory = get_memory()
    result = memory.semantic.add_fact(
        content=entry["content"],
        fact_type=FactType.GENERAL,
        confidence=max(0.5, entry.get("quality_score", 0.5)),
        metadata={"source": entry.get("source", "review")},
    )
    return {
        "success": True,
        "fact_id": result.id,
        "content": result.content,
    }


@router.post("/review/reject/{pending_id}")
async def reject_pending_fact(pending_id: str):
    """Reject a pending fact (permanently discard it)."""
    if not _pending_store.reject(pending_id):
        raise HTTPException(status_code=404, detail="Pending fact not found")
    return {"success": True}


@router.post("/review/approve-all")
async def approve_all_pending():
    """Approve every pending fact in one operation."""
    memory = get_memory()
    entries = _pending_store.approve_all()
    stored = 0
    for entry in entries:
        memory.semantic.add_fact(
            content=entry["content"],
            fact_type=FactType.GENERAL,
            confidence=max(0.5, entry.get("quality_score", 0.5)),
            metadata={"source": entry.get("source", "review")},
        )
        stored += 1
    return {"success": True, "approved": stored}


@router.post("/review/reject-all")
async def reject_all_pending():
    """Reject every pending fact in one operation."""
    count = _pending_store.reject_all()
    return {"success": True, "rejected": count}


class EditApproveRequest(BaseModel):
    content: str


@router.post("/review/edit-approve/{pending_id}")
async def edit_approve_pending_fact(pending_id: str, request: EditApproveRequest):
    """Edit a pending fact's content, then approve and commit to the knowledge base."""
    if len(request.content.strip()) < Config.MIN_FACT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Content must be at least {Config.MIN_FACT_LENGTH} characters",
        )

    entry = _pending_store.approve(pending_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Pending fact not found")

    memory = get_memory()
    result = memory.semantic.add_fact(
        content=request.content.strip(),
        fact_type=FactType.GENERAL,
        confidence=max(0.5, entry.get("quality_score", 0.5)),
        metadata={"source": entry.get("source", "review"), "edited": True},
    )
    return {
        "success": True,
        "fact_id": result.id,
        "content": result.content,
    }
