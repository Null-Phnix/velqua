"""
Activity log endpoints — chronological feed of system events.
"""
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from backend.activity.db import list_events, count_events, clear_events, EVENT_TYPES
from backend.logging_config import get_logger

logger = get_logger("routes.activity")

router = APIRouter(prefix="/activity", tags=["activity"])


def _error_response(message: str, status_code: int) -> JSONResponse:
    """Return a consistent API error payload."""
    return JSONResponse(
        status_code=status_code,
        content={"error": message, "code": status_code},
    )


@router.get("")
async def get_activity(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    event_type: Optional[str] = Query(None),
):
    """Return activity events in reverse chronological order."""
    normalized_event_type = (event_type or "").strip() or None
    if normalized_event_type and normalized_event_type not in EVENT_TYPES:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown event type: {normalized_event_type}", "events": [], "total": 0},
        )

    try:
        events = list_events(limit=limit, offset=offset, event_type=normalized_event_type)
        total = count_events(event_type=normalized_event_type)
        return {"events": events or [], "total": total or 0}
    except Exception as e:
        logger.error("Failed to fetch activity events: %s", e, exc_info=True)
        return _error_response("Failed to fetch activity events", 500)


@router.get("/types")
async def get_event_types():
    """Return all known event types."""
    try:
        return {"types": sorted(EVENT_TYPES)}
    except Exception as e:
        logger.error("Failed to fetch activity event types: %s", e, exc_info=True)
        return _error_response("Failed to fetch activity event types", 500)


@router.delete("")
async def delete_activity():
    """Clear all activity events."""
    try:
        deleted = clear_events()
        logger.info("Cleared %d activity events", deleted)
        return {"deleted": deleted or 0}
    except Exception as e:
        logger.error("Failed to clear activity events: %s", e, exc_info=True)
        return _error_response("Failed to clear activity events", 500)
