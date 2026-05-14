"""
Graph relationship routes.

Endpoints for detecting and querying fact-to-fact relationships
(contradiction, elaboration, temporal sequence).
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from backend.anamnesis.graph.relationships import (
    RelationshipType,
    detect_relationships,
)
from backend.routes._shared import get_memory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/relationships")
def get_relationships(
    fact_id: Optional[str] = Query(None, description="Filter to edges involving this fact"),
    type: Optional[str] = Query(None, description="Filter by relationship type"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """
    Get stored fact relationships.

    Returns edges from the fact_relationships table, optionally filtered
    by fact_id, relationship type, and minimum confidence.
    """
    mem = get_memory()
    if mem is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    # Validate type if provided
    valid_types = {rt.value for rt in RelationshipType}
    if type is not None and type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid type '{type}'. Valid: {sorted(valid_types)}",
        )

    edges = mem.backend.get_fact_relationships(
        fact_id=fact_id,
        relationship_type=type,
        min_confidence=min_confidence,
        limit=limit,
        offset=offset,
    )
    total = mem.backend.count_fact_relationships(
        fact_id=fact_id,
        relationship_type=type,
        min_confidence=min_confidence,
    )

    return {
        "relationships": edges,
        "count": len(edges),
        "total": total,
    }


@router.post("/relationships/detect")
def detect_and_store_relationships(
    type: Optional[str] = Query(None, description="Only detect this relationship type"),
    limit: int = Query(500, ge=1, le=5000, description="Max facts to analyze"),
):
    """
    Run relationship detection on stored facts and persist edges.

    Analyzes all facts pairwise (up to limit) and stores detected
    relationships in the fact_relationships table.
    """
    mem = get_memory()
    if mem is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    # Validate type if provided
    types_filter = None
    if type is not None:
        valid_types = {rt.value for rt in RelationshipType}
        if type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid type '{type}'. Valid: {sorted(valid_types)}",
            )
        types_filter = [RelationshipType(type)]

    # Fetch facts
    facts = mem.backend.list_facts(limit=limit)
    if not facts:
        return {"detected": 0, "stored": 0, "facts_analyzed": 0}

    # Detect relationships
    relationships = detect_relationships(facts, types=types_filter)

    # Persist to database
    stored = 0
    for rel in relationships:
        try:
            mem.backend.save_fact_relationship(
                source_id=rel.source_id,
                target_id=rel.target_id,
                relationship_type=rel.relationship_type.value,
                confidence=rel.confidence,
                evidence=rel.evidence,
                metadata=rel.metadata,
            )
            stored += 1
        except Exception as e:
            logger.warning("Failed to store relationship %s->%s: %s",
                           rel.source_id, rel.target_id, e)

    return {
        "detected": len(relationships),
        "stored": stored,
        "facts_analyzed": len(facts),
    }
