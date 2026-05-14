"""
Import routes: smart import, ChatGPT import, fact JSON import/export.

The smart import endpoint auto-detects file type (Claude memories,
Claude conversations, Claude projects, ChatGPT) and routes to the
appropriate extraction logic.
"""
import asyncio
import json
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from anamnesis.integration.claude_importer import import_claude_memories
from anamnesis.models import FactType
from backend.file_detector import (
    detect_file_type, FileType,
    extract_facts_from_conversations,
    extract_facts_from_projects,
    extract_facts_from_chatgpt,
)
from backend.validators import validate_upload, sanitize_filename, ValidationError
from backend.config import VelquaConfig as Config
from backend.logging_config import get_logger
from backend.routes._shared import get_memory, import_history

logger = get_logger("routes.imports")

router = APIRouter()


class ImportResult(BaseModel):
    success: bool
    facts_extracted: int
    facts_stored: int
    fiction_filtered: int = 0
    duplicates_skipped: int = 0
    projects: int = 0
    file_type: str = "unknown"
    message: str
    warning: str = ""


def _store_facts_batch(raw_facts: List[str], fact_type, confidence: float,
                       filter_fiction: bool = True) -> dict:
    """
    Store a batch of extracted fact strings, handling dedup and fiction filtering.

    Returns dict with stored, duplicates, fiction_count, and fact_ids.

    This is the shared core that all import handlers use. Each handler
    extracts facts differently but stores them the same way.
    """
    memory = get_memory()
    stored = 0
    duplicates = 0
    fiction_count = 0
    fact_ids = []

    # Use Anamnesis fantasy keywords for richer fiction filtering
    try:
        from anamnesis.consolidation.context_detector import FANTASY_KEYWORDS
        fiction_words = FANTASY_KEYWORDS
    except ImportError:
        fiction_words = set(Config.FICTION_KEYWORDS)

    for fact_text in raw_facts:
        # Skip facts that are too short (not counted as fiction)
        if len(fact_text) <= Config.MIN_FACT_LENGTH:
            continue

        if filter_fiction:
            fact_words = set(fact_text.lower().split())
            is_fiction = bool(fact_words & fiction_words)
            if is_fiction:
                fiction_count += 1
                continue

        # Build metadata with topic + sentiment enrichment
        fact_metadata = {}
        try:
            from anamnesis.topics.detector import TopicDetector
            topic_result = TopicDetector().detect(fact_text)
            fact_metadata["topic"] = topic_result.main_topic
            fact_metadata["category"] = topic_result.category
        except Exception:
            pass

        result = memory.semantic.add_fact(
            content=fact_text,
            fact_type=fact_type,
            confidence=confidence,
            metadata=fact_metadata if fact_metadata else None,
        )
        if result.confirmation_count > 1:
            duplicates += 1
        else:
            stored += 1
            fact_ids.append(result.id)

    return {
        "stored": stored,
        "duplicates": duplicates,
        "fiction_count": fiction_count,
        "fact_ids": fact_ids,
    }


def _save_upload_to_temp(content: bytes) -> Path:
    """Write upload content to a secure temp file and return the path."""
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.json', delete=False) as tmp:
        tmp.write(content)
        return Path(tmp.name)


@router.post("/import/smart", response_model=ImportResult)
async def smart_import(file: UploadFile = File(...)):
    """
    Smart import that detects file type and extracts facts.

    Supports Claude memories.json, conversations.json, projects.json,
    and ChatGPT conversations.json. Auto-detects format from structure.
    """
    temp_path = None
    try:
        if file.size and file.size > Config.MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (max {Config.MAX_UPLOAD_SIZE_MB}MB)"
            )

        safe_filename = sanitize_filename(file.filename)
        content = await file.read()
        file_size = len(content) / (1024 * 1024)

        warning = ""
        if file_size > Config.LARGE_FILE_THRESHOLD_MB:
            warning = f"Large file ({file_size:.1f}MB) - this may take a minute"

        temp_path = _save_upload_to_temp(content)

        is_valid, error_msg = validate_upload(temp_path)
        if not is_valid:
            raise ValidationError(error_msg)

        file_type, metadata = detect_file_type(str(temp_path))

        # --- Claude memories.json ---
        if file_type == FileType.CLAUDE_MEMORIES:
            general_facts, project_facts = import_claude_memories(str(temp_path))

            stored = 0
            duplicates = 0
            fact_ids = []
            for ef in general_facts:
                result = get_memory().semantic.add_fact(
                    content=ef.content,
                    fact_type=ef.fact_type,
                    confidence=ef.confidence,
                )
                if result.confirmation_count > 1:
                    duplicates += 1
                else:
                    stored += 1
                    fact_ids.append(result.id)

            import_history.record(file_type, stored, duplicates, safe_filename, fact_ids)

            return ImportResult(
                success=True,
                facts_extracted=len(general_facts),
                facts_stored=stored,
                duplicates_skipped=duplicates,
                projects=len(project_facts),
                file_type=file_type,
                message=f"Imported {stored} facts from Claude memories ({duplicates} duplicates skipped)",
                warning=warning,
            )

        # --- Claude conversations.json ---
        elif file_type == FileType.CLAUDE_CONVERSATIONS:
            with open(temp_path) as f:
                conversations = json.load(f)

            raw_facts = extract_facts_from_conversations(
                conversations,
                max_conversations=min(Config.MAX_CONVERSATIONS, len(conversations)),
            )
            batch = _store_facts_batch(raw_facts, FactType.GENERAL, Config.DEFAULT_CONFIDENCE)
            import_history.record(file_type, batch["stored"], batch["duplicates"],
                                  safe_filename, batch["fact_ids"])

            return ImportResult(
                success=True,
                facts_extracted=len(raw_facts),
                facts_stored=batch["stored"],
                fiction_filtered=batch["fiction_count"],
                duplicates_skipped=batch["duplicates"],
                file_type=file_type,
                message=f"Extracted {batch['stored']} facts from {metadata['conversations']} conversations",
                warning=warning,
            )

        # --- Claude projects.json ---
        elif file_type == FileType.CLAUDE_PROJECTS:
            with open(temp_path) as f:
                projects = json.load(f)

            raw_facts = extract_facts_from_projects(
                projects,
                max_projects=min(Config.MAX_PROJECTS, len(projects)),
            )
            # Projects aren't fiction-filtered the same way — they're metadata
            batch = _store_facts_batch(raw_facts, FactType.PROJECT, Config.HIGH_CONFIDENCE,
                                       filter_fiction=False)
            import_history.record(file_type, batch["stored"], batch["duplicates"],
                                  safe_filename, batch["fact_ids"])

            return ImportResult(
                success=True,
                facts_extracted=len(raw_facts),
                facts_stored=batch["stored"],
                duplicates_skipped=batch["duplicates"],
                projects=len(projects),
                file_type=file_type,
                message=f"Extracted {batch['stored']} facts from {len(projects)} projects",
                warning=warning,
            )

        # --- ChatGPT conversations.json ---
        elif file_type == FileType.CHATGPT_CONVERSATIONS:
            with open(temp_path) as f:
                conversations = json.load(f)

            raw_facts = extract_facts_from_chatgpt(
                conversations,
                max_conversations=min(Config.MAX_CONVERSATIONS, len(conversations)),
            )
            batch = _store_facts_batch(raw_facts, FactType.GENERAL, Config.DEFAULT_CONFIDENCE)
            import_history.record(file_type, batch["stored"], batch["duplicates"],
                                  safe_filename, batch["fact_ids"])

            return ImportResult(
                success=True,
                facts_extracted=len(raw_facts),
                facts_stored=batch["stored"],
                fiction_filtered=batch["fiction_count"],
                duplicates_skipped=batch["duplicates"],
                file_type=file_type,
                message=f"Extracted {batch['stored']} facts from {len(conversations)} ChatGPT conversations",
                warning=warning,
            )

        else:
            return ImportResult(
                success=False,
                facts_extracted=0,
                facts_stored=0,
                file_type=file_type,
                message="Unsupported file format. Expected Claude or ChatGPT export",
            )

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e.msg}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Smart import failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Import failed")
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


@router.post("/import/smart/stream")
async def smart_import_stream(file: UploadFile = File(...)):
    """
    Smart import with real-time SSE progress.

    Yields 'data: {...}\\n\\n' events at each stage so the UI can show
    a live progress bar instead of a fake timed animation.

    Event shapes:
      { stage: 'uploading'|'validating'|'detecting'|'extracting'|'storing'|'complete'|'error',
        pct: 0-100, msg: string, ...extra }
    """
    content = await file.read()
    safe_filename = sanitize_filename(file.filename)

    async def _generate():
        temp_path = None
        try:
            yield f"data: {json.dumps({'stage': 'validating', 'pct': 20, 'msg': 'Validating format...'})}\n\n"
            await asyncio.sleep(0)

            temp_path = _save_upload_to_temp(content)
            is_valid, error_msg = validate_upload(temp_path)
            if not is_valid:
                yield f"data: {json.dumps({'stage': 'error', 'pct': 0, 'msg': error_msg})}\n\n"
                return

            yield f"data: {json.dumps({'stage': 'detecting', 'pct': 35, 'msg': 'Detecting file type...'})}\n\n"
            await asyncio.sleep(0)

            file_type, metadata = detect_file_type(str(temp_path))

            supported = (
                FileType.CLAUDE_MEMORIES, FileType.CLAUDE_CONVERSATIONS,
                FileType.CLAUDE_PROJECTS, FileType.CHATGPT_CONVERSATIONS,
            )
            if file_type not in supported:
                yield f"data: {json.dumps({'stage': 'error', 'pct': 0, 'msg': 'Unsupported file format'})}\n\n"
                return

            yield f"data: {json.dumps({'stage': 'extracting', 'pct': 50, 'msg': f'Extracting facts ({file_type})...'})}\n\n"
            await asyncio.sleep(0)

            # Extract raw fact strings from whichever format
            if file_type == FileType.CLAUDE_MEMORIES:
                general_facts, _ = import_claude_memories(str(temp_path))
                raw_texts = [ef.content for ef in general_facts]
                is_project = False
            elif file_type == FileType.CLAUDE_CONVERSATIONS:
                with open(temp_path) as f:
                    convos = json.load(f)
                raw_texts = extract_facts_from_conversations(
                    convos, max_conversations=min(Config.MAX_CONVERSATIONS, len(convos))
                )
                is_project = False
            elif file_type == FileType.CLAUDE_PROJECTS:
                with open(temp_path) as f:
                    projects = json.load(f)
                raw_texts = extract_facts_from_projects(
                    projects, max_projects=min(Config.MAX_PROJECTS, len(projects))
                )
                is_project = True
            else:  # CHATGPT_CONVERSATIONS
                with open(temp_path) as f:
                    convos = json.load(f)
                raw_texts = extract_facts_from_chatgpt(
                    convos, max_conversations=min(Config.MAX_CONVERSATIONS, len(convos))
                )
                is_project = False

            total = len(raw_texts)
            fact_type = FactType.PROJECT if is_project else FactType.GENERAL
            confidence = Config.HIGH_CONFIDENCE if is_project else Config.DEFAULT_CONFIDENCE

            yield f"data: {json.dumps({'stage': 'storing', 'pct': 60, 'msg': f'Storing {total} facts...', 'total': total})}\n\n"
            await asyncio.sleep(0)

            try:
                from anamnesis.consolidation.context_detector import FANTASY_KEYWORDS
                fiction_words = FANTASY_KEYWORDS
            except ImportError:
                fiction_words = set(Config.FICTION_KEYWORDS)

            memory = get_memory()
            stored = 0
            duplicates = 0
            fiction_count = 0
            fact_ids = []

            for i, fact_text in enumerate(raw_texts):
                if len(fact_text) <= Config.MIN_FACT_LENGTH:
                    continue
                if not is_project:
                    if bool(set(fact_text.lower().split()) & fiction_words):
                        fiction_count += 1
                        continue

                fact_metadata = {}
                try:
                    from anamnesis.topics.detector import TopicDetector
                    tr = TopicDetector().detect(fact_text)
                    fact_metadata["topic"] = tr.main_topic
                    fact_metadata["category"] = tr.category
                except Exception:
                    pass

                result = memory.semantic.add_fact(
                    content=fact_text,
                    fact_type=fact_type,
                    confidence=confidence,
                    metadata=fact_metadata if fact_metadata else None,
                )
                if result.confirmation_count > 1:
                    duplicates += 1
                else:
                    stored += 1
                    fact_ids.append(result.id)

                if (i + 1) % 25 == 0:
                    pct = 60 + int(((i + 1) / max(total, 1)) * 35)
                    yield f"data: {json.dumps({'stage': 'storing', 'pct': pct, 'msg': f'Stored {stored} facts...', 'stored': stored, 'i': i + 1, 'total': total})}\n\n"
                    await asyncio.sleep(0)

            import_history.record(file_type, stored, duplicates, safe_filename, fact_ids)

            yield f"data: {json.dumps({'stage': 'complete', 'pct': 100, 'msg': f'{stored} facts stored', 'stored': stored, 'extracted': total, 'duplicates': duplicates, 'fiction': fiction_count, 'file_type': file_type})}\n\n"

        except Exception as e:
            logger.error("SSE import failed: %s", e, exc_info=True)
            yield f"data: {json.dumps({'stage': 'error', 'pct': 0, 'msg': str(e)})}\n\n"
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink()

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/import/claude-memory", response_model=ImportResult)
async def import_claude_memory(file: UploadFile = File(...)):
    """Legacy endpoint — delegates to smart import which auto-detects."""
    return await smart_import(file)


@router.post("/import/chatgpt-export", response_model=ImportResult)
async def import_chatgpt(file: UploadFile = File(...)):
    """
    Dedicated ChatGPT import endpoint.

    The smart import handles this too, but this endpoint validates
    the ChatGPT list structure explicitly before processing.
    """
    temp_path = None
    try:
        if file.size and file.size > Config.MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (max {Config.MAX_UPLOAD_SIZE_MB}MB)"
            )

        safe_filename = sanitize_filename(file.filename)
        content = await file.read()
        temp_path = _save_upload_to_temp(content)

        is_valid, error_msg = validate_upload(temp_path)
        if not is_valid:
            raise ValidationError(error_msg)

        with open(temp_path) as f:
            conversations = json.load(f)

        if not isinstance(conversations, list):
            raise HTTPException(status_code=400, detail="Invalid ChatGPT format (expected list)")

        raw_facts = extract_facts_from_chatgpt(
            conversations,
            max_conversations=min(Config.MAX_CONVERSATIONS, len(conversations)),
        )
        batch = _store_facts_batch(raw_facts, FactType.GENERAL, Config.DEFAULT_CONFIDENCE)

        return ImportResult(
            success=True,
            facts_extracted=len(raw_facts),
            facts_stored=batch["stored"],
            fiction_filtered=batch["fiction_count"],
            duplicates_skipped=batch["duplicates"],
            file_type="chatgpt_conversations",
            message=f"Extracted {batch['stored']} facts from {len(conversations)} ChatGPT conversations",
        )

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e.msg}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("ChatGPT import failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Import failed")
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


@router.post("/import/facts-json")
async def import_facts_json(file: UploadFile = File(...)):
    """Import facts from a previously exported Velqua JSON file."""
    temp_path = None
    try:
        content = await file.read()
        temp_path = _save_upload_to_temp(content)

        is_valid, error_msg = validate_upload(temp_path)
        if not is_valid:
            raise ValidationError(error_msg)

        with open(temp_path) as f:
            data = json.load(f)

        facts_list = data.get("facts", data) if isinstance(data, dict) else data
        if not isinstance(facts_list, list):
            raise HTTPException(status_code=400, detail="Expected JSON with 'facts' array")

        memory = get_memory()
        stored = 0
        duplicates = 0
        for item in facts_list:
            fact_content = item.get("content", "") if isinstance(item, dict) else str(item)
            if not fact_content or len(fact_content) <= Config.MIN_FACT_LENGTH:
                continue

            confidence = Config.DEFAULT_CONFIDENCE
            if isinstance(item, dict):
                confidence = item.get("confidence", confidence)

            result = memory.semantic.add_fact(
                content=fact_content,
                fact_type=FactType.GENERAL,
                confidence=confidence,
            )
            if result.confirmation_count > 1:
                duplicates += 1
            else:
                stored += 1

        return ImportResult(
            success=True,
            facts_extracted=len(facts_list),
            facts_stored=stored,
            duplicates_skipped=duplicates,
            file_type="facts_json",
            message=f"Imported {stored} facts from export ({duplicates} duplicates skipped)",
        )

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Facts JSON import failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Import failed")
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
