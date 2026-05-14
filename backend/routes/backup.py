"""
Backup and export routes.

Handles database backup/restore and fact export/import as JSON.
"""
import shutil
from datetime import datetime

from fastapi import APIRouter, HTTPException

from anamnesis import Anamnesis
from backend.validators import sanitize_filename
from backend.config import VelquaConfig as Config
from backend.logging_config import get_logger
from backend.routes._shared import get_memory, init_shared

logger = get_logger("routes.backup")

router = APIRouter()


@router.post("/backup/create")
async def create_backup():
    """Copy the current database to data/backups/ with a timestamp."""
    try:
        backup_dir = Config.DATA_DIR / "backups"
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"velqua_backup_{timestamp}.db"

        shutil.copy2(str(Config.DB_PATH), str(backup_path))
        size_mb = backup_path.stat().st_size / (1024 * 1024)

        return {
            "success": True,
            "backup_path": str(backup_path),
            "size_mb": round(size_mb, 2),
            "timestamp": timestamp,
        }
    except Exception as e:
        logger.error("Backup failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Backup failed")


@router.get("/backup/list")
async def list_backups():
    """List all available backup files sorted newest-first."""
    backup_dir = Config.DATA_DIR / "backups"
    if not backup_dir.exists():
        return {"backups": []}

    backups = []
    for f in sorted(backup_dir.glob("velqua_backup_*.db"), reverse=True):
        backups.append({
            "filename": f.name,
            "path": str(f),
            "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
            "created": f.stat().st_mtime,
        })
    return {"backups": backups}


@router.post("/backup/restore/{filename}")
async def restore_backup(filename: str):
    """
    Restore the database from a backup file.

    Creates a safety backup of the current database before overwriting,
    so the user can always undo a bad restore.
    """
    backup_dir = Config.DATA_DIR / "backups"
    backup_path = backup_dir / sanitize_filename(filename)

    if not backup_path.exists():
        raise HTTPException(status_code=404, detail="Backup not found")

    try:
        # Safety net: snapshot current state before we overwrite
        safety = Config.DATA_DIR / "velqua_pre_restore.db"
        shutil.copy2(str(Config.DB_PATH), str(safety))

        shutil.copy2(str(backup_path), str(Config.DB_PATH))

        # Reinitialize the shared memory instance with the restored DB
        new_memory = Anamnesis(str(Config.DB_PATH))
        init_shared(new_memory)

        return {
            "success": True,
            "restored_from": filename,
            "safety_backup": str(safety),
        }
    except Exception as e:
        logger.error("Restore failed for %s: %s", filename, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Restore failed")


@router.get("/export/facts")
async def export_facts():
    """Export all facts as a JSON document for backup or migration."""
    memory = get_memory()
    try:
        all_facts = memory.semantic.list_all(limit=Config.MAX_FACTS_LIST)
        def _meta(f, key, default=""):
            return f.metadata.get(key, default) if hasattr(f, "metadata") and f.metadata else default

        exported = [
            {
                "content": f.content,
                "type": str(f.fact_type),
                "confidence": f.confidence,
                "confirmation_count": getattr(f, "confirmation_count", 1),
                "tags": _meta(f, "tags", []),
                "topic": _meta(f, "topic"),
                "category": _meta(f, "category"),
                "emotion": _meta(f, "emotion"),
                "sentiment_score": _meta(f, "sentiment_score", 0),
                "created_at": (f.created_at.isoformat()
                               if hasattr(f, "created_at") and f.created_at else None),
            }
            for f in all_facts
        ]
        return {
            "facts": exported,
            "count": len(exported),
            "exported_at": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error("Export failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Export failed")
