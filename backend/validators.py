"""
Input validation for file uploads.
"""
from pathlib import Path
from typing import Tuple
import json
import re
from backend.config import VelquaConfig as Config


class ValidationError(Exception):
    """File validation failed."""
    pass


def validate_upload(file_path: Path, max_size_mb: int = None) -> Tuple[bool, str]:
    """
    Validate uploaded file.

    Args:
        file_path: Path to uploaded file
        max_size_mb: Maximum file size in MB

    Returns:
        (is_valid, error_message) - error_message is empty if valid
    """
    # Use default from config if not specified
    if max_size_mb is None:
        max_size_mb = Config.MAX_UPLOAD_SIZE_MB

    # Check file exists
    if not file_path.exists():
        return False, "File not found"

    # Check file size
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"File too large: {size_mb:.1f}MB (max {max_size_mb}MB)"

    # Check valid JSON
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e.msg}"
    except UnicodeDecodeError:
        return False, "File is not valid UTF-8 text"

    return True, ""


def sanitize_filename(filename: str) -> str:
    """
    Remove path traversal attempts and dangerous characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for filesystem use
    """
    # Remove directory separators
    filename = filename.replace('/', '').replace('\\', '')

    # Remove null bytes
    filename = filename.replace('\x00', '')

    # Remove path traversal attempts (.. becomes empty)
    filename = filename.replace('..', '')

    # Keep only alphanumeric, dash, underscore, dot
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)

    # Remove leading/trailing underscores from substitution
    filename = filename.strip('_')

    # Ensure it's not empty or just dots/underscores
    if not filename or filename.replace('.', '').replace('_', '') == '':
        filename = "upload.json"

    return filename
