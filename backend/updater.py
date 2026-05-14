"""
Auto-update version checker.

Checks for new Velqua releases via GitHub API (or configurable URL).
No auto-install — only surfaces a notification in the Settings tab.
"""
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

from backend import __version__

logger = logging.getLogger(__name__)

# Default check URL — GitHub releases API
RELEASES_URL = "https://api.github.com/repos/velqua/velqua/releases/latest"
CHECK_TIMEOUT = 10.0  # seconds


@dataclass
class UpdateInfo:
    """Result of an update check."""
    update_available: bool
    current_version: str
    latest_version: str
    release_url: str = ""
    release_notes: str = ""
    error: str = ""


def _parse_version(v: str) -> tuple:
    """Parse a version string into comparable tuple. Handles 2.0.0-alpha.2 etc."""
    # Strip leading 'v'
    v = v.lstrip("v")
    # Split on '-' for pre-release tags
    parts = v.split("-", 1)
    base = parts[0]
    pre = parts[1] if len(parts) > 1 else ""

    # Parse base version numbers
    nums = []
    for part in base.split("."):
        try:
            nums.append(int(part))
        except ValueError:
            nums.append(0)

    # Pre-release sorts before release: alpha < beta < rc < (empty = release)
    pre_order = 4  # release
    if pre:
        pre_lower = pre.lower()
        if "alpha" in pre_lower or pre_lower.startswith("a"):
            pre_order = 1
        elif "beta" in pre_lower or pre_lower.startswith("b"):
            pre_order = 2
        elif "rc" in pre_lower:
            pre_order = 3

    return (*nums, pre_order)


def is_newer(latest: str, current: str) -> bool:
    """Return True if latest version is newer than current."""
    try:
        return _parse_version(latest) > _parse_version(current)
    except Exception:
        return False


async def check_for_updates(url: Optional[str] = None) -> UpdateInfo:
    """
    Check for a newer version of Velqua.

    Queries the GitHub releases API (or custom URL) and compares
    the latest tag against the running version.
    """
    current = __version__
    check_url = url or RELEASES_URL

    try:
        async with httpx.AsyncClient(timeout=CHECK_TIMEOUT) as client:
            response = await client.get(
                check_url,
                headers={"Accept": "application/vnd.github.v3+json"},
            )

        if response.status_code != 200:
            return UpdateInfo(
                update_available=False,
                current_version=current,
                latest_version=current,
                error=f"GitHub API returned {response.status_code}",
            )

        data = response.json()
        latest_tag = data.get("tag_name", "").lstrip("v")
        html_url = data.get("html_url", "")
        body = data.get("body", "")

        if not latest_tag:
            return UpdateInfo(
                update_available=False,
                current_version=current,
                latest_version=current,
                error="No tag found in release data",
            )

        available = is_newer(latest_tag, current)

        return UpdateInfo(
            update_available=available,
            current_version=current,
            latest_version=latest_tag,
            release_url=html_url,
            release_notes=body[:500] if body else "",
        )

    except httpx.ConnectError:
        return UpdateInfo(
            update_available=False,
            current_version=current,
            latest_version=current,
            error="Could not reach update server (no internet?)",
        )
    except Exception as e:
        logger.warning("Update check failed: %s", e)
        return UpdateInfo(
            update_available=False,
            current_version=current,
            latest_version=current,
            error=str(e),
        )
