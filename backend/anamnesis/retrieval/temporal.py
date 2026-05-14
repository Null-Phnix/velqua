"""
Temporal retrieval for time-based memory queries.

Handles queries like:
- "What did we discuss last week?"
- "Conversations from October"
- "Recent memories"
"""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from ..models import Episode
from ..stores.episodic import EpisodicStore


@dataclass
class TimeRange:
    """A time range for querying."""
    start: datetime
    end: datetime
    description: str


class TemporalRetriever:
    """
    Retrieves memories based on temporal queries.

    Supports:
    - Relative time ("last week", "yesterday", "2 months ago")
    - Absolute time ("October 2025", "January 15th")
    - Ranges ("between X and Y")
    """

    # Patterns for parsing temporal expressions
    RELATIVE_PATTERNS = [
        # Days
        (r"today", lambda: (datetime.now().replace(hour=0, minute=0, second=0), datetime.now())),
        (r"yesterday", lambda: (
            datetime.now().replace(hour=0, minute=0, second=0) - timedelta(days=1),
            datetime.now().replace(hour=0, minute=0, second=0)
        )),
        (r"(\d+)\s*days?\s*ago", lambda m: (
            datetime.now() - timedelta(days=int(m.group(1))),
            datetime.now() - timedelta(days=int(m.group(1)) - 1)
        )),
        # Weeks
        (r"this\s*week", lambda: (
            datetime.now() - timedelta(days=datetime.now().weekday()),
            datetime.now()
        )),
        (r"last\s*week", lambda: (
            datetime.now() - timedelta(days=datetime.now().weekday() + 7),
            datetime.now() - timedelta(days=datetime.now().weekday())
        )),
        (r"(\d+)\s*weeks?\s*ago", lambda m: (
            datetime.now() - timedelta(weeks=int(m.group(1))),
            datetime.now() - timedelta(weeks=int(m.group(1)) - 1)
        )),
        # Months
        (r"this\s*month", lambda: (
            datetime.now().replace(day=1, hour=0, minute=0, second=0),
            datetime.now()
        )),
        (r"last\s*month", lambda: (
            (datetime.now().replace(day=1) - timedelta(days=1)).replace(day=1),
            datetime.now().replace(day=1) - timedelta(days=1)
        )),
        (r"(\d+)\s*months?\s*ago", lambda m: (
            datetime.now() - timedelta(days=30 * int(m.group(1))),
            datetime.now() - timedelta(days=30 * (int(m.group(1)) - 1))
        )),
        # Recent
        (r"recent(?:ly)?", lambda: (datetime.now() - timedelta(days=7), datetime.now())),
        (r"earlier", lambda: (datetime.now() - timedelta(days=30), datetime.now())),
    ]

    # Month names for parsing
    MONTH_NAMES = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12,
    }

    def __init__(self, episodic_store: EpisodicStore):
        self.episodic_store = episodic_store

    def parse_temporal_query(self, query: str) -> Optional[TimeRange]:
        """
        Parse a temporal expression into a time range.

        Args:
            query: Natural language time expression

        Returns:
            TimeRange if parseable, None otherwise
        """
        query_lower = query.lower().strip()

        # Try relative patterns
        for pattern, resolver in self.RELATIVE_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                try:
                    # Try with match first (for patterns with groups)
                    start, end = resolver(match)
                except TypeError:
                    # Fall back to no args (for simple patterns)
                    start, end = resolver()
                return TimeRange(start=start, end=end, description=query)

        # Try month + year pattern (e.g., "October 2025")
        month_year = re.search(
            r"(\w+)\s*(\d{4})",
            query_lower
        )
        if month_year:
            month_name = month_year.group(1)
            year = int(month_year.group(2))
            month = self.MONTH_NAMES.get(month_name)
            if month:
                start = datetime(year, month, 1)
                if month == 12:
                    end = datetime(year + 1, 1, 1)
                else:
                    end = datetime(year, month + 1, 1)
                return TimeRange(start=start, end=end, description=query)

        # Try just month name (assume current/recent year)
        for month_name, month_num in self.MONTH_NAMES.items():
            if month_name in query_lower:
                now = datetime.now()
                year = now.year
                # If month is in the future, use last year
                if month_num > now.month:
                    year -= 1
                start = datetime(year, month_num, 1)
                if month_num == 12:
                    end = datetime(year + 1, 1, 1)
                else:
                    end = datetime(year, month_num + 1, 1)
                return TimeRange(start=start, end=end, description=query)

        return None

    def search(
        self,
        query: str,
        limit: int = 20,
        fallback_days: int = 30,
    ) -> Tuple[List[Episode], Optional[TimeRange]]:
        """
        Search for episodes in a time range.

        Args:
            query: Temporal query string
            limit: Maximum results
            fallback_days: Days to search if no time expression found

        Returns:
            Tuple of (episodes, parsed_time_range)
        """
        time_range = self.parse_temporal_query(query)

        if time_range:
            episodes = self._get_episodes_in_range(
                time_range.start,
                time_range.end,
                limit,
            )
        else:
            # Fallback to recent
            episodes = self.episodic_store.get_recent(
                days=fallback_days,
                limit=limit,
            )
            time_range = TimeRange(
                start=datetime.now() - timedelta(days=fallback_days),
                end=datetime.now(),
                description=f"last {fallback_days} days",
            )

        return episodes, time_range

    def _get_episodes_in_range(
        self,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> List[Episode]:
        """Get episodes within a time range."""
        all_episodes = self.episodic_store.list_all(limit=1000)

        matching = []
        for ep in all_episodes:
            if ep.started_at:
                if start <= ep.started_at <= end:
                    matching.append(ep)

        # Sort by date (most recent first)
        matching.sort(key=lambda e: e.started_at or datetime.min, reverse=True)

        return matching[:limit]

    def get_timeline(
        self,
        days: int = 30,
        group_by: str = "day",
    ) -> Dict[str, List[Episode]]:
        """
        Get a timeline view of memories.

        Args:
            days: Number of days to include
            group_by: "day", "week", or "month"

        Returns:
            Dict mapping time period to episodes
        """
        end = datetime.now()
        start = end - timedelta(days=days)

        episodes = self._get_episodes_in_range(start, end, limit=500)

        timeline = {}

        for ep in episodes:
            if not ep.started_at:
                continue

            if group_by == "day":
                key = ep.started_at.strftime("%Y-%m-%d")
            elif group_by == "week":
                # ISO week
                key = ep.started_at.strftime("%Y-W%W")
            elif group_by == "month":
                key = ep.started_at.strftime("%Y-%m")
            else:
                key = ep.started_at.strftime("%Y-%m-%d")

            if key not in timeline:
                timeline[key] = []
            timeline[key].append(ep)

        return timeline

    def find_anniversary_memories(
        self,
        tolerance_days: int = 3,
    ) -> List[Tuple[Episode, int]]:
        """
        Find memories from the same time in previous years.

        Useful for "on this day" features.

        Returns:
            List of (episode, years_ago) tuples
        """
        now = datetime.now()
        all_episodes = self.episodic_store.list_all(limit=1000)

        anniversary_memories = []

        for ep in all_episodes:
            if not ep.started_at:
                continue

            # Check if same day (or within tolerance) in a previous year
            years_diff = now.year - ep.started_at.year
            if years_diff <= 0:
                continue

            # Create a date in current year with episode's month/day
            try:
                anniversary_date = ep.started_at.replace(year=now.year)
            except ValueError:
                # Handle Feb 29 in non-leap years
                continue

            days_diff = abs((now - anniversary_date).days)

            if days_diff <= tolerance_days:
                anniversary_memories.append((ep, years_diff))

        # Sort by years ago
        anniversary_memories.sort(key=lambda x: x[1])

        return anniversary_memories
