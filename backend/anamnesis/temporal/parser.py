"""
Temporal expression parser.

Parses natural language time expressions into concrete date ranges.
Examples:
- "last week"
- "yesterday"
- "a few days ago"
- "last month"
- "when we first met"
- "before we discussed Python"
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional


class RelativeTime(Enum):
    """Relative time indicators."""
    TODAY = "today"
    YESTERDAY = "yesterday"
    THIS_WEEK = "this_week"
    LAST_WEEK = "last_week"
    THIS_MONTH = "this_month"
    LAST_MONTH = "last_month"
    RECENT = "recent"  # ~3 days
    A_WHILE_AGO = "a_while_ago"  # ~2 weeks
    LONG_AGO = "long_ago"  # ~1 month+


@dataclass
class TemporalRange:
    """A concrete date/time range."""
    start: Optional[datetime] = None
    end: Optional[datetime] = None

    @property
    def is_bounded(self) -> bool:
        """Check if range has at least one bound."""
        return self.start is not None or self.end is not None

    @property
    def duration_days(self) -> Optional[float]:
        """Get duration in days if both bounds set."""
        if self.start and self.end:
            return (self.end - self.start).total_seconds() / 86400
        return None

    def contains(self, dt: datetime) -> bool:
        """Check if datetime is within range."""
        if self.start and dt < self.start:
            return False
        if self.end and dt > self.end:
            return False
        return True

    @classmethod
    def from_relative(cls, relative: RelativeTime, reference: Optional[datetime] = None) -> "TemporalRange":
        """Create range from relative time indicator."""
        now = reference or datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if relative == RelativeTime.TODAY:
            return cls(start=today_start, end=now)

        elif relative == RelativeTime.YESTERDAY:
            yesterday = today_start - timedelta(days=1)
            return cls(start=yesterday, end=today_start)

        elif relative == RelativeTime.THIS_WEEK:
            week_start = today_start - timedelta(days=today_start.weekday())
            return cls(start=week_start, end=now)

        elif relative == RelativeTime.LAST_WEEK:
            this_week_start = today_start - timedelta(days=today_start.weekday())
            last_week_start = this_week_start - timedelta(days=7)
            return cls(start=last_week_start, end=this_week_start)

        elif relative == RelativeTime.THIS_MONTH:
            month_start = today_start.replace(day=1)
            return cls(start=month_start, end=now)

        elif relative == RelativeTime.LAST_MONTH:
            month_start = today_start.replace(day=1)
            last_month_end = month_start - timedelta(days=1)
            last_month_start = last_month_end.replace(day=1)
            return cls(start=last_month_start, end=month_start)

        elif relative == RelativeTime.RECENT:
            return cls(start=now - timedelta(days=3), end=now)

        elif relative == RelativeTime.A_WHILE_AGO:
            return cls(start=now - timedelta(days=21), end=now - timedelta(days=7))

        elif relative == RelativeTime.LONG_AGO:
            return cls(start=None, end=now - timedelta(days=30))

        return cls()


@dataclass
class TemporalExpression:
    """A parsed temporal expression."""
    original: str
    range: TemporalRange
    relative: Optional[RelativeTime] = None
    reference_topic: Optional[str] = None  # "before we talked about X"
    confidence: float = 1.0
    keywords: List[str] = field(default_factory=list)


class TemporalParser:
    """
    Parser for natural language temporal expressions.

    Converts phrases like "last week" or "a few days ago" into
    concrete date ranges for memory filtering.
    """

    # Pattern definitions
    PATTERNS = {
        # Specific times
        r'\btoday\b': (RelativeTime.TODAY, 1.0),
        r'\byesterday\b': (RelativeTime.YESTERDAY, 1.0),

        # Week patterns
        r'\bthis week\b': (RelativeTime.THIS_WEEK, 1.0),
        r'\blast week\b': (RelativeTime.LAST_WEEK, 1.0),
        r'\bpast week\b': (RelativeTime.LAST_WEEK, 0.9),
        r'\bover the (past |last )?week\b': (RelativeTime.LAST_WEEK, 0.9),

        # Month patterns
        r'\bthis month\b': (RelativeTime.THIS_MONTH, 1.0),
        r'\blast month\b': (RelativeTime.LAST_MONTH, 1.0),
        r'\bpast month\b': (RelativeTime.LAST_MONTH, 0.9),

        # Relative phrases
        r'\brecently\b': (RelativeTime.RECENT, 0.9),
        r'\ba (few|couple( of)?) days ago\b': (RelativeTime.RECENT, 0.9),
        r'\bnot long ago\b': (RelativeTime.RECENT, 0.8),
        r'\bthe other day\b': (RelativeTime.RECENT, 0.8),
        r'\bjust (the other day|recently)\b': (RelativeTime.RECENT, 0.9),

        # Longer periods
        r'\ba while (ago|back)\b': (RelativeTime.A_WHILE_AGO, 0.8),
        r'\bsome time (ago|back)\b': (RelativeTime.A_WHILE_AGO, 0.8),
        r'\ba (few|couple( of)?) weeks ago\b': (RelativeTime.A_WHILE_AGO, 0.9),

        # Very long ago
        r'\blong (time )?ago\b': (RelativeTime.LONG_AGO, 0.8),
        r'\bmonths? ago\b': (RelativeTime.LONG_AGO, 0.9),
        r'\bages ago\b': (RelativeTime.LONG_AGO, 0.7),
        r'\bway back\b': (RelativeTime.LONG_AGO, 0.7),
    }

    # Reference patterns (before/after topic)
    REFERENCE_PATTERNS = [
        r'\bbefore (?:we )?(?:talked|discussed|mentioned) (?:about )?(.+?)(?:\.|,|$)',
        r'\bafter (?:we )?(?:talked|discussed|mentioned) (?:about )?(.+?)(?:\.|,|$)',
        r'\bwhen (?:we )?(?:talked|discussed|mentioned) (?:about )?(.+?)(?:\.|,|$)',
        r'\bremember (?:when (?:we )?)?(?:talked|discussed|mentioned) (?:about )?(.+?)(?:\.|,|$)',
    ]

    # Day patterns
    DAY_PATTERNS = [
        (r'\b(\d+) days? ago\b', lambda m: timedelta(days=int(m.group(1)))),
        (r'\b(\d+) weeks? ago\b', lambda m: timedelta(weeks=int(m.group(1)))),
        (r'\b(\d+) months? ago\b', lambda m: timedelta(days=int(m.group(1)) * 30)),
    ]

    def __init__(self, reference_time: Optional[datetime] = None):
        """
        Initialize parser.

        Args:
            reference_time: Reference datetime for relative calculations
        """
        self.reference_time = reference_time

    @property
    def now(self) -> datetime:
        """Get reference time or current time."""
        return self.reference_time or datetime.now()

    def parse(self, text: str) -> Optional[TemporalExpression]:
        """
        Parse a temporal expression from text.

        Args:
            text: Text containing temporal expression

        Returns:
            TemporalExpression or None if no temporal found
        """
        text_lower = text.lower()
        best_match = None
        best_confidence = 0.0

        # Try specific day patterns first
        for pattern, delta_func in self.DAY_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                delta = delta_func(match)
                target_date = self.now - delta
                # Create range around target date
                range_obj = TemporalRange(
                    start=target_date - timedelta(hours=12),
                    end=target_date + timedelta(hours=12),
                )
                return TemporalExpression(
                    original=match.group(0),
                    range=range_obj,
                    confidence=1.0,
                    keywords=[match.group(0)],
                )

        # Try relative patterns
        for pattern, (relative, confidence) in self.PATTERNS.items():
            match = re.search(pattern, text_lower)
            if match and confidence > best_confidence:
                best_match = (match, relative, confidence)
                best_confidence = confidence

        if best_match:
            match, relative, confidence = best_match
            range_obj = TemporalRange.from_relative(relative, self.now)

            return TemporalExpression(
                original=match.group(0),
                range=range_obj,
                relative=relative,
                confidence=confidence,
                keywords=[match.group(0)],
            )

        # Try reference patterns
        for pattern in self.REFERENCE_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                topic = match.group(1).strip()
                return TemporalExpression(
                    original=match.group(0),
                    range=TemporalRange(),  # No concrete range
                    reference_topic=topic,
                    confidence=0.9,
                    keywords=[match.group(0), topic],
                )

        return None

    def extract_all(self, text: str) -> List[TemporalExpression]:
        """
        Extract all temporal expressions from text.

        Args:
            text: Text to parse

        Returns:
            List of temporal expressions found
        """
        results = []
        text_lower = text.lower()

        # Day patterns
        for pattern, delta_func in self.DAY_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                delta = delta_func(match)
                target_date = self.now - delta
                range_obj = TemporalRange(
                    start=target_date - timedelta(hours=12),
                    end=target_date + timedelta(hours=12),
                )
                results.append(TemporalExpression(
                    original=match.group(0),
                    range=range_obj,
                    confidence=1.0,
                    keywords=[match.group(0)],
                ))

        # Relative patterns
        for pattern, (relative, confidence) in self.PATTERNS.items():
            for match in re.finditer(pattern, text_lower):
                range_obj = TemporalRange.from_relative(relative, self.now)
                results.append(TemporalExpression(
                    original=match.group(0),
                    range=range_obj,
                    relative=relative,
                    confidence=confidence,
                    keywords=[match.group(0)],
                ))

        # Reference patterns
        for pattern in self.REFERENCE_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                topic = match.group(1).strip()
                results.append(TemporalExpression(
                    original=match.group(0),
                    range=TemporalRange(),
                    reference_topic=topic,
                    confidence=0.9,
                    keywords=[match.group(0), topic],
                ))

        return results

    def has_temporal(self, text: str) -> bool:
        """
        Check if text contains a temporal expression.

        Args:
            text: Text to check

        Returns:
            True if temporal expression found
        """
        return self.parse(text) is not None

    def remove_temporal(self, text: str) -> str:
        """
        Remove temporal expressions from text.

        Useful for getting the "core" query without time qualifiers.

        Args:
            text: Text to process

        Returns:
            Text with temporal expressions removed
        """
        result = text

        # Remove day patterns
        for pattern, _ in self.DAY_PATTERNS:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)

        # Remove relative patterns
        for pattern in self.PATTERNS.keys():
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)

        # Remove reference patterns
        for pattern in self.REFERENCE_PATTERNS:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)

        # Clean up whitespace
        result = re.sub(r'\s+', ' ', result).strip()

        return result
