"""
Shared CLI utilities and context factory.
"""

from dataclasses import dataclass

from ..models import EmotionalValence
from ..stores.episodic import EpisodicStore
from ..stores.semantic import SemanticStore
from ..stores.sqlite_backend import SQLiteBackend


@dataclass
class CommandContext:
    """Shared context for CLI commands, eliminating repeated setup boilerplate."""
    backend: SQLiteBackend
    episodic: EpisodicStore
    semantic: SemanticStore


def make_context(args) -> CommandContext:
    """Create a CommandContext from parsed args."""
    backend = SQLiteBackend(args.db)
    return CommandContext(
        backend=backend,
        episodic=EpisodicStore(backend),
        semantic=SemanticStore(backend),
    )


def print_header(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def parse_emotion(emotion_str: str) -> EmotionalValence:
    """Parse emotion string to EmotionalValence."""
    mapping = {
        'very_positive': EmotionalValence.VERY_POSITIVE,
        'positive': EmotionalValence.POSITIVE,
        'neutral': EmotionalValence.NEUTRAL,
        'negative': EmotionalValence.NEGATIVE,
        'very_negative': EmotionalValence.VERY_NEGATIVE,
        # Aliases
        'happy': EmotionalValence.POSITIVE,
        'sad': EmotionalValence.NEGATIVE,
        'excited': EmotionalValence.VERY_POSITIVE,
        'frustrated': EmotionalValence.NEGATIVE,
        'angry': EmotionalValence.VERY_NEGATIVE,
    }
    return mapping.get(emotion_str.lower())


def valence_to_str(valence: EmotionalValence) -> str:
    """Convert valence to display string."""
    symbols = {
        EmotionalValence.VERY_POSITIVE: "++",
        EmotionalValence.POSITIVE: "+",
        EmotionalValence.NEUTRAL: "~",
        EmotionalValence.NEGATIVE: "-",
        EmotionalValence.VERY_NEGATIVE: "--",
    }
    return symbols.get(valence, "~")


def print_health_summary(health):
    """Print a health summary."""
    total = health["total_episodes"]
    healthy_pct = (health["healthy"] / total * 100) if total > 0 else 0
    at_risk_pct = (health["at_risk"] / total * 100) if total > 0 else 0

    print(f"  Episodes: {total}")
    print(f"    Healthy: {health['healthy']} ({healthy_pct:.1f}%)")
    print(f"    Aging: {health['aging']}")
    print(f"    At Risk: {health['at_risk']} ({at_risk_pct:.1f}%)")
    print(f"    Forgotten: {health['forgotten']}")
    print(f"  Facts: {health['total_facts']}")
    print(f"    High confidence: {health['high_confidence_facts']}")
    print(f"    Low confidence: {health['low_confidence_facts']}")
