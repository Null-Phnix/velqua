"""
Auto-learning module for Velqua proxy.

Extracts facts from user messages during live conversations,
scores them for quality, and holds them for user review before
committing to the knowledge base.
"""
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Dict

from backend.config import VelquaConfig as Config
from backend.logging_config import get_logger

logger = get_logger("auto_learner")

# Self-disclosure patterns that indicate personal facts
FACT_MARKERS = [
    "i am", "i'm", "i work", "i live", "my name",
    "i like", "i love", "i hate", "i want", "i need",
    "i have", "i've been", "i'm working on", "i study",
    "i prefer", "i usually", "my job", "my hobby",
    "i moved", "i grew up", "i speak", "i play",
    "i own", "i drive", "i built", "i created",
]

# Transient markers — these indicate temporary state, not permanent facts
TRANSIENT_MARKERS = [
    "i'm debugging", "i'm trying to", "i'm getting an error",
    "i'm testing", "i'm looking at", "i'm reading",
    "i'm stuck on", "i'm confused about", "i need help with",
    "i'm running", "i just ran", "i'm checking",
    "i'm writing a test", "i'm fixing", "i'm updating",
]

# High-value markers — permanent personal facts
HIGH_VALUE_MARKERS = [
    "my name", "i live", "i moved", "i grew up",
    "i work", "my job", "i speak", "i study",
    "i own", "i drive",
]

# Assistant confirmation markers — second-person phrases that echo user facts back.
# When the model says "you work at DataForge", that confirms the fact "I work at DataForge".
ASSISTANT_FACT_MARKERS = [
    "you are", "you're a", "you're the", "you work", "you live",
    "you have", "you've been", "you moved", "you grew up",
    "you study", "you own", "you drive", "you speak",
    "as a ", "since you ", "given that you ", "because you ",
    "you mentioned", "you said you", "you told me",
    "your name is", "your job", "your background",
]

# Substitutions to convert assistant second-person to first-person for storage
_SECOND_TO_FIRST = [
    ("You are ", "I am "), ("you are ", "I am "),
    ("You're ", "I'm "), ("you're ", "I'm "),
    ("You work ", "I work "), ("you work ", "I work "),
    ("You live ", "I live "), ("you live ", "I live "),
    ("You have ", "I have "), ("you have ", "I have "),
    ("You've ", "I've "), ("you've ", "I've "),
    ("You moved ", "I moved "), ("you moved ", "I moved "),
    ("You grew ", "I grew "), ("you grew ", "I grew "),
    ("You study ", "I study "), ("you study ", "I study "),
    ("You own ", "I own "), ("you own ", "I own "),
    ("You drive ", "I drive "), ("you drive ", "I drive "),
    ("You speak ", "I speak "), ("you speak ", "I speak "),
    ("Your name is", "My name is"), ("your name is", "my name is"),
    ("Your job ", "My job "), ("your job ", "my job "),
]


def _to_first_person(text: str) -> str:
    """Convert a second-person assistant confirmation to first-person."""
    for old, new in _SECOND_TO_FIRST:
        if old in text:
            return text.replace(old, new, 1)
    return text


def extract_facts_from_assistant(text: str) -> List[str]:
    """
    Extract user facts from assistant confirmation phrases.

    When the model echoes "you work at DataForge" or "since you live in Vancouver",
    that confirms a user fact. Converts to first-person before returning.
    Returns list of first-person fact strings.
    """
    if not text or len(text) < 15:
        return []

    facts = []
    text_lower = text.lower()

    for marker in ASSISTANT_FACT_MARKERS:
        if marker in text_lower:
            sentences = text.split(". ")
            for sentence in sentences:
                if marker in sentence.lower():
                    cleaned = sentence.strip().rstrip(".,!?")
                    if Config.MIN_FACT_LENGTH < len(cleaned) < Config.MAX_FACT_LENGTH:
                        first_person = _to_first_person(cleaned)
                        facts.append(first_person)
                    break  # One fact per marker

    return facts


def score_fact_quality(fact_text: str) -> float:
    """
    Score a fact for quality and permanence.

    Returns 0.0-1.0 where higher = more likely to be a real, permanent fact.
    Below 0.4 = filtered out. 0.4-0.7 = pending review. 0.7+ = auto-accept.
    """
    text_lower = fact_text.lower()
    score = 0.5  # Base score

    # Boost for high-value markers (permanent personal facts)
    if any(m in text_lower for m in HIGH_VALUE_MARKERS):
        score += 0.25

    # Penalty for transient markers (temporary state)
    if any(m in text_lower for m in TRANSIENT_MARKERS):
        score -= 0.3

    # Penalty for code-like content
    code_indicators = ["()", "=>", "def ", "class ", "import ", "{", "}", "```"]
    if any(ind in fact_text for ind in code_indicators):
        score -= 0.2

    # Penalty for questions
    if fact_text.strip().endswith("?"):
        score -= 0.3

    # Boost for specificity (contains proper nouns — capitalized words)
    words = fact_text.split()
    proper_nouns = sum(1 for w in words[1:] if w[0].isupper() and w.isalpha())
    if proper_nouns >= 1:
        score += 0.1

    # Penalty for very generic statements
    generic = ["i have a question", "i need help", "i want to know", "i am trying"]
    if any(g in text_lower for g in generic):
        score -= 0.2

    # Length sweet spot: 30-150 chars is ideal for a fact
    if 30 <= len(fact_text) <= 150:
        score += 0.05
    elif len(fact_text) > 300:
        score -= 0.1

    return max(0.0, min(1.0, score))


def extract_facts_from_text(text: str) -> List[str]:
    """
    Extract self-disclosure facts from a single user message.

    Returns list of fact strings found in the text.
    """
    if not text or len(text) < 15:
        return []

    facts = []
    text_lower = text.lower()

    # Use Anamnesis fantasy keywords for richer fiction filtering
    try:
        from anamnesis.consolidation.context_detector import FANTASY_KEYWORDS
        fiction_words = FANTASY_KEYWORDS
    except ImportError:
        fiction_words = set(Config.FICTION_KEYWORDS)

    for marker in FACT_MARKERS:
        if marker in text_lower:
            # Extract the sentence containing the marker
            sentences = text.split(". ")
            for sentence in sentences:
                if marker in sentence.lower():
                    cleaned = sentence.strip().rstrip(".")
                    if Config.MIN_FACT_LENGTH < len(cleaned) < Config.MAX_FACT_LENGTH:
                        # Check fiction keywords (word-boundary matching)
                        cleaned_words = set(cleaned.lower().split())
                        is_fiction = bool(cleaned_words & fiction_words)
                        if not is_fiction:
                            facts.append(cleaned)
                    break  # One fact per marker per message

    return facts


class PendingFactStore:
    """Stores pending facts as a JSON file for review."""

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Config.DATA_DIR
        self.data_dir.mkdir(exist_ok=True)
        self.file_path = self.data_dir / "pending_facts.json"
        self._pending = self._load()

    def _load(self) -> List[Dict]:
        if self.file_path.exists():
            try:
                with open(self.file_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save(self):
        # Atomic write: temp file + rename prevents corruption on crash
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.data_dir), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._pending, f, indent=2)
            os.replace(tmp_path, str(self.file_path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def add(self, content: str, quality_score: float, source: str = "proxy"):
        """Add a fact to pending review, enriched with topic + sentiment detection."""
        self._pending = self._load()  # Re-read for cross-process coherency

        # Detect topic
        detected_topic = ""
        detected_category = ""
        try:
            from anamnesis.topics.detector import TopicDetector
            topic_result = TopicDetector().detect(content)
            detected_topic = topic_result.main_topic
            detected_category = topic_result.category
        except Exception:
            pass

        # Detect emotion
        detected_emotion = ""
        try:
            from anamnesis.emotional.analyzer import SentimentAnalyzer
            sentiment = SentimentAnalyzer().analyze(content)
            detected_emotion = sentiment.primary_emotion.value
        except Exception:
            pass

        entry = {
            "id": f"pending-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}",
            "content": content,
            "quality_score": round(quality_score, 3),
            "source": source,
            "timestamp": time.time(),
            "detected_topic": detected_topic,
            "detected_category": detected_category,
            "detected_emotion": detected_emotion,
        }
        self._pending.append(entry)
        self._save()
        return entry

    def list_all(self) -> List[Dict]:
        """Get all pending facts (re-reads from disk for cross-process coherency)."""
        self._pending = self._load()
        return list(self._pending)

    def approve(self, pending_id: str) -> Dict:
        """Remove a fact from pending and return it for storage."""
        self._pending = self._load()  # Re-read for cross-process coherency
        for i, entry in enumerate(self._pending):
            if entry["id"] == pending_id:
                approved = self._pending.pop(i)
                self._save()
                return approved
        return None

    def reject(self, pending_id: str) -> bool:
        """Remove a fact from pending without storing it."""
        self._pending = self._load()  # Re-read for cross-process coherency
        for i, entry in enumerate(self._pending):
            if entry["id"] == pending_id:
                self._pending.pop(i)
                self._save()
                return True
        return False

    def approve_all(self) -> List[Dict]:
        """Approve all pending facts."""
        self._pending = self._load()  # Re-read for cross-process coherency
        approved = list(self._pending)
        self._pending = []
        self._save()
        return approved

    def reject_all(self) -> int:
        """Reject all pending facts."""
        self._pending = self._load()  # Re-read for cross-process coherency
        count = len(self._pending)
        self._pending = []
        self._save()
        return count

    def count(self) -> int:
        self._pending = self._load()
        return len(self._pending)


# Quality thresholds
QUALITY_AUTO_ACCEPT = 0.7  # High quality — store immediately
QUALITY_MIN_THRESHOLD = 0.4  # Below this — discard silently


class AutoLearner:
    """Learns facts from proxy conversations in the background."""

    def __init__(self, memory, retriever=None, pending_store: PendingFactStore = None):
        self.memory = memory
        self.retriever = retriever  # For vector indexing new facts
        self.pending = pending_store or PendingFactStore()
        self.facts_learned = 0
        self.facts_pending = 0
        self.facts_rejected = 0
        self.duplicates_seen = 0
        self.contradictions_found = 0
        self.enabled = True
        self.auto_approve = False  # When True, skip review queue

    async def learn_from_message(self, text: str, source: str = "proxy"):
        """
        Extract and store facts from a user message.

        High-quality facts (>0.7) are stored immediately.
        Medium-quality facts (0.4-0.7) go to pending review.
        Low-quality facts (<0.4) are silently discarded.
        """
        if not self.enabled:
            return

        try:
            from anamnesis.models import FactType

            facts = extract_facts_from_text(text)

            for fact_text in facts:
                quality = score_fact_quality(fact_text)

                if quality < QUALITY_MIN_THRESHOLD:
                    # Too low quality — discard
                    self.facts_rejected += 1
                    logger.debug("Rejected low-quality fact (%.2f): %s", quality, fact_text[:60])
                    continue

                if quality >= QUALITY_AUTO_ACCEPT or self.auto_approve:
                    # High quality or auto-approve mode — store directly
                    self._store_fact(fact_text, FactType.GENERAL, source)
                else:
                    # Medium quality — queue for review
                    self.pending.add(fact_text, quality, source)
                    self.facts_pending += 1
                    logger.info("Queued for review (%.2f): %s", quality, fact_text[:60])

        except Exception as e:
            logger.error("Auto-learn failed: %s", str(e))

    def _store_fact(self, fact_text: str, fact_type, source: str):
        """Store a fact in the knowledge base, enriching with topic + sentiment metadata."""
        metadata = {"source": source}

        # Enrich with topic detection
        try:
            from anamnesis.topics.detector import TopicDetector
            topic_result = TopicDetector().detect(fact_text)
            metadata["topic"] = topic_result.main_topic
            metadata["category"] = topic_result.category
        except Exception:
            pass  # Topic detection is optional enrichment

        # Enrich with sentiment analysis
        try:
            from anamnesis.emotional.analyzer import SentimentAnalyzer
            sentiment = SentimentAnalyzer().analyze(fact_text)
            metadata["emotion"] = sentiment.primary_emotion.value
            metadata["sentiment_score"] = round(sentiment.sentiment_score, 3)
        except Exception:
            pass  # Sentiment analysis is optional enrichment

        result = self.memory.semantic.add_fact(
            content=fact_text,
            fact_type=fact_type,
            confidence=0.5,
            metadata=metadata,
        )
        if result.confirmation_count > 1:
            self.duplicates_seen += 1
            logger.debug(
                "Duplicate confirmed: %s (count=%d)",
                fact_text[:50],
                result.confirmation_count,
            )
        else:
            self.facts_learned += 1
            logger.info("Learned new fact: %s", fact_text[:80])

            # Check for contradictions with existing facts
            self._check_contradictions(result)

            # Auto-link to related facts via memory graph
            self._auto_link_related(result)

            # Index into vector store for similarity search in proxy
            if self.retriever:
                try:
                    self.retriever.index_fact(result)
                except Exception as e:
                    logger.warning("Failed to index fact into vector store: %s", e)

    def _check_contradictions(self, new_fact):
        """Check if a new fact contradicts existing facts."""
        try:
            from anamnesis.consolidation.contradiction import detect_contradictions

            # Get existing facts of the same type
            existing = self.memory.semantic.list_all(limit=Config.CONTRADICTION_CHECK_LIMIT)
            if len(existing) < 2:
                return

            contradictions = detect_contradictions(
                new_fact, existing, threshold=0.5
            )

            for c in contradictions:
                if c.is_contradiction and c.existing_fact:
                    logger.info(
                        "Contradiction detected: '%s' vs '%s' (type=%s, conf=%.2f)",
                        new_fact.content[:50],
                        c.existing_fact.content[:50],
                        c.contradiction_type,
                        c.confidence,
                    )
                    # Mark old fact as superseded if new fact is confident enough
                    if c.confidence >= 0.7:
                        c.existing_fact.is_superseded = True
                        c.existing_fact.contradicted_by = getattr(
                            c.existing_fact, "contradicted_by", []
                        ) + [new_fact.id]
                        self.memory.semantic.save(c.existing_fact)
                        self.contradictions_found += 1
                        logger.info(
                            "Superseded: '%s' (replaced by '%s')",
                            c.existing_fact.content[:50],
                            new_fact.content[:50],
                        )
        except ImportError:
            pass  # Contradiction module not available
        except Exception as e:
            logger.debug("Contradiction check failed: %s", str(e))

    def _auto_link_related(self, new_fact):
        """Find related facts and create graph links for the new fact."""
        try:
            from anamnesis.graph.memory_graph import MemoryGraph, LinkType

            # Search for semantically similar facts
            related = self.memory.semantic.search(
                query=new_fact.content, limit=5
            )
            if not related:
                return

            graph = MemoryGraph(str(Config.DB_PATH))
            for rf in related:
                if rf.id != new_fact.id:
                    graph.add_link(
                        source_id=new_fact.id,
                        target_id=rf.id,
                        link_type=LinkType.RELATED,
                        weight=0.5,
                    )
        except ImportError:
            pass  # Graph module not available
        except Exception as e:
            logger.debug("Auto-link failed: %s", str(e))

    async def learn_from_assistant_message(self, text: str, source: str = "proxy"):
        """
        Extract and store confirmed user facts from an assistant response.

        The model often echoes user facts back in second-person ("you work at X").
        These are high-confidence confirmations — store them directly without review.
        """
        if not self.enabled:
            return
        try:
            from anamnesis.models import FactType
            facts = extract_facts_from_assistant(text)
            for fact_text in facts:
                # Assistant confirmations go straight to store — they're already confirmed
                self._store_fact(fact_text, FactType.GENERAL, f"{source}:assistant_confirmed")
        except Exception as e:
            logger.debug("Assistant learn failed: %s", e)

    def approve_pending(self, pending_id: str) -> bool:
        """Approve a pending fact and store it."""
        try:
            from anamnesis.models import FactType
        except ImportError:
            return False

        entry = self.pending.approve(pending_id)
        if entry:
            self._store_fact(entry["content"], FactType.GENERAL, entry.get("source", "proxy"))
            self.facts_pending = max(0, self.facts_pending - 1)
            return True
        return False

    def reject_pending(self, pending_id: str) -> bool:
        """Reject a pending fact."""
        if self.pending.reject(pending_id):
            self.facts_pending = max(0, self.facts_pending - 1)
            self.facts_rejected += 1
            return True
        return False

    def approve_all_pending(self) -> int:
        """Approve all pending facts."""
        try:
            from anamnesis.models import FactType
        except ImportError:
            return 0

        entries = self.pending.approve_all()
        for entry in entries:
            self._store_fact(entry["content"], FactType.GENERAL, entry.get("source", "proxy"))
        self.facts_pending = 0
        return len(entries)

    def reject_all_pending(self) -> int:
        """Reject all pending facts."""
        count = self.pending.reject_all()
        self.facts_pending = 0
        self.facts_rejected += count
        return count

    def get_stats(self) -> dict:
        """Return learning statistics."""
        return {
            "enabled": self.enabled,
            "auto_approve": self.auto_approve,
            "facts_learned": self.facts_learned,
            "facts_pending": self.pending.count(),
            "facts_rejected": self.facts_rejected,
            "duplicates_seen": self.duplicates_seen,
            "contradictions_found": self.contradictions_found,
        }
