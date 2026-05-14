#!/usr/bin/env python3
"""
Velqua Memory Proxy - Transparent memory injection for any LLM provider.

Sits between user and their LLM backend, automatically injecting personal
memory context. Supports Ollama, OpenAI, Anthropic, Groq, and any
OpenAI-compatible backend.

User points apps to localhost:11435, proxy enriches requests with memory.
"""
import asyncio
import math
import re
import sys
import time
import unicodedata
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import httpx
import uvicorn

# Add both backend dir (for anamnesis) and project root (for backend.xxx) to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from anamnesis import Anamnesis
from anamnesis.retrieval.embedder import get_default_embedder
from anamnesis.retrieval.vector_store import InMemoryVectorStore
from anamnesis.retrieval.hybrid import HybridRetriever, SearchMode
from anamnesis.forgetting.decay import AdaptiveDecay, MemoryStrengthFactors
from backend import __version__
from backend.config import VelquaConfig as Config
from backend.logging_config import setup_logging, get_logger
from backend.auto_learner import AutoLearner
from backend.providers import registry, ProviderConfig, PROVIDER_CLASSES, DEFAULT_URLS
from backend.providers.base import BaseProvider
from backend.providers.ollama import OllamaProvider
from backend.providers.openai_compat import OpenAICompatProvider
from backend.providers.anthropic import AnthropicProvider
from backend.mesh.registry import registry as mesh_registry, detect_agent_id
from backend.mesh.noteboard import noteboard as mesh_noteboard
from backend.mesh.shared_memory import pool as mesh_pool
from backend.anamnesis.retrieval.reranker import CrossEncoderReranker

# Initialize logging
setup_logging(level=Config.LOG_LEVEL)
logger = get_logger("proxy")

def _init_memory() -> "Anamnesis":
    """Create the Anamnesis memory instance."""
    return Anamnesis(str(Config.DB_PATH))


def _init_vector_retriever(mem: "Anamnesis") -> "tuple[HybridRetriever | None, InMemoryVectorStore | None, bool]":
    """
    Initialise vector retrieval.

    Returns (retriever, vector_store, vector_enabled). On any failure falls
    back to FTS-only mode and returns (None, None, False).
    """
    try:
        embedder = get_default_embedder()
        vs = InMemoryVectorStore()
        ret = HybridRetriever(
            sqlite_backend=mem.backend,
            vector_store=vs,
            embedder=embedder,
            text_weight=Config.FTS_WEIGHT,
            vector_weight=Config.VECTOR_WEIGHT,
            decay_lambda=Config.DECAY_LAMBDA,
            decay_floor=Config.DECAY_FLOOR,
        )
        logger.info("Vector retrieval initialized (384-dim embeddings)")
        return ret, vs, True
    except Exception as e:
        logger.warning("Vector retrieval unavailable, using FTS only: %s", e)
        return None, None, False


def _init_reranker() -> "CrossEncoderReranker | None":
    """
    Initialise the cross-encoder reranker if enabled via config.

    The reranker itself lazy-loads its model on first use, so this is cheap.
    Returns None when disabled or on import failure.
    """
    if not Config.RERANKER_ENABLED:
        return None
    try:
        reranker = CrossEncoderReranker(model_name=Config.RERANKER_MODEL)
        logger.info("Cross-encoder reranker enabled (model: %s)", Config.RERANKER_MODEL)
        return reranker
    except Exception as e:
        logger.warning("Cross-encoder reranker init failed, disabled: %s", e)
        return None


def _init_learner(mem: "Anamnesis", ret) -> AutoLearner:
    """Create the AutoLearner bound to memory and (optionally) a retriever."""
    return AutoLearner(mem, retriever=ret)


def _load_api_keys():
    """Load stored API keys from keystore into provider configs."""
    try:
        from backend.keystore import KeyStore
        ks = KeyStore(Config.DATA_DIR)
        for provider_name in ks.list_providers():
            key = ks.get(provider_name)
            if key:
                registry.update_api_key(provider_name, key)
        logger.info("Loaded API keys for %d providers", len(ks.list_providers()))
    except Exception as e:
        logger.warning("Could not load keystore: %s", e)


def _index_existing_facts():
    """
    Index any facts already in the DB into the in-memory vector store.

    Called once at startup so hybrid retrieval works immediately without
    requiring a chat turn to prime the index.
    """
    if not VECTOR_ENABLED or not retriever:
        return
    try:
        all_facts = memory.semantic.list_all(limit=Config.MAX_FACTS_INDEX)
        if all_facts:
            retriever.index_all(episodes=[], facts=all_facts)
            logger.info("Indexed %d facts into vector store", len(all_facts))
        else:
            logger.info("No existing facts to index")
    except Exception as e:
        logger.error("Failed to index existing facts: %s", e)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup: load config, keys, and index existing facts."""
    providers_file = Config.DATA_DIR / "providers.json"
    registry._config_path = providers_file
    registry.load(providers_file)
    _load_api_keys()
    _index_existing_facts()
    yield


app = FastAPI(title="Velqua Memory Proxy", version=__version__, lifespan=lifespan)

# Module-level singletons — created once at import, shared across all requests
memory = _init_memory()
retriever, vector_store, VECTOR_ENABLED = _init_vector_retriever(memory)
reranker = _init_reranker()
learner = _init_learner(memory, retriever)

# Ollama configuration (for backward compat)
OLLAMA_BASE = Config.OLLAMA_BASE_URL
PROXY_PORT = Config.PROXY_PORT

# Memory injection settings (configurable based on hardware)
class MemoryConfig:
    """Memory injection configuration based on user hardware."""

    def __init__(self):
        self.budget = "minimal"  # minimal | standard | generous
        self.max_tokens = 200    # Token budget for memory context
        self.format = "natural"  # natural | structured | bullets
        self.active_provider = "ollama"

    def set_budget(self, gpu_vram_gb: int):
        """Auto-configure based on GPU VRAM."""
        if gpu_vram_gb <= 8:
            self.budget = "minimal"
            self.max_tokens = 200
        elif gpu_vram_gb <= 16:
            self.budget = "standard"
            self.max_tokens = 500
        elif gpu_vram_gb <= 24:
            self.budget = "generous"
            self.max_tokens = 1000
        else:  # 32GB+ (future Rubin GPUs)
            self.budget = "generous"
            self.max_tokens = 2000


config = MemoryConfig()


# ============================================================
# In-memory proxy metrics
# ============================================================

class ProxyMetrics:
    """In-memory metrics collector for proxy request telemetry."""

    def __init__(self):
        self.start_time = time.monotonic()
        self.total_requests = 0
        self.total_latency_ms = 0.0
        self.total_facts_injected = 0
        self.total_episodes_injected = 0
        self.total_tokens_used = 0
        self.total_token_budget = 0
        self.vector_searches = 0
        self.fts_fallbacks = 0
        self.errors = 0
        self.fact_retrieval_counts: Counter = Counter()
        self.requests_by_source: Counter = Counter()

    def record_request(
        self,
        latency_ms: float,
        facts_injected: int,
        episodes_injected: int,
        tokens_used: int,
        token_budget: int,
        search_mode: str,
        source: str,
        fact_contents: list[str] | None = None,
    ):
        self.total_requests += 1
        self.total_latency_ms += latency_ms
        self.total_facts_injected += facts_injected
        self.total_episodes_injected += episodes_injected
        self.total_tokens_used += tokens_used
        self.total_token_budget += token_budget
        self.requests_by_source[source] += 1

        if search_mode in ("hybrid", "hybrid+rerank"):
            self.vector_searches += 1
        else:
            self.fts_fallbacks += 1

        if fact_contents:
            for fact in fact_contents:
                self.fact_retrieval_counts[fact] += 1

    def record_error(self):
        self.errors += 1

    def to_dict(self) -> dict:
        uptime_s = time.monotonic() - self.start_time
        avg_latency = (
            self.total_latency_ms / self.total_requests
            if self.total_requests else 0
        )
        avg_facts = (
            self.total_facts_injected / self.total_requests
            if self.total_requests else 0
        )
        total_cache_ops = self.vector_searches + self.fts_fallbacks
        cache_hit_rate = (
            self.vector_searches / total_cache_ops
            if total_cache_ops else 0
        )
        avg_budget_pct = (
            self.total_tokens_used / self.total_token_budget * 100
            if self.total_token_budget else 0
        )
        top_facts = self.fact_retrieval_counts.most_common(10)

        return {
            "uptime_seconds": round(uptime_s),
            "total_requests": self.total_requests,
            "avg_latency_ms": round(avg_latency, 1),
            "total_facts_injected": self.total_facts_injected,
            "avg_facts_per_request": round(avg_facts, 2),
            "total_episodes_injected": self.total_episodes_injected,
            "cache_hit_rate": round(cache_hit_rate, 3),
            "vector_searches": self.vector_searches,
            "fts_fallbacks": self.fts_fallbacks,
            "avg_budget_usage_pct": round(avg_budget_pct, 1),
            "errors": self.errors,
            "requests_by_source": dict(self.requests_by_source),
            "top_retrieved_facts": [
                {"content": content, "retrievals": count}
                for content, count in top_facts
            ],
        }


proxy_metrics = ProxyMetrics()


# ============================================================
# FTS query expansion helpers
# ============================================================

# Named-entity alternative spellings/transliterations for mythology-heavy text.
# Keys and variants are stored in accent-folded lowercase form for robust matching.
_FTS_ENTITY_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "odin": ("odin", "odinn", "wodan", "woden", "wotan"),
    "odinn": ("odin", "odinn", "wodan", "woden", "wotan"),
    "wodan": ("odin", "odinn", "wodan", "woden", "wotan"),
    "woden": ("odin", "odinn", "wodan", "woden", "wotan"),
    "wotan": ("odin", "odinn", "wodan", "woden", "wotan"),
    "thor": ("thor", "thorr", "donar"),
    "thorr": ("thor", "thorr", "donar"),
    "donar": ("thor", "thorr", "donar"),
    "loki": ("loki", "loptr", "lodurr"),
    "loptr": ("loki", "loptr", "lodurr"),
    "lodurr": ("loki", "loptr", "lodurr"),
    "freyja": ("freyja", "freya"),
    "freya": ("freyja", "freya"),
    "frigg": ("frigg", "frigga"),
    "frigga": ("frigg", "frigga"),
    "indra": ("indra", "sakra", "shakra"),
    "sakra": ("indra", "sakra", "shakra"),
    "shakra": ("indra", "sakra", "shakra"),
    "krishna": ("krishna", "krsna", "kesava", "govinda"),
    "krsna": ("krishna", "krsna", "kesava", "govinda"),
    "zeus": ("zeus", "jupiter"),
    "jupiter": ("zeus", "jupiter"),
    "hera": ("hera", "juno"),
    "juno": ("hera", "juno"),
    "ares": ("ares", "mars"),
    "mars": ("ares", "mars"),
    "aphrodite": ("aphrodite", "venus"),
    "venus": ("aphrodite", "venus"),
    "hermes": ("hermes", "mercury"),
    "mercury": ("hermes", "mercury"),
    "athena": ("athena", "minerva", "pallas"),
    "minerva": ("athena", "minerva", "pallas"),
    "pallas": ("athena", "minerva", "pallas"),
}

# Small mythology synonym/related-term map to make the FTS component less brittle.
_FTS_TERM_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "myth": ("myth", "legend", "tradition"),
    "myths": ("myths", "legends", "traditions"),
    "legend": ("legend", "myth", "saga"),
    "legends": ("legends", "myths", "sagas"),
    "god": ("god", "deity", "divinity"),
    "gods": ("gods", "deities", "divinities"),
    "goddess": ("goddess", "deity", "divinity"),
    "goddesses": ("goddesses", "deities", "divinities"),
    "deity": ("deity", "god", "divinity"),
    "deities": ("deities", "gods", "divinities"),
    "pantheon": ("pantheon", "gods", "deities"),
    "underworld": ("underworld", "hell", "netherworld"),
    "afterlife": ("afterlife", "underworld", "netherworld"),
    "giant": ("giant", "jotunn", "jotun"),
    "giants": ("giants", "jotnar", "jotuns"),
    "jotun": ("jotun", "jotunn", "giant"),
    "jotunn": ("jotunn", "jotun", "giant"),
    "jotnar": ("jotnar", "giants"),
    "asgard": ("asgard", "aesir"),
    "aesir": ("aesir", "asgard"),
    "vanir": ("vanir", "vanaheim"),
    "ragnarok": ("ragnarok", "ragna rok"),
    "ritual": ("ritual", "rite", "ceremony"),
    "rituals": ("rituals", "rites", "ceremonies"),
    "rite": ("rite", "ritual", "ceremony"),
    "rites": ("rites", "rituals", "ceremonies"),
    "sacrifice": ("sacrifice", "offering", "oblation"),
    "sacrifices": ("sacrifices", "offerings", "oblations"),
    "offering": ("offering", "sacrifice", "oblation"),
    "offerings": ("offerings", "sacrifices", "oblations"),
    "epic": ("epic", "poem", "saga"),
    "epics": ("epics", "poems", "sagas"),
    "hero": ("hero", "champion", "demigod"),
    "heroes": ("heroes", "champions", "demigods"),
    "demigod": ("demigod", "hero", "half-god"),
    "demigods": ("demigods", "heroes", "half-gods"),
}


def _normalize_for_expansion(text: str) -> str:
    """Lowercase and strip diacritics for stable term matching."""
    text = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in text if not unicodedata.combining(ch)).lower()


def _tokenize_query_terms(query: str) -> list[str]:
    """Extract simple word tokens from a query for expansion."""
    return re.findall(r"\b[\w'-]+\b", _normalize_for_expansion(query))


def _expand_query_for_fts(query: str) -> str:
    """
    Expand a user query with mythology-aware aliases and synonyms.

    The returned string is still plain natural text so it remains compatible
    with existing FTS search code, but includes useful alternates such as:
      Odin -> Odin Óðinn Wodan
      Indra -> Indra Sakra/Shakra
      god -> deity divinity

    This boosts the usefulness of the FTS component in hybrid retrieval.
    """
    query = (query or "").strip()
    if not query:
        return query

    seen: set[str] = set()
    expanded_terms: list[str] = []

    def add_term(term: str) -> None:
        cleaned = term.strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            expanded_terms.append(cleaned)

    # Keep the original query first to preserve user intent.
    add_term(query)

    for token in _tokenize_query_terms(query):
        for variant in _FTS_ENTITY_EXPANSIONS.get(token, ()):
            add_term(variant)
        for variant in _FTS_TERM_EXPANSIONS.get(token, ()):
            add_term(variant)

    if len(expanded_terms) == 1:
        return query

    expanded_query = " ".join(expanded_terms)
    logger.debug("Expanded FTS query: '%s' -> '%s'", query, expanded_query)
    return expanded_query


def _dedupe_ranked_contents(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """
    Deduplicate ranked (content, score) pairs while preserving best score/order.

    Hybrid retrieval, MMR, reranking, and query expansion can surface the same
    fact text multiple times. Deduping here prevents wasting memory budget on
    duplicate lines and avoids skewing metrics.
    """
    seen: set[str] = set()
    deduped: list[tuple[str, float]] = []

    for content, score in items:
        normalized = " ".join((content or "").split()).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append((content, score))

    return deduped


def _score_ranked_fact_results(results: list, query_category: str) -> list[tuple[str, float]]:
    """
    Apply post-retrieval scoring consistently across retrieval modes.

    Combines freshness, topic boost, confirmation weighting, and retrieval score
    (when available) so hybrid/MMR/reranked candidates preserve base relevance
    instead of being reordered only by metadata-derived multipliers.
    """
    scored: list[tuple[str, float]] = []

    for result in results:
        content = getattr(result, "content", None)
        if not content:
            continue

        base_score = getattr(result, "score", None)
        if not isinstance(base_score, (int, float)):
            base_score = 1.0

        freshness = score_fact_freshness(result)
        boost = _topic_boost(result, query_category)
        conf = _confirmation_weight(result)

        scored.append((content, float(base_score) * freshness * boost * conf))

    scored.sort(key=lambda x: x[1], reverse=True)
    return _dedupe_ranked_contents(scored)


# ============================================================
# Streaming helper
# ============================================================

async def _stream_proxy(
    url: str, body: dict, headers: dict | None = None,
    media_type: str = "application/x-ndjson",
):
    """
    True streaming proxy — starts forwarding chunks as they arrive.

    Opens an httpx streaming connection, checks the status code from headers,
    then returns a StreamingResponse that yields chunks as they arrive from
    the backend. The client/response are kept alive until the generator
    finishes (or the downstream client disconnects).
    """
    client = httpx.AsyncClient(timeout=Config.PROXY_TIMEOUT)
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    req = client.build_request("POST", url, json=body, headers=req_headers)
    response = await client.send(req, stream=True)

    if response.status_code != 200:
        error_text = (await response.aread()).decode(errors="replace")[:200]
        await response.aclose()
        await client.aclose()
        raise HTTPException(
            status_code=502,
            detail=f"Backend returned {response.status_code}: {error_text}",
        )

    async def _yield_chunks():
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    return StreamingResponse(_yield_chunks(), media_type=media_type)


def _log_task_error(task: asyncio.Task) -> None:
    """Callback for fire-and-forget tasks — logs exceptions instead of swallowing them."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.warning("Background task failed: %s", exc)


# ============================================================
# Decay scoring for fact freshness
# ============================================================

decay = AdaptiveDecay(
    base_halflife_hours=168 * Config.DECAY_HALFLIFE_WEEKS,
    importance_factor=Config.DECAY_IMPORTANCE_FACTOR,
    access_factor=Config.DECAY_ACCESS_FACTOR,
    floor=Config.DECAY_FLOOR,
)

# Episode-specific decay — shorter halflife, stronger emotional factor
episode_decay = AdaptiveDecay(
    base_halflife_hours=168 * Config.EPISODE_DECAY_HALFLIFE_WEEKS,
    importance_factor=Config.DECAY_IMPORTANCE_FACTOR,
    access_factor=Config.DECAY_ACCESS_FACTOR,
    emotion_factor=Config.EPISODE_EMOTIONAL_BOOST,
    floor=0.02,  # Episodes can fade to near-zero
)


def _compute_fact_decay_multiplier(fact) -> float:
    """
    Compute an explicit forgetting-curve multiplier for retrieval scoring.

    Models the requested adaptive decay:
        score *= exp(-lambda * days_since_last_access)

    Frequently accessed facts resist decay by dividing effective age by a
    confirmation-based reinforcement term. This keeps long-lived, repeatedly
    useful facts competitive while allowing stale, unconfirmed facts to fade.
    """
    from datetime import datetime

    now = datetime.now()
    meta = getattr(fact, "metadata", {}) or {}

    last_touched = (
        getattr(fact, "last_confirmed", None)
        or meta.get("last_confirmed")
        or getattr(fact, "first_learned", None)
        or meta.get("first_learned")
    )

    if isinstance(last_touched, str):
        try:
            last_touched = datetime.fromisoformat(last_touched)
        except (ValueError, TypeError):
            last_touched = None

    if last_touched:
        age_days = max(0.0, (now - last_touched).total_seconds() / 86400.0)
    else:
        age_days = 0.0

    confirmation_count = (
        getattr(fact, "confirmation_count", None)
        or meta.get("confirmation_count", 1)
    )
    if not isinstance(confirmation_count, (int, float)) or confirmation_count < 1:
        confirmation_count = 1

    reinforcement = 1.0 + (Config.DECAY_ACCESS_FACTOR * max(0.0, confirmation_count - 1))
    effective_age_days = age_days / reinforcement
    multiplier = math.exp(-Config.DECAY_LAMBDA * effective_age_days)

    return max(Config.DECAY_FLOOR, min(1.0, multiplier))


def score_fact_freshness(fact) -> float:
    """
    Score a fact's freshness/relevance based on decay model.

    Accepts Fact objects (from FTS search) or HybridSearchResult objects
    (from hybrid retrieval). HybridSearchResult stores fact data in its
    metadata dict rather than as direct attributes.

    Returns 0.0-1.0 where higher = fresher/more relevant.
    """
    from datetime import datetime

    now = datetime.now()

    # Extract fields — try direct attributes first (Fact objects),
    # then fall back to metadata dict (HybridSearchResult objects)
    meta = getattr(fact, "metadata", {}) or {}

    last_touched = (
        getattr(fact, "last_confirmed", None)
        or meta.get("last_confirmed")
        or getattr(fact, "first_learned", None)
        or meta.get("first_learned")
    )

    # Parse string timestamps from metadata dict
    if isinstance(last_touched, str):
        try:
            last_touched = datetime.fromisoformat(last_touched)
        except (ValueError, TypeError):
            last_touched = None

    if last_touched:
        age_hours = (now - last_touched).total_seconds() / 3600
    else:
        age_hours = 0  # Unknown age = treat as fresh

    confirmation_count = (
        getattr(fact, "confirmation_count", None)
        or meta.get("confirmation_count", 1)
    )
    importance = (
        getattr(fact, "importance", None)
        or meta.get("importance", 0.5)
    )

    factors = MemoryStrengthFactors(
        base_importance=importance,
        age_hours=age_hours,
        access_count=confirmation_count,
        emotional_intensity=0.3,  # Default neutral
        reinforcement_count=max(0, confirmation_count - 1),
    )

    adaptive_strength = decay.calculate_strength(factors)
    explicit_decay = _compute_fact_decay_multiplier(fact)
    return adaptive_strength * explicit_decay


def score_episode_freshness(episode) -> float:
    """
    Score an episode's freshness weighted by recency and emotional relevance.

    Episodes use a shorter halflife than facts (1 week vs 4 weeks) so they
    fade faster — but emotionally charged episodes persist longer via the
    emotion_factor in the episode-specific decay model.

    Returns 0.0-1.0 where higher = fresher/more relevant.
    """
    from datetime import datetime
    from anamnesis.models import EmotionalValence

    now = datetime.now()

    # Extract timestamp
    started_at = getattr(episode, "started_at", None)
    if isinstance(started_at, str):
        try:
            started_at = datetime.fromisoformat(started_at)
        except (ValueError, TypeError):
            started_at = None

    if started_at:
        age_hours = (now - started_at).total_seconds() / 3600
    else:
        age_hours = 0  # Unknown = treat as fresh

    # Extract emotional intensity from valence
    valence = getattr(episode, "overall_valence", EmotionalValence.NEUTRAL)
    if isinstance(valence, EmotionalValence):
        emotional_intensity = abs(valence.value) / 2.0  # 0.0 to 1.0
    elif isinstance(valence, (int, float)):
        emotional_intensity = abs(valence) / 2.0
    else:
        emotional_intensity = 0.0

    importance = getattr(episode, "importance", 0.5)
    access_count = getattr(episode, "access_count", 0)

    factors = MemoryStrengthFactors(
        base_importance=importance,
        age_hours=age_hours,
        access_count=access_count,
        emotional_intensity=emotional_intensity,
        reinforcement_count=access_count,
    )

    return episode_decay.calculate_strength(factors)


def _detect_query_topic(query: str) -> str:
    """Detect the topic category of a user query for relevance boosting."""
    try:
        from anamnesis.topics.detector import TopicDetector
        result = TopicDetector().detect(query)
        return result.category if result.confidence > 0.3 else ""
    except (ImportError, Exception):
        return ""


def _topic_boost(fact, query_category: str) -> float:
    """
    Return a multiplier (1.0-1.3) based on topic match between
    fact metadata and the detected query category.
    """
    if not query_category:
        return 1.0
    fact_category = ""
    if hasattr(fact, "metadata") and fact.metadata:
        fact_category = fact.metadata.get("category", "")
    if fact_category and fact_category == query_category:
        return 1.3  # 30% boost for topic match
    return 1.0


def _confirmation_weight(obj) -> float:
    """
    Extract confirmation_count from a Fact, HybridSearchResult, or dict
    and return a score multiplier: log(1 + confirmation_count).

    A fact confirmed once (count=1) gets multiplier ~0.69,
    confirmed 5 times gets ~1.79, 10 times gets ~2.40.
    This ensures repeatedly-confirmed facts rank progressively higher.
    """
    meta = getattr(obj, "metadata", {}) or {}
    count = (
        getattr(obj, "confirmation_count", None)
        or meta.get("confirmation_count", 1)
    )
    if not isinstance(count, (int, float)) or count < 1:
        count = 1
    return math.log(1 + count)


def _get_result_fact_id(obj) -> str | None:
    """Extract a fact id from a Fact or HybridSearchResult-like object."""
    fact_id = getattr(obj, "id", None)
    if fact_id:
        return fact_id
    meta = getattr(obj, "metadata", {}) or {}
    return meta.get("id")


def _increment_retrieval_confirmations(results: list) -> None:
    """
    Increment confirmation_count for retrieved facts so future ranking improves.

    This makes retrieval itself act as a reinforcement signal:
    facts that are repeatedly useful become easier to retrieve again.

    Best-effort only — failures should never break the request path.
    """
    if not results:
        return

    seen_ids: set[str] = set()
    updated = 0

    for item in results:
        fact_id = _get_result_fact_id(item)
        if not fact_id or fact_id in seen_ids:
            continue
        seen_ids.add(fact_id)

        try:
            fact = memory.semantic.get(fact_id)
            if not fact:
                continue

            current = getattr(fact, "confirmation_count", 1) or 1
            fact.confirmation_count = current + 1

            from datetime import datetime
            fact.last_confirmed = datetime.now()

            memory.semantic.save(fact)

            # Keep in-memory result metadata aligned for this request's scoring path
            if hasattr(item, "confirmation_count"):
                try:
                    item.confirmation_count = fact.confirmation_count
                except Exception:
                    pass
            if hasattr(item, "metadata") and isinstance(item.metadata, dict):
                item.metadata["confirmation_count"] = fact.confirmation_count
                item.metadata["last_confirmed"] = fact.last_confirmed.isoformat()

            updated += 1
        except Exception as e:
            logger.debug("Failed to increment confirmation_count for fact %s: %s", fact_id, e)

    if updated:
        logger.debug("Incremented retrieval confirmations for %d facts", updated)


# ============================================================
# Memory retrieval + injection pipeline
# ============================================================

def _retrieve_relevant_facts(query: str) -> tuple[list[str], str]:
    """
    Retrieve facts relevant to the query, scored by freshness + topic relevance.

    Uses hybrid retrieval (FTS/vector weights from config, default 20/80) when available.
    Falls back to FTS-only when sentence-transformers is unavailable.

    When VELQUA_RERANKER=true:
      1. Over-fetch RERANKER_CANDIDATES (default 20) from hybrid search
      2. Cross-encoder re-scores (query, passage) pairs independently
      3. Freshness & topic boost applied as multipliers on the CE score

    This breaks metric alignment bias — the bi-encoder retrieves candidates,
    but the cross-encoder provides an independent relevance judgment.

    Returns:
        (fact_contents, search_mode) where search_mode is "hybrid+rerank",
        "hybrid", or "fts"
    """
    query = (query or "").strip()
    if not query:
        return [], "fts"

    query_category = _detect_query_topic(query)
    retrieval_query = _expand_query_for_fts(query)

    if VECTOR_ENABLED and retriever:
        # Over-fetch when reranker is active so it has more candidates to rescore
        fetch_limit = Config.RERANKER_CANDIDATES if reranker else Config.RETRIEVAL_LIMIT

        results = retriever.search(
            query=retrieval_query,
            limit=fetch_limit,
            mode=SearchMode.HYBRID,
            search_facts=True,
            search_episodes=False,
        )

        _increment_retrieval_confirmations(results)

        if reranker and results:
            # Cross-encoder rerank — independent relevance scoring
            candidates = [(r.content, r.score) for r in results if r.content]
            try:
                reranked = reranker.rerank(
                    query=query,
                    candidates=candidates,
                    top_k=Config.RETRIEVAL_LIMIT,
                )
                # Apply freshness & topic boost on top of CE scores
                scored = []
                # Build a lookup from content → original result for metadata.
                # Keep the highest-base-score result when duplicate content appears.
                result_map = {}
                for r in results:
                    if not getattr(r, "content", None):
                        continue
                    existing = result_map.get(r.content)
                    if existing is None or getattr(r, "score", 0.0) > getattr(existing, "score", 0.0):
                        result_map[r.content] = r

                for content, ce_score in reranked:
                    orig = result_map.get(content)
                    freshness = score_fact_freshness(orig) if orig else 1.0
                    boost = _topic_boost(orig, query_category) if orig else 1.0
                    conf = _confirmation_weight(orig) if orig else math.log(2)
                    scored.append((content, float(ce_score) * freshness * boost * conf))

                scored.sort(key=lambda x: x[1], reverse=True)
                scored = _dedupe_ranked_contents(scored)
                return [content for content, _ in scored], "hybrid+rerank"
            except Exception as e:
                logger.warning("Reranker failed, falling back to hybrid: %s", e)
                # Fall through to standard hybrid scoring below

        scored = _score_ranked_fact_results(results, query_category)
        return [content for content, _ in scored], "hybrid"

    facts = memory.semantic.search(query=retrieval_query, limit=Config.FTS_LIMIT)
    _increment_retrieval_confirmations(facts)

    scored = _score_ranked_fact_results(facts, query_category)
    return [content for content, _ in scored], "fts"


def _retrieve_relevant_episodes(query: str) -> list[tuple[str, float]]:
    """
    Retrieve episodes relevant to the query, scored by recency + emotional relevance.

    Returns list of (summary, score) tuples sorted by score descending.
    Episodes are scored with the episode-specific decay model which has a
    shorter halflife and stronger emotional weighting than facts.
    """
    try:
        episodes = memory.episodic.search(
            query=query, limit=Config.EPISODE_RETRIEVAL_LIMIT
        )
        if not episodes:
            return []

        scored = []
        for ep in episodes:
            score = score_episode_freshness(ep)
            if ep.summary:
                scored.append((ep.summary, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    except Exception as e:
        logger.warning("Episode retrieval failed: %s", e)
        return []


def _build_memory_context(
    fact_contents: list[str],
    header: str = "Context about the user:",
    episode_contents: list[tuple[str, float]] | None = None,
) -> tuple[str, int, int]:
    """
    Build a token-budgeted memory context block from ranked facts and episodes.

    Episodes get a configured share of the token budget (EPISODE_TOKEN_SHARE,
    default 30%). They're formatted under a separate header to distinguish
    temporal experiences from stable knowledge.

    Token counting is approximate (word count), which is close enough
    for context budgeting without needing a tokenizer.

    Returns:
        (context_string, facts_used, episodes_used)
    """
    episodes = episode_contents or []
    if not fact_contents and not episodes:
        return "", 0, 0

    total_budget = config.max_tokens
    sections = []
    token_count = 0
    facts_used = 0
    episodes_used = 0

    # Allocate budget: episodes get their share first, facts get the rest
    if episodes and fact_contents:
        episode_budget = int(total_budget * Config.EPISODE_TOKEN_SHARE)
        fact_budget = total_budget - episode_budget
    elif episodes:
        episode_budget = total_budget
        fact_budget = 0
    else:
        episode_budget = 0
        fact_budget = total_budget

    # Build episode section (recency-weighted temporal memories)
    if episodes:
        ep_header = "[Recent experiences:]"
        ep_lines = [ep_header]
        ep_tokens = len(ep_header.split())

        for summary, _score in episodes:
            line = f"- {summary}"
            line_tokens = len(line.split())
            if ep_tokens + line_tokens > episode_budget:
                break
            ep_lines.append(line)
            ep_tokens += line_tokens
            episodes_used += 1

        if episodes_used > 0:
            sections.append("\n".join(ep_lines))
            token_count += ep_tokens

    # Build fact section (stable knowledge)
    if fact_contents:
        fact_header = "[Known facts:]" if episodes_used > 0 else header
        fact_lines = [fact_header]
        fact_tokens = len(fact_header.split())

        for content in fact_contents:
            line = f"- {content}"
            line_tokens = len(line.split())
            if fact_tokens + line_tokens > fact_budget:
                break
            fact_lines.append(line)
            fact_tokens += line_tokens
            facts_used += 1

        if facts_used > 0:
            sections.append("\n".join(fact_lines))
            token_count += fact_tokens

    if facts_used == 0 and episodes_used == 0:
        return "", 0, 0

    return "\n\n".join(sections), facts_used, episodes_used


def inject_memory(prompt: str, max_tokens: int = 200) -> tuple[str, dict]:
    """
    Inject memory context into a raw prompt string (for /api/generate).

    Returns:
        (enhanced_prompt, metadata)
    """
    fact_contents, search_mode = _retrieve_relevant_facts(prompt)
    episode_contents = _retrieve_relevant_episodes(prompt)
    context, facts_used, episodes_used = _build_memory_context(
        fact_contents,
        header="Here's what I remember about you:",
        episode_contents=episode_contents,
    )

    if not context:
        return prompt, {"facts_injected": 0, "episodes_injected": 0, "search_mode": search_mode}

    return f"{context}\n\n{prompt}", {
        "facts_injected": facts_used,
        "episodes_injected": episodes_used,
        "tokens_added": len(context.split()),
        "budget": config.budget,
        "search_mode": search_mode,
    }


# ============================================================
# Unified chat handler — all endpoints route through this
# ============================================================

async def _handle_chat_request(
    body: dict,
    source: str,
    provider: BaseProvider | None = None,
    request: Request | None = None,
) -> dict | StreamingResponse:
    """
    Core chat handler shared by all chat endpoints.

    1. Detect agent identity (Mesh)
    2. Extract user message for retrieval
    3. Auto-learn in background
    4. Retrieve + inject memory (personal + mesh notes/shared memory)
    5. Route to the appropriate provider

    Args:
        body: The request body (messages, model, stream, etc.)
        source: Logging tag ("chat", "openai", "anthropic")
        provider: Explicit provider override (None = use active)
        request: Original FastAPI Request (for Mesh agent detection)

    Returns:
        Response dict or StreamingResponse
    """
    t0 = time.monotonic()

    if provider is None:
        provider = registry.get_active()

    # --- Mesh: detect agent identity and record heartbeat ---
    agent_id = "unknown"
    try:
        x_agent = request.headers.get("X-Velqua-Agent") if request else None
        ua = request.headers.get("User-Agent") if request else None
        agent_id = detect_agent_id(x_velqua_agent=x_agent, user_agent=ua)
    except Exception:
        pass

    messages = body.get("messages", [])
    stream = body.get("stream", False)
    model = body.get("model", "")

    # Collect user and assistant messages for multi-turn context
    user_msgs = [
        m["content"] for m in messages
        if m.get("role") == "user" and m.get("content")
    ]
    last_user_msg = user_msgs[-1] if user_msgs else ""

    # Build retrieval query from last 3 user turns — richer context than single message
    retrieval_query = " ".join(user_msgs[-3:]) if user_msgs else ""

    # --- Mesh: heartbeat + inject unread notes addressed to this agent ---
    mesh_note_context = ""
    try:
        mesh_registry.heartbeat(agent_id, task_hint=last_user_msg[:200])
        unread_notes = mesh_noteboard.get_for_agent(agent_id, unread_only=True, limit=5)
        if unread_notes:
            note_lines = ["[Mesh notes from other agents:]"]
            for note in unread_notes:
                note_lines.append(f"- From {note['from_agent']}: {note['content'][:300]}")
                mesh_noteboard.mark_read(note["id"])
            mesh_note_context = "\n".join(note_lines)
    except Exception as e:
        logger.debug("Mesh note injection failed: %s", e)

    # Learn from user message in background
    if last_user_msg:
        task = asyncio.create_task(
            learner.learn_from_message(last_user_msg, source=source)
        )
        task.add_done_callback(_log_task_error)

    # Learn from last assistant message — model confirmations are high-signal
    last_assistant_msg = next(
        (m["content"] for m in reversed(messages)
         if m.get("role") == "assistant" and m.get("content")),
        ""
    )
    if last_assistant_msg:
        task = asyncio.create_task(
            learner.learn_from_assistant_message(last_assistant_msg, source=source)
        )
        task.add_done_callback(_log_task_error)

    # Retrieve and inject memory using multi-turn query
    query = retrieval_query or last_user_msg
    fact_contents, search_mode = _retrieve_relevant_facts(query)
    episode_contents = _retrieve_relevant_episodes(query)
    context, facts_used, episodes_used = _build_memory_context(
        fact_contents, episode_contents=episode_contents
    )

    # Prepend mesh notes to personal memory context if any
    if mesh_note_context:
        context = (mesh_note_context + "\n\n" + context).strip() if context else mesh_note_context

    if facts_used > 0 or episodes_used > 0 or mesh_note_context:
        messages = provider.inject_memory(messages, context)
        body["messages"] = messages

    metadata = {
        "facts_injected": facts_used,
        "episodes_injected": episodes_used,
        "budget": config.budget,
        "search_mode": search_mode,
        "provider": provider.name,
        "agent_id": agent_id,
        "mesh_notes_injected": len(unread_notes) if "unread_notes" in locals() else 0,
    }

    # Record proxy metrics (measures retrieval + injection overhead, not LLM time)
    tokens_used = len(context.split()) if context else 0
    proxy_metrics.record_request(
        latency_ms=(time.monotonic() - t0) * 1000,
        facts_injected=facts_used,
        episodes_injected=episodes_used,
        tokens_used=tokens_used,
        token_budget=config.max_tokens,
        search_mode=search_mode,
        source=source,
        fact_contents=fact_contents[:facts_used] if fact_contents else None,
    )

    # Route to provider
    if isinstance(provider, OllamaProvider):
        return await _forward_ollama_chat(body, stream, metadata)
    elif isinstance(provider, AnthropicProvider):
        return await _forward_anthropic(body, stream, metadata, provider)
    else:
        return await _forward_openai_compat(body, stream, metadata, provider)


async def _forward_ollama_chat(
    body: dict, stream: bool, metadata: dict,
) -> dict | StreamingResponse:
    """Forward to Ollama /api/chat."""
    url = f"{OLLAMA_BASE}/api/chat"
    if stream:
        return await _stream_proxy(url, body, media_type="application/x-ndjson")

    async with httpx.AsyncClient(timeout=Config.PROXY_TIMEOUT) as client:
        response = await client.post(url, json=body)
        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Ollama returned {response.status_code}: {response.text[:200]}"
            )
        try:
            result = response.json()
        except ValueError:
            raise HTTPException(status_code=502, detail="Ollama returned invalid JSON")
        result["velqua_metadata"] = metadata
        return result


async def _forward_anthropic(
    body: dict, stream: bool, metadata: dict, provider: AnthropicProvider,
) -> dict | StreamingResponse:
    """Forward to Anthropic Messages API."""
    messages = body.get("messages", [])

    # Anthropic: extract system from messages, put as top-level param
    system_text, user_messages = provider._extract_system(messages)

    anthropic_body = {
        "model": body.get("model") or provider._resolve_model(""),
        "messages": user_messages,
        "max_tokens": body.get("max_tokens", 4096),
    }
    if system_text:
        anthropic_body["system"] = system_text
    if body.get("temperature") is not None:
        anthropic_body["temperature"] = body["temperature"]
    if stream:
        anthropic_body["stream"] = True

    url = f"{provider.config.base_url}/v1/messages"
    headers = provider.get_auth_headers()

    if stream:
        return await _stream_proxy(url, anthropic_body, headers, "text/event-stream")

    async with httpx.AsyncClient(timeout=Config.PROXY_TIMEOUT) as client:
        response = await client.post(
            url, json=anthropic_body,
            headers={"Content-Type": "application/json", **headers},
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Anthropic returned {response.status_code}: {response.text[:200]}"
            )
        try:
            result = response.json()
        except ValueError:
            raise HTTPException(status_code=502, detail="Anthropic returned invalid JSON")
        result["velqua_metadata"] = metadata
        return result


async def _forward_openai_compat(
    body: dict, stream: bool, metadata: dict, provider: BaseProvider,
) -> dict | StreamingResponse:
    """Forward to any OpenAI-compatible backend."""
    url = f"{provider.config.base_url}/v1/chat/completions"
    headers = provider.get_auth_headers()

    if stream:
        return await _stream_proxy(url, body, headers, "text/event-stream")

    async with httpx.AsyncClient(timeout=Config.PROXY_TIMEOUT) as client:
        response = await client.post(
            url, json=body,
            headers={"Content-Type": "application/json", **headers},
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Backend returned {response.status_code}: {response.text[:200]}"
            )
        try:
            result = response.json()
        except ValueError:
            raise HTTPException(status_code=502, detail="Backend returned invalid JSON")
        result["velqua_metadata"] = metadata
        return result


# ============================================================
# Config endpoints
# ============================================================

@app.get("/proxy/config")
async def get_proxy_config():
    """Get current memory configuration."""
    return {
        "budget": config.budget,
        "max_tokens": config.max_tokens,
        "format": config.format,
        "database": str(Config.DB_PATH),
        "ollama_base": OLLAMA_BASE,
        "active_provider": registry.active_name,
    }


@app.post("/proxy/config")
async def update_proxy_config(
    gpu_vram_gb: Optional[int] = None,
    budget: Optional[str] = None,
):
    """Update memory configuration."""
    if gpu_vram_gb is not None:
        config.set_budget(gpu_vram_gb)
    if budget is not None:
        valid_budgets = ["minimal", "standard", "generous"]
        if budget not in valid_budgets:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid budget '{budget}'. Must be one of: {', '.join(valid_budgets)}",
            )
        config.budget = budget

    return await get_proxy_config()


# ============================================================
# Proxy endpoints — backward compatible + new
# ============================================================

@app.post("/api/generate")
async def proxy_generate(request: Request):
    """
    Proxy Ollama /api/generate with memory injection.
    Always routes to Ollama (generate is Ollama-specific).
    """
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")

    try:
        t0 = time.monotonic()
        original_prompt = body.get("prompt", "")

        # Auto-learn from user prompt
        task = asyncio.create_task(
            learner.learn_from_message(original_prompt, source="generate")
        )
        task.add_done_callback(_log_task_error)

        # Inject memory
        enhanced_prompt, metadata = inject_memory(original_prompt, config.max_tokens)
        body["prompt"] = enhanced_prompt

        # Record proxy metrics
        proxy_metrics.record_request(
            latency_ms=(time.monotonic() - t0) * 1000,
            facts_injected=metadata.get("facts_injected", 0),
            episodes_injected=metadata.get("episodes_injected", 0),
            tokens_used=metadata.get("tokens_added", 0),
            token_budget=config.max_tokens,
            search_mode=metadata.get("search_mode", "fts"),
            source="generate",
        )

        # Always forward to Ollama (generate is Ollama-native)
        if body.get("stream", False):
            return await _stream_proxy(
                f"{OLLAMA_BASE}/api/generate", body,
                media_type="application/x-ndjson",
            )
        else:
            async with httpx.AsyncClient(timeout=Config.PROXY_TIMEOUT) as client:
                response = await client.post(
                    f"{OLLAMA_BASE}/api/generate",
                    json=body,
                )
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Ollama returned {response.status_code}: {response.text[:200]}"
                    )
                try:
                    result = response.json()
                except ValueError:
                    raise HTTPException(
                        status_code=502, detail="Ollama returned invalid JSON"
                    )
                result["velqua_metadata"] = metadata
                return result

    except HTTPException:
        raise
    except httpx.ConnectError:
        proxy_metrics.record_error()
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama. Is it running on localhost:11434?"
        )
    except Exception as e:
        proxy_metrics.record_error()
        logger.error("Proxy generate failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def proxy_chat(request: Request):
    """
    Proxy Ollama /api/chat with memory injection.
    Routes to Ollama provider (this is an Ollama-native endpoint).
    """
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")

    try:
        ollama = registry.get("ollama")
        return await _handle_chat_request(body, source="chat", provider=ollama, request=request)

    except HTTPException:
        raise
    except httpx.ConnectError:
        proxy_metrics.record_error()
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama. Is it running on localhost:11434?"
        )
    except Exception as e:
        proxy_metrics.record_error()
        logger.error("Proxy chat failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def proxy_openai_chat(request: Request):
    """
    OpenAI-compatible chat completions proxy with memory injection.

    Routes to the active provider. Works with OpenAI, Groq, local backends,
    and any OpenAI API-compatible service.
    """
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")

    try:
        # Use the active provider (could be OpenAI, Groq, local, etc.)
        active = registry.get_active()

        # If active is Anthropic, convert format internally
        if isinstance(active, AnthropicProvider):
            return await _handle_chat_request(body, source="openai", provider=active, request=request)

        return await _handle_chat_request(body, source="openai", provider=active, request=request)

    except HTTPException:
        raise
    except httpx.ConnectError:
        proxy_metrics.record_error()
        provider_name = registry.active_name
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to {provider_name} backend.",
        )
    except Exception as e:
        proxy_metrics.record_error()
        logger.error("Proxy OpenAI chat failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/messages")
async def proxy_anthropic_messages(request: Request):
    """
    Anthropic Messages API proxy with memory injection.

    Apps using the Anthropic SDK can point at Velqua to get memory injection.
    The 'system' parameter is preserved and memory context is prepended to it.
    """
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")

    try:
        # Get Anthropic provider (or active if it happens to be Anthropic)
        provider = registry.get("anthropic") or registry.get_active()

        # Normalize: Anthropic sends system as top-level param, not in messages
        messages = body.get("messages", [])
        system_text = body.get("system", "")
        if system_text:
            messages = [{"role": "system", "content": system_text}] + messages
            body["messages"] = messages

        return await _handle_chat_request(body, source="anthropic", provider=provider, request=request)

    except HTTPException:
        raise
    except httpx.ConnectError:
        proxy_metrics.record_error()
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Anthropic API.",
        )
    except Exception as e:
        proxy_metrics.record_error()
        logger.error("Proxy Anthropic messages failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Pass-through & stats endpoints
# ============================================================

@app.get("/api/tags")
async def proxy_tags():
    """Pass-through for Ollama model listing."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{OLLAMA_BASE}/api/tags")
            if response.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"Ollama returned {response.status_code}"
                )
            return response.json()
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Ollama not running")
        except HTTPException:
            raise
        except ValueError:
            raise HTTPException(status_code=502, detail="Ollama returned invalid JSON")


@app.get("/proxy/learning")
async def get_learning_stats():
    """Get auto-learning statistics."""
    return learner.get_stats()


@app.post("/proxy/learning")
async def toggle_learning(enabled: Optional[bool] = None):
    """Enable or disable auto-learning."""
    if enabled is not None:
        learner.enabled = enabled
        logger.info("Auto-learning %s", "enabled" if enabled else "disabled")
    return learner.get_stats()


@app.post("/proxy/preview")
async def preview_memory_injection(request: Request):
    """
    Preview what memory would be injected for a given query.

    Runs the full retrieval pipeline without actually sending anything to an LLM.
    Useful for debugging memory quality — see exactly what facts get injected
    and in what order before committing to a real request.

    Body: { "query": "your message here" }
    Returns: facts scored + ranked, the context string, token counts.
    """
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    query = (body.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    # Run the same pipeline as a real chat request
    query_category = _detect_query_topic(query)
    retrieval_query = _expand_query_for_fts(query)

    candidates = []
    if VECTOR_ENABLED and retriever:
        results = retriever.search(
            query=retrieval_query,
            limit=min(Config.RETRIEVAL_LIMIT * 2, 30),
            mode=SearchMode.HYBRID,
            search_facts=True,
            search_episodes=False,
        )
        ranked = _score_ranked_fact_results(results, query_category)
        score_lookup = {content: score for content, score in ranked}

        for r in results:
            if r.content and r.content in score_lookup:
                candidates.append({
                    "content": r.content,
                    "score": round(score_lookup[r.content], 3),
                    "freshness": round(score_fact_freshness(r), 3),
                    "topic_boost": round(_topic_boost(r, query_category), 3),
                    "confirmation_weight": round(_confirmation_weight(r), 3),
                    "decay_multiplier": round(_compute_fact_decay_multiplier(r), 3),
                    "base_score": round(float(getattr(r, "score", 1.0) or 1.0), 3),
                    "mode": "hybrid",
                })
        search_mode = "hybrid"
    else:
        facts = memory.semantic.search(query=retrieval_query, limit=30)
        ranked = _score_ranked_fact_results(facts, query_category)
        score_lookup = {content: score for content, score in ranked}

        for f in facts:
            if f.content and f.content in score_lookup:
                candidates.append({
                    "content": f.content,
                    "score": round(score_lookup[f.content], 3),
                    "freshness": round(score_fact_freshness(f), 3),
                    "topic_boost": round(_topic_boost(f, query_category), 3),
                    "confirmation_weight": round(_confirmation_weight(f), 3),
                    "decay_multiplier": round(_compute_fact_decay_multiplier(f), 3),
                    "base_score": round(float(getattr(f, "score", 1.0) or 1.0), 3),
                    "mode": "fts",
                })
        search_mode = "fts"

    deduped_candidates = []
    seen_candidate_content = set()
    for item in sorted(candidates, key=lambda x: x["score"], reverse=True):
        normalized = " ".join(item["content"].split()).strip().lower()
        if normalized in seen_candidate_content:
            continue
        seen_candidate_content.add(normalized)
        deduped_candidates.append(item)
    candidates = deduped_candidates

    # Retrieve episodes too
    episode_results = _retrieve_relevant_episodes(query)
    episode_candidates = [
        {"summary": summary, "score": round(score, 3)}
        for summary, score in episode_results
    ]

    # Build context using the real function
    fact_strings = [c["content"] for c in candidates]
    context, facts_injected, episodes_injected = _build_memory_context(
        fact_strings, episode_contents=episode_results
    )

    return {
        "query": query,
        "query_category": query_category,
        "search_mode": search_mode,
        "expanded_query": retrieval_query,
        "token_budget": config.max_tokens,
        "tokens_used": len(context.split()) if context else 0,
        "facts_available": len(candidates),
        "facts_injected": facts_injected,
        "episodes_available": len(episode_candidates),
        "episodes_injected": episodes_injected,
        "fact_candidates": candidates,
        "episode_candidates": episode_candidates,
        "context": context,
    }


@app.post("/proxy/summarize-session")
async def summarize_session(request: Request):
    """
    Process a full conversation and extract facts from all turns.

    Useful for ingesting a completed conversation in one shot — e.g.,
    after copying a chat from another tool. Runs the same extraction
    pipeline as live proxy interception but awaits each turn sequentially
    so the result count is accurate.

    Body: { "messages": [{"role": "user"|"assistant", "content": "..."}],
            "source": "optional_tag" }
    Returns: { "success": true, "messages_processed": N, "facts_stored": N }
    """
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    messages = body.get("messages", [])
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be an array")

    source = body.get("source", "session_summary")
    facts_before = learner.facts_learned
    messages_processed = 0

    for msg in messages:
        role = msg.get("role", "")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        messages_processed += 1
        try:
            if role == "user":
                await learner.learn_from_message(content, source=source)
            elif role == "assistant":
                await learner.learn_from_assistant_message(content, source=source)
        except Exception as e:
            logger.debug("Session summary learn failed for %s: %s", role, e)

    facts_stored = learner.facts_learned - facts_before
    return {
        "success": True,
        "messages_processed": messages_processed,
        "facts_stored": facts_stored,
        "source": source,
    }


@app.get("/proxy/retrieval")
async def get_retrieval_stats():
    """Get vector retrieval statistics."""
    stats = {
        "vector_enabled": VECTOR_ENABLED,
        "vectors_indexed": vector_store.count() if VECTOR_ENABLED else 0,
    }
    if VECTOR_ENABLED and retriever:
        stats.update(retriever.get_stats())
    return stats


def _check_model_cached(model_name: str = "all-MiniLM-L6-v2") -> bool:
    """Return True if the sentence-transformer model is already in the HuggingFace cache."""
    from pathlib import Path
    cache_base = Path.home() / ".cache" / "huggingface" / "hub"
    normalized = "models--sentence-transformers--" + model_name.replace("/", "--")
    return (cache_base / normalized).exists()


@app.get("/metrics")
async def get_metrics():
    """Return real-time proxy metrics (in-memory, not persisted)."""
    return proxy_metrics.to_dict()


@app.get("/")
async def root():
    """Proxy status."""
    return {
        "service": "Velqua Memory Proxy",
        "version": __version__,
        "active_provider": registry.active_name,
        "providers": [p["name"] for p in registry.list_providers() if p.get("enabled")],
        "endpoints": {
            "ollama_generate": "/api/generate",
            "ollama_chat": "/api/chat",
            "openai_chat": "/v1/chat/completions",
            "anthropic_messages": "/v1/messages",
        },
        "proxy_port": PROXY_PORT,
        "memory_config": {
            "budget": config.budget,
            "max_tokens": config.max_tokens
        },
        "vector_retrieval": VECTOR_ENABLED,
        "model_cached": _check_model_cached() if VECTOR_ENABLED else None,
        "auto_learning": learner.get_stats(),
    }


def main():
    """Entry point for the `velqua-proxy` CLI command."""
    logger.info("Starting Velqua Memory Proxy...")
    logger.info("Proxying on port %d", PROXY_PORT)
    logger.info("Active provider: %s", registry.active_name)
    logger.info("Memory budget: %s (%d tokens)", config.budget, config.max_tokens)
    logger.info("Database: %s", Config.DB_PATH)

    uvicorn.run(
        app,
        host=Config.HOST,
        port=PROXY_PORT,
        log_level=Config.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    main()
