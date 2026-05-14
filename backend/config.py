"""
Central configuration for Velqua.
"""
from pathlib import Path
import os
import sys


def _default_data_dir() -> Path:
    """
    Return the platform-appropriate user data directory.

    When running as a PyInstaller bundle (sys.frozen is True), never write
    to the bundle/AppImage directory — it's read-only or ephemeral.
    Instead, use the OS-standard user data location:
      Linux:   $XDG_DATA_HOME/velqua  (default: ~/.local/share/velqua)
      macOS:   ~/Library/Application Support/velqua
      Windows: %APPDATA%/velqua
    """
    if getattr(sys, "frozen", False):
        if sys.platform == "win32":
            base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        elif sys.platform == "darwin":
            base = Path.home() / "Library" / "Application Support"
        else:
            base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        return base / "velqua"
    # Development / script mode — use project-root /data/
    return Path(__file__).parent.parent / "data"


class VelquaConfig:
    """Central configuration for Velqua."""

    # Server settings — localhost by default for security
    HOST: str = os.getenv("VELQUA_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("VELQUA_PORT", "8765"))
    PROXY_PORT: int = int(os.getenv("VELQUA_PROXY_PORT", "11435"))

    # Auth — optional bearer token (set to enable remote access)
    AUTH_TOKEN: str = os.getenv("VELQUA_AUTH_TOKEN", "")

    # CORS — localhost only by default, set to "*" for development
    CORS_ORIGINS: list = os.getenv("VELQUA_CORS_ORIGINS", "http://localhost:8765,http://127.0.0.1:8765").split(",")

    # File upload limits
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("VELQUA_MAX_UPLOAD_MB", "100"))
    MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    # Import limits
    MAX_CONVERSATIONS: int = int(os.getenv("VELQUA_MAX_CONVERSATIONS", "50"))
    MAX_PROJECTS: int = int(os.getenv("VELQUA_MAX_PROJECTS", "20"))
    MAX_MESSAGES_PER_CONV: int = int(os.getenv("VELQUA_MAX_MESSAGES", "100"))

    # Fact filtering
    MIN_FACT_LENGTH: int = 20
    MAX_FACT_LENGTH: int = 500
    FICTION_KEYWORDS: list = [
        "my character", "in the story", "my protagonist",
        "wizard", "dragon", "mana", "spell", "dungeon",
        "elves", "orcs", "magic", "quest", "tavern"
    ]

    # Memory settings
    DEFAULT_CONFIDENCE: float = 0.6
    HIGH_CONFIDENCE: float = 0.8
    MEMORY_BUDGET_TOKENS: int = int(os.getenv("VELQUA_MEMORY_BUDGET", "200"))

    # Retrieval settings — how many facts to fetch/index for memory injection
    MAX_FACTS_INDEX: int = 10000     # Upper bound for startup vector indexing
    MAX_FACTS_LIST: int = 100000     # Upper bound for listing/exporting all facts
    RETRIEVAL_LIMIT: int = 15        # Over-fetch count for hybrid search (ranked after)
    FTS_LIMIT: int = 10              # Fallback FTS-only search limit
    CONTRADICTION_CHECK_LIMIT: int = 100  # How many facts to compare for contradictions

    # Hybrid retrieval weights — FTS5 keyword + vector cosine similarity
    # Default 20/80 per arxiv-draft Section 5.1: FTS dilutes vector signal at higher weights
    FTS_WEIGHT: float = float(os.getenv("VELQUA_FTS_WEIGHT", "0.2"))
    VECTOR_WEIGHT: float = float(os.getenv("VELQUA_VECTOR_WEIGHT", "0.8"))

    # Cross-encoder reranker — breaks metric alignment bias by scoring
    # (query, passage) pairs with an independent model after hybrid retrieval.
    # Requires sentence-transformers; gracefully degrades if unavailable.
    RERANKER_ENABLED: bool = os.getenv("VELQUA_RERANKER", "").lower() in ("true", "1", "yes")
    RERANKER_MODEL: str = os.getenv("VELQUA_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANKER_CANDIDATES: int = int(os.getenv("VELQUA_RERANKER_CANDIDATES", "20"))

    # Evaluation scoring mode
    # "hybrid" = combine semantic similarity + LLM judge (legacy/default behavior)
    # "judge_only" = use ONLY the LLM judge score to avoid retrieval/eval model alignment bias
    EVAL_SCORING_MODE: str = os.getenv("VELQUA_EVAL_SCORING_MODE", "hybrid").strip().lower()

    # Decay model — controls how quickly facts lose relevance
    # Adaptive decay: score *= exp(-lambda * days_since_last_access)
    # Lambda 0.01 ≈ 70-day half-life (ln(2)/0.01 ≈ 69.3 days)
    DECAY_LAMBDA: float = float(os.getenv("VELQUA_DECAY_LAMBDA", "0.01"))
    DECAY_HALFLIFE_WEEKS: int = 4    # Personal facts decay slowly
    DECAY_IMPORTANCE_FACTOR: float = 2.0
    DECAY_ACCESS_FACTOR: float = 0.5
    DECAY_FLOOR: float = 0.1        # Never go below 10% strength

    # Episode-specific settings — episodes are temporal memories that decay
    # faster than facts but are weighted by recency and emotional relevance
    EPISODE_RETRIEVAL_LIMIT: int = int(os.getenv("VELQUA_EPISODE_LIMIT", "10"))
    EPISODE_DECAY_HALFLIFE_WEEKS: int = 1       # Episodes fade 4x faster than facts
    EPISODE_EMOTIONAL_BOOST: float = 1.5        # Emotional episodes persist 50% longer
    EPISODE_TOKEN_SHARE: float = 0.3            # 30% of token budget reserved for episodes

    # File size warnings
    LARGE_FILE_THRESHOLD_MB: int = 20

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = _default_data_dir()
    LOGS_DIR: Path = DATA_DIR / "logs"

    DB_PATH: Path = Path(os.getenv(
        "VELQUA_DB_PATH",
        str(DATA_DIR / "velqua.db")
    ))

    # Proxy settings
    PROXY_TIMEOUT: float = float(os.getenv("VELQUA_PROXY_TIMEOUT", "300"))

    # Ollama settings
    OLLAMA_BASE_URL: str = os.getenv("VELQUA_OLLAMA_URL", "http://localhost:11434")

    # OpenAI-compatible backend (llama.cpp, vLLM, LocalAI, LM Studio)
    OPENAI_BASE_URL: str = os.getenv("VELQUA_OPENAI_BASE_URL", "http://localhost:8080")

    # Logging
    LOG_LEVEL: str = os.getenv("VELQUA_LOG_LEVEL", "INFO")

    @classmethod
    def ensure_directories(cls):
        """Create required directories if they don't exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_summary(cls) -> dict:
        """Get config summary for debugging."""
        return {
            "server": f"{cls.HOST}:{cls.PORT}",
            "proxy": f"{cls.HOST}:{cls.PROXY_PORT}",
            "database": str(cls.DB_PATH),
            "max_upload_mb": cls.MAX_UPLOAD_SIZE_MB,
            "max_conversations": cls.MAX_CONVERSATIONS,
            "log_level": cls.LOG_LEVEL,
            "eval_scoring_mode": cls.EVAL_SCORING_MODE,
        }
