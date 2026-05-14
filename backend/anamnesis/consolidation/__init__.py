"""
Consolidation pipeline for converting raw experiences into memories.

Handles:
- Summarization (conversation -> episode summary)
- Extraction (conversation -> facts, preferences, emotions)
- Compression (detailed -> gist)
"""

from .extractor import EmotionExtractor, FactExtractor
from .merger import EpisodeMerger, MergeCandidate, MergeResult, find_duplicates, merge_episodes
from .pipeline import ConsolidationPipeline
from .summarizer import EnhancedSummarizer, HeuristicSummarizer, Summarizer

__all__ = [
    "ConsolidationPipeline",
    "Summarizer",
    "HeuristicSummarizer",
    "EnhancedSummarizer",
    "FactExtractor",
    "EmotionExtractor",
    "EpisodeMerger",
    "MergeCandidate",
    "MergeResult",
    "find_duplicates",
    "merge_episodes",
]
