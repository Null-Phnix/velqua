"""
Episode clustering based on topic similarity.

Groups similar episodes together for better organization.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from ..models import Episode
from .detector import TopicDetector, TopicResult


@dataclass
class ClusterResult:
    """A cluster of related episodes."""
    cluster_id: str
    name: str  # Derived from common topics
    episodes: List[Episode]
    common_keywords: List[str]
    category: str


class EpisodeClusterer:
    """
    Clusters episodes by topic similarity.

    Uses keyword overlap and category matching to
    group related episodes together.
    """

    def __init__(self, min_similarity: float = 0.2):
        """
        Initialize clusterer.

        Args:
            min_similarity: Minimum similarity (0-1) to cluster together
        """
        self.min_similarity = min_similarity
        self.detector = TopicDetector()

    def cluster(
        self,
        episodes: List[Episode],
        max_clusters: int = 20,
    ) -> List[ClusterResult]:
        """
        Cluster episodes by topic similarity.

        Args:
            episodes: List of episodes to cluster
            max_clusters: Maximum number of clusters

        Returns:
            List of ClusterResult objects
        """
        if not episodes:
            return []

        # First, detect topics for each episode
        episode_topics: Dict[str, TopicResult] = {}
        for ep in episodes:
            topic = self.detector.detect_from_episode(ep)
            episode_topics[ep.id] = topic

        # Group by category first (coarse clustering)
        by_category = defaultdict(list)
        for ep in episodes:
            cat = episode_topics[ep.id].category
            by_category[cat].append(ep)

        # Within each category, cluster by keyword similarity
        clusters = []

        for category, cat_episodes in by_category.items():
            if len(cat_episodes) == 1:
                # Single episode category
                ep = cat_episodes[0]
                topic = episode_topics[ep.id]
                clusters.append(ClusterResult(
                    cluster_id=f"{category}-0",
                    name=topic.main_topic,
                    episodes=[ep],
                    common_keywords=topic.keywords[:5],
                    category=category,
                ))
                continue

            # Cluster by keyword overlap
            sub_clusters = self._cluster_by_keywords(
                cat_episodes,
                episode_topics,
                category,
            )
            clusters.extend(sub_clusters)

        # Limit to max clusters by merging smallest
        while len(clusters) > max_clusters:
            # Find smallest cluster
            smallest = min(clusters, key=lambda c: len(c.episodes))
            clusters.remove(smallest)

            # Merge into most similar cluster
            if clusters:
                best_match = self._find_most_similar(smallest, clusters)
                if best_match:
                    best_match.episodes.extend(smallest.episodes)
                    # Update keywords
                    best_match.common_keywords = self._merge_keywords(
                        best_match.common_keywords,
                        smallest.common_keywords,
                    )

        # Sort by cluster size
        clusters.sort(key=lambda c: len(c.episodes), reverse=True)

        return clusters

    def _cluster_by_keywords(
        self,
        episodes: List[Episode],
        topics: Dict[str, TopicResult],
        category: str,
    ) -> List[ClusterResult]:
        """Cluster episodes within a category by keyword similarity."""
        if not episodes:
            return []

        clusters = []
        assigned = set()

        # Sort by number of keywords (more keywords = better center)
        sorted_eps = sorted(
            episodes,
            key=lambda e: len(topics[e.id].keywords),
            reverse=True,
        )

        for ep in sorted_eps:
            if ep.id in assigned:
                continue

            # Start a new cluster with this episode
            topic = topics[ep.id]
            cluster_eps = [ep]
            cluster_keywords = set(topic.keywords)
            assigned.add(ep.id)

            # Find similar episodes
            for other in sorted_eps:
                if other.id in assigned:
                    continue

                other_topic = topics[other.id]
                similarity = self._keyword_similarity(
                    cluster_keywords,
                    set(other_topic.keywords),
                )

                if similarity >= self.min_similarity:
                    cluster_eps.append(other)
                    cluster_keywords.update(other_topic.keywords)
                    assigned.add(other.id)

            # Create cluster
            cluster_name = self._generate_cluster_name(cluster_keywords)
            clusters.append(ClusterResult(
                cluster_id=f"{category}-{len(clusters)}",
                name=cluster_name,
                episodes=cluster_eps,
                common_keywords=list(cluster_keywords)[:10],
                category=category,
            ))

        return clusters

    def _keyword_similarity(self, kw1: Set[str], kw2: Set[str]) -> float:
        """Calculate Jaccard similarity between keyword sets."""
        if not kw1 or not kw2:
            return 0.0

        intersection = len(kw1 & kw2)
        union = len(kw1 | kw2)

        return intersection / union if union > 0 else 0.0

    def _generate_cluster_name(self, keywords: Set[str]) -> str:
        """Generate a name for the cluster from keywords."""
        if not keywords:
            return "Miscellaneous"

        # Use top 2-3 keywords
        top = sorted(keywords, key=len, reverse=True)[:3]
        return " & ".join(w.title() for w in top)

    def _find_most_similar(
        self,
        cluster: ClusterResult,
        candidates: List[ClusterResult],
    ) -> Optional[ClusterResult]:
        """Find the most similar cluster to merge with."""
        if not candidates:
            return None

        best = None
        best_sim = -1

        for candidate in candidates:
            # Same category preferred
            if candidate.category != cluster.category:
                continue

            sim = self._keyword_similarity(
                set(cluster.common_keywords),
                set(candidate.common_keywords),
            )

            if sim > best_sim:
                best_sim = sim
                best = candidate

        return best

    def _merge_keywords(
        self,
        kw1: List[str],
        kw2: List[str],
    ) -> List[str]:
        """Merge keyword lists, keeping most common."""
        combined = list(set(kw1) | set(kw2))
        return combined[:10]

    def get_cluster_stats(
        self,
        clusters: List[ClusterResult],
    ) -> Dict[str, any]:
        """Get statistics about clusters."""
        total_episodes = sum(len(c.episodes) for c in clusters)

        return {
            "num_clusters": len(clusters),
            "total_episodes": total_episodes,
            "avg_cluster_size": total_episodes / len(clusters) if clusters else 0,
            "by_category": {
                cat: sum(1 for c in clusters if c.category == cat)
                for cat in ["technical", "creative", "personal", "general"]
            },
            "largest_cluster": max(len(c.episodes) for c in clusters) if clusters else 0,
        }


def cluster_episodes(
    episodes: List[Episode],
    min_similarity: float = 0.2,
    max_clusters: int = 20,
) -> List[ClusterResult]:
    """
    Convenience function to cluster episodes.

    Args:
        episodes: Episodes to cluster
        min_similarity: Minimum similarity threshold
        max_clusters: Maximum number of clusters

    Returns:
        List of ClusterResult objects
    """
    clusterer = EpisodeClusterer(min_similarity)
    return clusterer.cluster(episodes, max_clusters)
