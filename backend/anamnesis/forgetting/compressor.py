"""
Memory compression for fading memories.

As memories age, details fade but gist remains.
This module handles the compression of memories while
preserving their essential meaning.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class CompressionResult:
    """Result of memory compression."""
    original_size: int  # Character count
    compressed_size: int
    compression_ratio: float
    preserved_elements: List[str]
    removed_elements: List[str]


class MemoryCompressor:
    """
    Compresses memories while preserving essential information.

    Compression levels:
    1. Light - Remove filler, keep all substance
    2. Medium - Keep key points and emotional moments
    3. Heavy - Keep only summary and core facts
    4. Gist - Single sentence essence
    """

    # Words/phrases that can be safely removed
    FILLER_PATTERNS = [
        r'\b(um|uh|like|you know|basically|actually|literally)\b',
        r'\b(i think|i guess|i mean|kind of|sort of)\b',
        r'\b(just|really|very|quite|pretty much)\b',
    ]

    # Patterns for important content to preserve
    IMPORTANT_PATTERNS = [
        r'\b(decided|concluded|learned|realized|discovered)\b',
        r'\b(important|crucial|key|main|critical)\b',
        r'\b(because|therefore|so|thus|hence)\b',
        r'\b(always|never|must|should|need to)\b',
    ]

    def __init__(self):
        pass

    def compress(
        self,
        messages: List[Dict[str, str]],
        level: str = "medium",
        max_length: Optional[int] = None,
    ) -> tuple[List[Dict[str, str]], CompressionResult]:
        """
        Compress a list of messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            level: "light", "medium", "heavy", or "gist"
            max_length: Optional maximum total character length

        Returns:
            Tuple of (compressed_messages, CompressionResult)
        """
        original_size = sum(len(m.get("content", "")) for m in messages)

        if level == "light":
            compressed = self._compress_light(messages)
        elif level == "medium":
            compressed = self._compress_medium(messages)
        elif level == "heavy":
            compressed = self._compress_heavy(messages)
        elif level == "gist":
            compressed = self._compress_gist(messages)
        else:
            compressed = messages

        # Apply max length if specified
        if max_length:
            compressed = self._truncate_to_length(compressed, max_length)

        compressed_size = sum(len(m.get("content", "")) for m in compressed)

        return compressed, CompressionResult(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compressed_size / original_size if original_size > 0 else 0,
            preserved_elements=self._identify_preserved(compressed),
            removed_elements=self._identify_removed(messages, compressed),
        )

    def _compress_light(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Light compression - remove filler words only."""
        compressed = []
        for msg in messages:
            content = msg.get("content", "")

            # Remove filler patterns
            for pattern in self.FILLER_PATTERNS:
                content = re.sub(pattern, "", content, flags=re.IGNORECASE)

            # Clean up extra whitespace
            content = re.sub(r'\s+', ' ', content).strip()

            if content:
                compressed.append({"role": msg.get("role", "user"), "content": content})

        return compressed

    def _compress_medium(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Medium compression - keep key sentences only."""
        compressed = []

        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")

            # Split into sentences
            sentences = re.split(r'[.!?]+', content)

            # Score and filter sentences
            kept_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Check if sentence is important
                score = self._score_sentence_importance(sentence)
                if score >= 0.3 or len(kept_sentences) < 2:  # Keep at least 2
                    kept_sentences.append(sentence)

            if kept_sentences:
                compressed.append({
                    "role": role,
                    "content": ". ".join(kept_sentences) + "."
                })

        return compressed

    def _compress_heavy(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Heavy compression - keep only the most important points."""
        # First do medium compression
        medium = self._compress_medium(messages)

        # Then keep only top 3 messages
        if len(medium) <= 3:
            return medium

        # Score messages by importance
        scored = []
        for msg in medium:
            score = self._score_sentence_importance(msg.get("content", ""))
            scored.append((msg, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Keep top 3, but maintain order
        top_msgs = [m for m, s in scored[:3]]

        # Restore original order
        result = []
        for msg in medium:
            if msg in top_msgs:
                result.append(msg)

        return result

    def _compress_gist(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Gist compression - single sentence summary."""
        # Combine all content
        all_content = " ".join(m.get("content", "") for m in messages)

        # Extract key phrases
        key_phrases = []

        # Look for conclusion/summary patterns
        patterns = [
            r'(?:in summary|to summarize|overall|in conclusion)[,:]?\s*(.+?)(?:\.|$)',
            r'(?:the main point|the key|essentially)[,:]?\s*(.+?)(?:\.|$)',
            r'(?:decided|concluded|learned|realized)\s+(?:that\s+)?(.+?)(?:\.|$)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, all_content, re.IGNORECASE)
            key_phrases.extend(matches)

        if key_phrases:
            gist = key_phrases[0].strip()
        else:
            # Fall back to first meaningful sentence
            sentences = re.split(r'[.!?]+', all_content)
            meaningful = [s.strip() for s in sentences if len(s.strip()) > 20]
            gist = meaningful[0] if meaningful else all_content[:100]

        # Ensure it ends properly
        if not gist.endswith(('.', '!', '?')):
            gist += '.'

        return [{"role": "summary", "content": gist}]

    def _truncate_to_length(
        self,
        messages: List[Dict[str, str]],
        max_length: int,
    ) -> List[Dict[str, str]]:
        """Truncate messages to fit within max_length."""
        result = []
        current_length = 0

        for msg in messages:
            content = msg.get("content", "")
            remaining = max_length - current_length

            if remaining <= 0:
                break

            if len(content) <= remaining:
                result.append(msg)
                current_length += len(content)
            else:
                # Truncate this message
                truncated = content[:remaining-3] + "..."
                result.append({"role": msg.get("role", "user"), "content": truncated})
                break

        return result

    def _score_sentence_importance(self, sentence: str) -> float:
        """Score how important a sentence is."""
        score = 0.0
        lower = sentence.lower()

        # Check for important patterns
        for pattern in self.IMPORTANT_PATTERNS:
            if re.search(pattern, lower):
                score += 0.2

        # Length bonus (not too short, not too long)
        word_count = len(sentence.split())
        if 5 <= word_count <= 30:
            score += 0.1

        # Question bonus
        if '?' in sentence:
            score += 0.1

        # Named entities bonus (capitalized words)
        caps = re.findall(r'\b[A-Z][a-z]+\b', sentence)
        score += len(caps) * 0.05

        return min(1.0, score)

    def _identify_preserved(self, messages: List[Dict[str, str]]) -> List[str]:
        """Identify what was preserved in compression."""
        preserved = []

        for msg in messages:
            content = msg.get("content", "")

            # Check for key concepts
            for pattern in self.IMPORTANT_PATTERNS:
                matches = re.findall(pattern, content, re.IGNORECASE)
                preserved.extend(matches)

        return list(set(preserved))

    def _identify_removed(
        self,
        original: List[Dict[str, str]],
        compressed: List[Dict[str, str]],
    ) -> List[str]:
        """Identify what was removed in compression."""
        original_content = " ".join(m.get("content", "") for m in original)
        " ".join(m.get("content", "") for m in compressed)

        # Find removed filler
        removed = []
        for pattern in self.FILLER_PATTERNS:
            matches = re.findall(pattern, original_content, re.IGNORECASE)
            removed.extend(matches)

        return list(set(removed))


def compress_episode_messages(
    messages: List[Dict[str, str]],
    target_ratio: float = 0.5,
) -> List[Dict[str, str]]:
    """
    Compress episode messages to target ratio.

    Automatically selects compression level based on target.
    """
    compressor = MemoryCompressor()

    # Try levels until we hit target
    for level in ["light", "medium", "heavy", "gist"]:
        compressed, result = compressor.compress(messages, level)
        if result.compression_ratio <= target_ratio:
            return compressed

    return compressed  # Return gist if nothing else works
