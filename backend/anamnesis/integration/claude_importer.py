"""
Claude memory direct importer.

Parses Claude's memories.json export format to extract high-quality,
LLM-extracted facts instead of using regex-based extraction.

Claude's memory structure:
- conversations_memory: Main markdown with sections (Work, Personal, Top of mind, Brief history)
- project_memories: Per-project markdown sections

This gives us instant perfect quality since Claude already did the extraction work.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..consolidation.extractor import ExtractedFact
from ..models import FactType


# Map Claude's memory sections to FactType
SECTION_TO_FACT_TYPE = {
    "work context": FactType.PROFESSIONAL,
    "personal context": FactType.PERSONAL,
    "top of mind": FactType.PROJECT,
    "brief history": FactType.GENERAL,
    "on the horizon": FactType.PROJECT,
    "key learnings": FactType.PREFERENCE,
    "approach & patterns": FactType.PREFERENCE,
    "purpose & context": FactType.PROJECT,
    "current state": FactType.PROJECT,
    "tools & resources": FactType.PROFESSIONAL,
}


@dataclass
class ClaudeMemoryData:
    """Parsed Claude memory export."""
    conversations_memory: str
    project_memories: Dict[str, str]  # project_id -> markdown content
    account_uuid: str


def parse_claude_memories(json_path: Path) -> ClaudeMemoryData:
    """
    Parse Claude's memories.json export.

    Expected structure:
    [
        {
            "conversations_memory": "**Work context**\n...",
            "project_memories": {
                "project-uuid": "**Purpose & context**\n..."
            },
            "account_uuid": "..."
        }
    ]
    """
    with open(json_path) as f:
        data = json.load(f)

    # Claude exports as a single-item list
    if isinstance(data, list) and len(data) > 0:
        data = data[0]

    return ClaudeMemoryData(
        conversations_memory=data.get("conversations_memory", ""),
        project_memories=data.get("project_memories", {}),
        account_uuid=data.get("account_uuid", "")
    )


def extract_facts_from_markdown(
    markdown_text: str,
    source_label: str = "claude_memory"
) -> List[ExtractedFact]:
    """
    Parse markdown sections into individual facts.

    Claude's markdown format:
    **Section Header**

    Sentence 1. Sentence 2. Sentence 3.

    **Next Section**

    More content.

    Strategy:
    1. Split by ## or ** headers to identify sections
    2. For each section, split content into sentences
    3. Each sentence becomes a fact
    4. Classify fact_type based on section header
    """
    facts = []

    # Split by markdown headers (both ** and ##)
    # Pattern: **Header** or ## Header
    # Result: ['', header1, None, content1, header2, None, content2, ...]
    section_pattern = r'(?:^|\n)(?:\*\*([^*]+)\*\*|##\s+([^\n]+))\n'
    sections = re.split(section_pattern, markdown_text)

    current_section = "general"
    current_fact_type = FactType.GENERAL

    # Process sections - they come in groups of 3: (header_group1, header_group2, content)
    i = 0
    while i < len(sections):
        # Skip empty first element
        if i == 0 and not sections[i].strip():
            i += 1
            continue

        # Check if we have a header + content group (need at least 3 elements)
        if i + 2 < len(sections):
            header1 = sections[i]
            header2 = sections[i + 1]
            content = sections[i + 2]

            # One of header1/header2 will be the actual header (other is None)
            header = (header1 or header2 or "").strip()

            if header:
                # Update current section context
                current_section = header.lower()
                current_fact_type = classify_section(current_section)

                # Extract sentences from content
                if content and len(content.strip()) > 20:
                    sentences = split_into_sentences(content)

                    for sentence in sentences:
                        # Create fact
                        fact = ExtractedFact(
                            content=sentence.strip(),
                            fact_type=current_fact_type.value,
                            confidence=0.9,  # High confidence - Claude extracted this
                            source_text=sentence,
                            context=f"section={current_section},source={source_label}"
                        )
                        facts.append(fact)

                # Move to next group (skip 3 elements: header1, header2, content)
                i += 3
            else:
                i += 1
        else:
            i += 1

    return facts


def classify_section(section_name: str) -> FactType:
    """
    Map section header to FactType enum.

    Examples:
    - "work context" -> PROFESSIONAL
    - "personal context" -> PERSONAL
    - "top of mind" -> PROJECT
    - "brief history" -> GENERAL
    """
    section_lower = section_name.lower().strip()

    # Direct match
    if section_lower in SECTION_TO_FACT_TYPE:
        return SECTION_TO_FACT_TYPE[section_lower]

    # Partial match
    for key, fact_type in SECTION_TO_FACT_TYPE.items():
        if key in section_lower or section_lower in key:
            return fact_type

    # Default
    return FactType.GENERAL


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    Strategy: Split on '. ', '! ', '? ' but not on abbreviations like 'Dr. '
    """
    # Replace newlines with spaces for paragraph text
    text = text.replace('\n', ' ')

    # Remove subsection italic markers like "*Recent months*"
    # These appear inline in Claude's memory format
    text = re.sub(r'\*([^*]{5,30})\*', r'\1:', text)

    # Split on sentence boundaries - look for period/question/exclamation followed by space and capital
    # This avoids splitting on abbreviations or decimals
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    sentences = []
    for part in parts:
        part = part.strip()
        # Skip very short fragments or meta-text
        if len(part) > 20 and not is_meta_text(part):
            # Ensure it ends with punctuation
            if not part.endswith(('.', '!', '?')):
                part += '.'
            sentences.append(part)

    return sentences


def is_meta_text(sentence: str) -> bool:
    """
    Check if sentence is meta-text (section labels, formatting) vs actual content.

    Examples of meta-text to skip:
    - "Recent months"
    - "Earlier context"
    - "Long-term background"
    - "*Recent months*" (italic subsection headers)
    - Very short labels
    """
    sentence_stripped = sentence.strip()

    # Skip very short text
    if len(sentence_stripped) < 10:
        return True

    # Remove markdown formatting for checking
    cleaned = re.sub(r'[\*_#]+', '', sentence_stripped).strip()

    meta_patterns = [
        r'^(?:recent|earlier|long-term|other)\s+(?:months|context|background|instructions)\.?$',
        r'^[A-Z][a-z]+\s+(?:months|context|background|history)\.?$',  # "Recent months", "Brief history"
        r'^(?:purpose|context|state|horizon|learnings|principles|approach|patterns|tools|resources)$',  # Common subsection words
    ]

    cleaned_lower = cleaned.lower()

    for pattern in meta_patterns:
        if re.match(pattern, cleaned_lower):
            return True

    return False


def import_claude_memories(
    memories_json_path: Path,
    include_project_memories: bool = True
) -> Tuple[List[ExtractedFact], Dict[str, List[ExtractedFact]]]:
    """
    Import facts from Claude's memories.json.

    Returns:
        (general_facts, project_facts_by_id)

        general_facts: Facts from conversations_memory
        project_facts_by_id: Dict mapping project_id -> facts for that project
    """
    # Parse JSON
    memory_data = parse_claude_memories(memories_json_path)

    # Extract general facts from conversations_memory
    general_facts = extract_facts_from_markdown(
        memory_data.conversations_memory,
        source_label="claude_conversations_memory"
    )

    # Extract project-specific facts
    project_facts = {}
    if include_project_memories:
        for project_id, project_markdown in memory_data.project_memories.items():
            facts = extract_facts_from_markdown(
                project_markdown,
                source_label=f"claude_project_{project_id[:8]}"
            )
            project_facts[project_id] = facts

    return general_facts, project_facts


def get_fact_stats(facts: List[ExtractedFact]) -> Dict[str, int]:
    """Get breakdown of fact types."""
    stats = {}
    for fact in facts:
        fact_type = fact.fact_type
        stats[fact_type] = stats.get(fact_type, 0) + 1
    return stats
