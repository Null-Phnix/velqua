"""Structure commands: merge, topics, chains, chunk, dedup."""

from ...chunking import ConversationChunker
from ...continuation import ContinuationDetector
from ...dedup import DuplicateDetector
from ..base import make_context, print_header


def cmd_merge(args):
    """Find and merge similar episodes."""
    print_header("Episode Merge")

    ctx = make_context(args)
    episodes = ctx.episodic.list_all(limit=1000)
    print(f"Analyzing {len(episodes)} episodes for duplicates...\n")

    if not episodes:
        print("No episodes to analyze.")
        return 0

    from ...consolidation.merger import EpisodeMerger

    merger = EpisodeMerger(
        similarity_threshold=args.threshold,
        temporal_window_hours=args.window,
    )

    candidates = merger.find_merge_candidates(episodes)

    if not candidates:
        print("No merge candidates found.")
        return 0

    print(f"Found {len(candidates)} potential merge candidates:\n")

    for i, candidate in enumerate(candidates[:args.limit], 1):
        ep1 = candidate.episode1
        ep2 = candidate.episode2

        topic1 = ep1.topic or 'Untitled'
        topic2 = ep2.topic or 'Untitled'
        date1 = ep1.started_at.strftime('%Y-%m-%d') if ep1.started_at else 'N/A'
        date2 = ep2.started_at.strftime('%Y-%m-%d') if ep2.started_at else 'N/A'

        print(f"{i}. Similarity: {candidate.similarity:.0%}")
        print(f"   Episode A: [{date1}] {topic1[:50]}")
        print(f"   Episode B: [{date2}] {topic2[:50]}")
        print(f"   Reason: {candidate.merge_reason}")
        if candidate.common_keywords:
            print(f"   Keywords: {', '.join(candidate.common_keywords[:5])}")
        print()

    if len(candidates) > args.limit:
        print(f"... and {len(candidates) - args.limit} more candidates")

    if args.auto:
        if args.dry_run:
            print("\n[DRY RUN - No changes will be made]\n")
            print(f"Would merge top {min(args.max_merges, len(candidates))} candidate pairs.")
            return 0

        print(f"\nAuto-merging top {min(args.max_merges, len(candidates))} candidates...")
        results = merger.auto_merge(episodes, max_merges=args.max_merges)

        merged_count = 0
        for result in results:
            ctx.episodic.save(result.merged_episode)
            for source_id in result.source_ids:
                if source_id != result.merged_episode.id:
                    ctx.episodic.delete(source_id)
            merged_count += 1
            print(f"  Merged {result.source_ids} -> {result.merged_episode.id}")
            print(f"    Preserved {result.preserved_messages} messages,"
                  f" removed {result.removed_duplicates} duplicates")

        print(f"\nMerged {merged_count} episode pairs.")

    return 0


def cmd_topics(args):
    """Analyze and cluster episodes by topic."""
    print_header("Topic Analysis")

    ctx = make_context(args)
    episodes = ctx.episodic.list_all(limit=1000)
    print(f"Analyzing {len(episodes)} episodes...\n")

    if not episodes:
        print("No episodes to analyze.")
        return 0

    from ...topics import EpisodeClusterer

    clusterer = EpisodeClusterer(min_similarity=args.threshold)
    clusters = clusterer.cluster(episodes, max_clusters=args.max_clusters)

    stats = clusterer.get_cluster_stats(clusters)
    print(f"Clusters: {stats['num_clusters']}")
    print(f"Average size: {stats['avg_cluster_size']:.1f} episodes")
    print("\nBy category:")
    for cat, count in stats['by_category'].items():
        if count > 0:
            print(f"  {cat.title()}: {count} clusters")

    print(f"\n{'='*60}\n")

    for i, cluster in enumerate(clusters[:args.limit], 1):
        print(f"{i}. {cluster.name} [{cluster.category}] ({len(cluster.episodes)} episodes)")
        print(f"   Keywords: {', '.join(cluster.common_keywords[:5])}")

        if args.verbose:
            for ep in cluster.episodes[:3]:
                date = ep.started_at.strftime('%Y-%m-%d') if ep.started_at else 'N/A'
                print(f"     [{date}] {ep.topic or 'Untitled'}")
            if len(cluster.episodes) > 3:
                print(f"     ... and {len(cluster.episodes) - 3} more")
        print()

    return 0


def cmd_chains(args):
    """Detect and display conversation chains."""
    print_header("Conversation Chains")

    ctx = make_context(args)

    detector = ContinuationDetector(
        ctx.episodic,
        min_confidence=args.threshold,
        temporal_window_hours=args.window,
    )

    if args.auto_detect:
        print("Auto-detecting conversation chains...")
        links = detector.auto_detect_chains(threshold=args.threshold)
        print(f"Created {links} continuation links.\n")

    if args.episode:
        chain = detector.get_episode_chain(args.episode)
        if not chain:
            print(f"Episode not found: {args.episode}")
            return 1

        print(f"Chain for episode: {args.episode}")
        print(f"Chain length: {len(chain)} episodes\n")

        for i, ep in enumerate(chain, 1):
            date = ep.started_at.strftime('%Y-%m-%d %H:%M') if ep.started_at else 'N/A'
            topic = ep.topic or 'Untitled'
            print(f"  {i}. [{date}] {topic}")
            print(f"     ID: {ep.id}")
            if ep.summary:
                print(f"     Summary: {ep.summary[:100]}...")
            print()
    else:
        all_episodes = ctx.episodic.list_all(limit=100)
        visited = set()
        chain_count = 0

        for ep in all_episodes:
            if ep.id in visited:
                continue

            chain = detector.get_episode_chain(ep.id)
            if len(chain) > 1:
                chain_count += 1
                print(f"Chain {chain_count}: {len(chain)} episodes")

                for ep_chain in chain:
                    visited.add(ep_chain.id)
                    date = ep_chain.started_at.strftime('%Y-%m-%d') if ep_chain.started_at else 'N/A'
                    topic = ep_chain.topic or 'Untitled'
                    print(f"  [{date}] {topic}")
                print()

        if chain_count == 0:
            print("No conversation chains detected.")
            print("Tip: Use --auto-detect to scan for chains.")

    return 0


def cmd_chunk(args):
    """Chunk a long conversation into topic-based episodes."""
    print_header("Conversation Chunking")

    ctx = make_context(args)

    chunker = ConversationChunker(
        min_chunk_size=args.min_size,
        max_chunk_size=args.max_size,
        similarity_threshold=args.threshold,
    )

    if args.episode:
        episode = ctx.episodic.get(args.episode)
        if not episode:
            print(f"Episode not found: {args.episode}")
            return 1

        print(f"Chunking episode: {episode.id}")
        print(f"Original messages: {len(episode.messages)}")

        result = chunker.chunk_conversation(
            episode.messages,
            source_id=episode.source_id,
            base_timestamp=episode.started_at,
        )

        print(f"\nChunks created: {result.chunk_count}")
        print(f"Boundaries detected: {len(result.boundaries)}")

        for i, boundary in enumerate(result.boundaries):
            print(f"\n  Boundary {i+1} at message {boundary.message_index}:")
            print(f"    Confidence: {boundary.confidence:.2f}")
            print(f"    Reason: {boundary.reason}")

        if args.save:
            print("\nSaving chunks to database...")
            for chunk in result.chunks:
                ctx.episodic.save(chunk)
                print(f"  Saved: {chunk.id} - {chunk.topic}")
            print(f"\nSaved {len(result.chunks)} chunks.")
        else:
            print("\nPreview (use --save to persist):")
            for chunk in result.chunks:
                print(f"  [{chunk.metadata.get('chunk_index')}] {chunk.topic}")
                print(f"      Messages: {len(chunk.messages)}")
    else:
        all_episodes = ctx.episodic.list_all(limit=100)
        candidates = [ep for ep in all_episodes if len(ep.messages) > args.max_size]

        if not candidates:
            print(f"No episodes found with more than {args.max_size} messages.")
            return 0

        print(f"Found {len(candidates)} episodes that could be chunked:\n")
        for ep in candidates[:10]:
            date = ep.started_at.strftime('%Y-%m-%d') if ep.started_at else 'N/A'
            print(f"  [{date}] {ep.topic or 'Untitled'}")
            print(f"      ID: {ep.id}")
            print(f"      Messages: {len(ep.messages)}")
            print()

        print("Use --episode <ID> to chunk a specific episode.")

    return 0


def cmd_dedup(args):
    """Detect and handle duplicate memories."""
    print_header("Duplicate Detection")

    ctx = make_context(args)

    detector = DuplicateDetector(
        episodic_store=ctx.episodic,
        semantic_store=ctx.semantic,
        similarity_threshold=args.threshold,
        temporal_window_hours=args.window,
    )

    if args.stats:
        stats = detector.get_stats()

        print("Duplicate Statistics:")
        print(f"  Episodes scanned: {stats.episodes_scanned}")
        print(f"  Facts scanned: {stats.facts_scanned}")
        print(f"  Episode duplicates found: {stats.episode_duplicates}")
        print(f"  Fact duplicates found: {stats.fact_duplicates}")
        if stats.highest_similarity > 0:
            print(f"  Highest similarity: {stats.highest_similarity:.0%}")
            print(f"  Average similarity: {stats.avg_similarity:.0%}")
        return 0

    if args.dedupe:
        if args.dry_run:
            print("[DRY RUN - No changes will be made]\n")

        memory_type = args.type or "all"

        if memory_type in ("all", "episode"):
            print("Deduplicating episodes...")
            ep_results = detector.dedupe_episodes(dry_run=args.dry_run)

            if ep_results["duplicates_found"] == 0:
                print("  No duplicate episodes found.")
            else:
                print(f"  Found {ep_results['duplicates_found']} duplicate pairs")
                if args.dry_run:
                    print(f"  Would remove: {len(ep_results['removed'])} episodes")
                else:
                    print(f"  Removed: {len(ep_results['removed'])} episodes")

                if args.verbose:
                    for kept, removed in zip(ep_results["kept"][:5], ep_results["removed"][:5]):
                        print(f"    Kept: {kept}")
                        print(f"    Removed: {removed}")
            print()

        if memory_type in ("all", "fact"):
            print("Deduplicating facts...")
            fact_results = detector.dedupe_facts(dry_run=args.dry_run)

            if fact_results["duplicates_found"] == 0:
                print("  No duplicate facts found.")
            else:
                print(f"  Found {fact_results['duplicates_found']} duplicate pairs")
                if args.dry_run:
                    print(f"  Would remove: {len(fact_results['removed'])} facts")
                else:
                    print(f"  Removed: {len(fact_results['removed'])} facts")

        return 0

    # Default: find and show duplicates
    memory_type = args.type

    if memory_type in (None, "episode"):
        print("Scanning for duplicate episodes...")
        episode_dups = detector.find_episode_duplicates(limit=args.limit)

        if not episode_dups:
            print("  No duplicate episodes found.\n")
        else:
            print(f"  Found {len(episode_dups)} potential duplicates:\n")

            for i, dup in enumerate(episode_dups[:args.limit], 1):
                ep1 = ctx.episodic.get(dup.memory1_id)
                ep2 = ctx.episodic.get(dup.memory2_id)

                topic1 = ep1.topic if ep1 else "Unknown"
                topic2 = ep2.topic if ep2 else "Unknown"

                print(f"  {i}. Similarity: {dup.similarity:.0%}")
                print(f"     Episode A: {topic1}")
                print(f"       ID: {dup.memory1_id}")
                print(f"     Episode B: {topic2}")
                print(f"       ID: {dup.memory2_id}")
                print(f"     Reasons: {', '.join(dup.match_reasons)}")
                print(f"     Recommendation: Keep {dup.keep_recommendation}")
                print(f"       ({dup.recommendation_reason})")
                print()

    if memory_type in (None, "fact"):
        print("Scanning for duplicate facts...")
        fact_dups = detector.find_fact_duplicates(limit=args.limit)

        if not fact_dups:
            print("  No duplicate facts found.\n")
        else:
            print(f"  Found {len(fact_dups)} potential duplicates:\n")

            for i, dup in enumerate(fact_dups[:args.limit], 1):
                fact1 = ctx.semantic.get(dup.memory1_id)
                fact2 = ctx.semantic.get(dup.memory2_id)

                content1 = fact1.content[:50] if fact1 else "Unknown"
                content2 = fact2.content[:50] if fact2 else "Unknown"

                print(f"  {i}. Similarity: {dup.similarity:.0%}")
                print(f"     Fact A: {content1}...")
                print(f"       ID: {dup.memory1_id}")
                print(f"     Fact B: {content2}...")
                print(f"       ID: {dup.memory2_id}")
                print(f"     Reasons: {', '.join(dup.match_reasons)}")
                print(f"     Recommendation: Keep {dup.keep_recommendation}")
                print(f"       ({dup.recommendation_reason})")
                print()

    print("Use --dedupe to remove duplicates (with --dry-run first).")
    return 0


def register(subparsers):
    """Register structure commands with argparse."""
    # Merge
    merge_p = subparsers.add_parser("merge", help="Find and merge similar episodes")
    merge_p.add_argument("--db", default="anamnesis.db", help="Database path")
    merge_p.add_argument("--threshold", type=float, default=0.6,
                         help="Similarity threshold (0.0-1.0)")
    merge_p.add_argument("--window", type=float, default=24.0,
                         help="Temporal window in hours for proximity bonus")
    merge_p.add_argument("--limit", type=int, default=10, help="Max candidates to show")
    merge_p.add_argument("--auto", action="store_true", help="Automatically merge candidates")
    merge_p.add_argument("--max-merges", type=int, default=5,
                         help="Maximum number of auto-merges")
    merge_p.add_argument("--dry-run", action="store_true",
                         help="Show what would be merged without merging")

    # Topics
    topics_p = subparsers.add_parser("topics", help="Analyze and cluster episodes by topic")
    topics_p.add_argument("--db", default="anamnesis.db", help="Database path")
    topics_p.add_argument("--threshold", type=float, default=0.2,
                          help="Similarity threshold (0.0-1.0)")
    topics_p.add_argument("--max-clusters", type=int, default=20, help="Maximum clusters")
    topics_p.add_argument("--limit", type=int, default=10, help="Clusters to show")
    topics_p.add_argument("--verbose", "-v", action="store_true", help="Show episode details")

    # Chains
    chains_p = subparsers.add_parser("chains", help="Detect and display conversation chains")
    chains_p.add_argument("--db", default="anamnesis.db", help="Database path")
    chains_p.add_argument("--episode", "-e", help="Show chain for specific episode ID")
    chains_p.add_argument("--auto-detect", "-a", action="store_true", help="Auto-detect chains")
    chains_p.add_argument("--threshold", type=float, default=0.4,
                          help="Confidence threshold (0-1)")
    chains_p.add_argument("--window", type=float, default=72, help="Temporal window in hours")

    # Chunk
    chunk_p = subparsers.add_parser("chunk", help="Chunk long conversations into episodes")
    chunk_p.add_argument("--db", default="anamnesis.db", help="Database path")
    chunk_p.add_argument("--episode", "-e", help="Episode ID to chunk")
    chunk_p.add_argument("--min-size", type=int, default=4, help="Minimum messages per chunk")
    chunk_p.add_argument("--max-size", type=int, default=50, help="Maximum messages per chunk")
    chunk_p.add_argument("--threshold", type=float, default=0.3,
                         help="Topic similarity threshold")
    chunk_p.add_argument("--save", "-s", action="store_true", help="Save chunks to database")

    # Dedup
    dedup_p = subparsers.add_parser("dedup", help="Detect and handle duplicate memories")
    dedup_p.add_argument("--db", default="anamnesis.db", help="Database path")
    dedup_p.add_argument("--type", "-t", choices=["episode", "fact"],
                         help="Memory type to check")
    dedup_p.add_argument("--threshold", type=float, default=0.6,
                         help="Similarity threshold (0-1)")
    dedup_p.add_argument("--window", type=float, default=24.0,
                         help="Temporal window in hours")
    dedup_p.add_argument("--limit", type=int, default=10, help="Max duplicates to show")
    dedup_p.add_argument("--stats", "-s", action="store_true",
                         help="Show duplicate statistics only")
    dedup_p.add_argument("--dedupe", action="store_true", help="Actually remove duplicates")
    dedup_p.add_argument("--dry-run", action="store_true", help="Show what would be removed")
    dedup_p.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    return {
        "merge": cmd_merge,
        "topics": cmd_topics,
        "chains": cmd_chains,
        "chunk": cmd_chunk,
        "dedup": cmd_dedup,
    }
