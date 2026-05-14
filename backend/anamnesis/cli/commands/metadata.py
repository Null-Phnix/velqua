"""Metadata commands: tags, link, quality."""

from ...linking import LinkManager, LinkType
from ...quality import QualityScorer
from ...tagging import AutoTagger, TagManager
from ..base import make_context, print_header


def cmd_tags(args):
    """Manage memory tags."""
    print_header("Memory Tags")

    ctx = make_context(args)

    tag_manager = TagManager(ctx.episodic, ctx.semantic)
    auto_tagger = AutoTagger(min_confidence=args.min_confidence)

    if args.add:
        memory_type = args.type or "episode"
        result = tag_manager.add_tag(args.memory_id, args.add, memory_type)
        if result:
            print(f"Added tag '{args.add}' to {memory_type} {args.memory_id}")
        else:
            print(f"Failed to add tag. Memory not found: {args.memory_id}")
        return 0 if result else 1

    elif args.remove:
        memory_type = args.type or "episode"
        result = tag_manager.remove_tag(args.memory_id, args.remove, memory_type)
        if result:
            print(f"Removed tag '{args.remove}' from {memory_type} {args.memory_id}")
        else:
            print(f"Failed to remove tag. Memory not found: {args.memory_id}")
        return 0 if result else 1

    elif args.find:
        results = tag_manager.find_by_tag(args.find, memory_type=args.type)

        if "episodes" in results and results["episodes"]:
            print(f"Episodes with tag '{args.find}' ({len(results['episodes'])}):")
            for ep in results["episodes"][:args.limit]:
                date = ep.started_at.strftime('%Y-%m-%d') if ep.started_at else 'N/A'
                topic = ep.topic or 'Untitled'
                tags_str = ', '.join(ep.tags[:5])
                print(f"  [{date}] {topic}")
                print(f"    Tags: {tags_str}")
            print()

        if "facts" in results and results["facts"]:
            print(f"Facts with tag '{args.find}' ({len(results['facts'])}):")
            for fact in results["facts"][:args.limit]:
                tags_str = ', '.join(fact.tags[:5])
                print(f"  [{fact.fact_type}] {fact.content[:60]}")
                print(f"    Tags: {tags_str}")

        if not results.get("episodes") and not results.get("facts"):
            print(f"No memories found with tag '{args.find}'")

    elif args.auto_tag:
        memory_type = args.type or "episode"
        if memory_type == "episode":
            episode = ctx.episodic.get(args.auto_tag)
            if not episode:
                print(f"Episode not found: {args.auto_tag}")
                return 1

            suggested = auto_tagger.auto_tag_episode(episode)
            if args.apply:
                for tag in suggested:
                    tag_manager.add_tag(args.auto_tag, tag, "episode")
                print(f"Applied {len(suggested)} tags to episode {args.auto_tag}:")
            else:
                print(f"Suggested tags for episode {args.auto_tag}:")

            for tag in suggested:
                print(f"  - {tag}")

        else:
            fact = ctx.semantic.get(args.auto_tag)
            if not fact:
                print(f"Fact not found: {args.auto_tag}")
                return 1

            suggested = auto_tagger.auto_tag_fact(fact)
            if args.apply:
                for tag in suggested:
                    tag_manager.add_tag(args.auto_tag, tag, "fact")
                print(f"Applied {len(suggested)} tags to fact {args.auto_tag}:")
            else:
                print(f"Suggested tags for fact {args.auto_tag}:")

            for tag in suggested:
                print(f"  - {tag}")

    elif args.auto_all:
        print("Auto-tagging untagged memories...")

        episodes = ctx.episodic.list_all(limit=1000)
        facts = ctx.semantic.list_all(limit=1000)

        episode_count = 0
        fact_count = 0

        for ep in episodes:
            if not ep.tags or len(ep.tags) == 0:
                suggested = auto_tagger.auto_tag_episode(ep)
                if suggested:
                    for tag in suggested:
                        tag_manager.add_tag(ep.id, tag, "episode")
                    episode_count += 1
                    if args.verbose:
                        print(f"  Episode: {ep.topic or ep.id[:20]} -> {', '.join(suggested)}")

        for fact in facts:
            if not fact.tags or len(fact.tags) == 0:
                suggested = auto_tagger.auto_tag_fact(fact)
                if suggested:
                    for tag in suggested:
                        tag_manager.add_tag(fact.id, tag, "fact")
                    fact_count += 1
                    if args.verbose:
                        print(f"  Fact: {fact.content[:30]} -> {', '.join(suggested)}")

        print(f"\nTagged {episode_count} episodes and {fact_count} facts")

    elif args.stats:
        stats = tag_manager.get_stats()

        print("Tag Statistics:")
        print(f"  Total tags used: {stats.total_tags}")
        print(f"  Unique tags: {stats.unique_tags}")
        print(f"  Episodes with tags: {stats.episodes_tagged}")
        print(f"  Facts with tags: {stats.facts_tagged}")
        print(f"  Avg tags per episode: {stats.avg_tags_per_episode:.1f}")
        print(f"  Avg tags per fact: {stats.avg_tags_per_fact:.1f}")

        if stats.most_common:
            print("\nMost Common Tags:")
            for tag, count in stats.most_common[:10]:
                print(f"  {tag}: {count}")

    else:
        all_tags = tag_manager.list_all_tags()

        if not all_tags:
            print("No tags found.")
            print("\nTo add tags, use:")
            print("  anamnesis tags --add <tag> --memory-id <id>")
            print("\nTo auto-tag memories:")
            print("  anamnesis tags --auto-all")
            return 0

        print(f"All Tags ({len(all_tags)}):\n")

        for tag, count in all_tags[:args.limit]:
            print(f"  {tag}: {count} memories")

        if len(all_tags) > args.limit:
            print(f"\n  ... and {len(all_tags) - args.limit} more tags")

    return 0


def cmd_link(args):
    """Manage memory links."""
    print_header("Memory Links")

    ctx = make_context(args)
    manager = LinkManager(ctx.backend, ctx.episodic, ctx.semantic)

    if args.create:
        try:
            link_type = LinkType(args.link_type)
        except ValueError:
            print(f"Invalid link type: {args.link_type}")
            print(f"Valid types: {', '.join(t.value for t in LinkType)}")
            return 1

        try:
            link = manager.create_link(
                source_id=args.source,
                source_type=args.source_type,
                target_id=args.target,
                target_type=args.target_type,
                link_type=link_type,
                strength=args.strength,
                note=args.note or "",
            )
            print(f"Created link: {link.id}")
            print(f"  {args.source} ({args.source_type}) -> {args.target} ({args.target_type})")
            print(f"  Type: {link_type.value}")
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    elif args.delete:
        if manager.delete_link(args.delete):
            print(f"Deleted link: {args.delete}")
        else:
            print(f"Link not found: {args.delete}")
            return 1

    elif args.show:
        memory_type = args.type or "episode"
        links = manager.get_all_links(args.show, memory_type)

        if not links:
            print(f"No links found for {memory_type} {args.show}")
            return 0

        print(f"Links for {memory_type} {args.show} ({len(links)}):\n")

        for link in links:
            if link.source_id == args.show:
                direction = "->"
                other_id = link.target_id
                other_type = link.target_type
            else:
                direction = "<-"
                other_id = link.source_id
                other_type = link.source_type

            print(f"  {direction} {other_id} ({other_type})")
            print(f"     Type: {link.link_type.value}")
            print(f"     Strength: {link.strength:.2f}")
            if link.note:
                print(f"     Note: {link.note}")
            print(f"     ID: {link.id}")
            print()

    elif args.path:
        parts = args.path.split(":")
        if len(parts) != 4:
            print("Usage: --path source_id:source_type:target_id:target_type")
            return 1

        source_id, source_type, target_id, target_type = parts
        path = manager.find_path(source_id, source_type, target_id, target_type)

        if path:
            print(f"Path found ({len(path)} hops):\n")
            current = (source_id, source_type)
            for link in path:
                if link.source_id == current[0]:
                    next_id = link.target_id
                    next_type = link.target_type
                else:
                    next_id = link.source_id
                    next_type = link.source_type

                print(f"  {current[0]} ({current[1]})")
                print(f"    -- {link.link_type.value} -->")
                current = (next_id, next_type)

            print(f"  {target_id} ({target_type})")
        else:
            print(f"No path found between {source_id} and {target_id}")

    elif args.auto_link:
        print("Auto-linking memories...")

        cont_count = manager.auto_link_continuations()
        print(f"  Created {cont_count} continuation links")

        source_count = manager.auto_link_fact_sources()
        print(f"  Created {source_count} fact source links")

        print(f"\nTotal: {cont_count + source_count} links created")

    elif args.stats:
        stats = manager.get_stats()

        print("Link Statistics:")
        print(f"  Total links: {stats.total_links}")
        print(f"  Orphaned links: {stats.orphaned_links}")

        if stats.links_by_type:
            print("\nLinks by type:")
            for link_type, count in stats.links_by_type.items():
                print(f"  {link_type}: {count}")

        if stats.most_linked_memories:
            print("\nMost linked memories:")
            for mem_id, mem_type, count in stats.most_linked_memories[:5]:
                print(f"  {mem_id} ({mem_type}): {count} links")

    elif args.cleanup:
        count = manager.cleanup_orphaned_links()
        print(f"Cleaned up {count} orphaned links")

    else:
        links = manager.list_all_links(limit=args.limit)

        if not links:
            print("No links found.")
            print("\nTo create a link:")
            print("  anamnesis link --create --source <id> --target <id> --link-type related_to")
            return 0

        print(f"All Links ({len(links)}):\n")

        for link in links:
            print(f"  {link.source_id} ({link.source_type})")
            print(f"    -- {link.link_type.value} --> {link.target_id} ({link.target_type})")
            if link.note:
                print(f"    Note: {link.note}")
            print()

    return 0


def cmd_quality(args):
    """Assess memory quality."""
    print_header("Memory Quality")

    ctx = make_context(args)
    scorer = QualityScorer()

    if args.memory_id:
        memory_type = args.type or "episode"

        if memory_type == "episode":
            memory = ctx.episodic.get(args.memory_id)
            if not memory:
                print(f"Episode not found: {args.memory_id}")
                return 1
            report = scorer.score_episode(memory)
        else:
            memory = ctx.semantic.get(args.memory_id)
            if not memory:
                print(f"Fact not found: {args.memory_id}")
                return 1
            report = scorer.score_fact(memory)

        print(f"Quality Report: {report.memory_id}")
        print(f"  Type: {report.memory_type}")
        print(f"  Overall Score: {report.overall_score:.2f}")
        print(f"  Quality Level: {report.quality_level.value.upper()}")
        print()
        print("  Dimension Scores:")
        print(f"    Completeness: {report.completeness_score:.2f}")
        print(f"    Richness:     {report.richness_score:.2f}")
        print(f"    Reliability:  {report.reliability_score:.2f}")
        print(f"    Activity:     {report.activity_score:.2f}")

        if report.missing_fields:
            print(f"\n  Missing Fields: {', '.join(report.missing_fields)}")

        if report.suggestions:
            print("\n  Suggestions:")
            for s in report.suggestions:
                print(f"    - {s}")

        if args.verbose and report.metrics:
            print("\n  Metrics:")
            for key, value in report.metrics.items():
                print(f"    {key}: {value}")

        return 0

    # Get all memories and show stats
    episodes = ctx.episodic.list_all(limit=args.limit)
    facts = ctx.semantic.list_all(limit=args.limit)

    stats = scorer.get_stats(episodes, facts)

    print("Quality Statistics:")
    print(f"  Total memories: {stats.total_memories}")
    print(f"  Average quality: {stats.avg_quality:.2f}")
    print()

    print("Quality Distribution:")
    level_order = ["excellent", "good", "fair", "poor", "low"]
    total = stats.total_memories or 1

    for level in level_order:
        count = stats.quality_distribution.get(level, 0)
        pct = (count / total) * 100
        bar_len = int(pct / 5)
        bar = "\u2588" * bar_len
        print(f"  {level.capitalize():10} {bar:20} {count:4} ({pct:5.1f}%)")

    if args.lowest:
        print("\nLowest Quality Memories:")
        for mem_id, score in stats.lowest_quality[:5]:
            print(f"  [{score:.2f}] {mem_id}")

    if args.highest:
        print("\nHighest Quality Memories:")
        for mem_id, score in stats.highest_quality[:5]:
            print(f"  [{score:.2f}] {mem_id}")

    if args.issues and stats.common_issues:
        print("\nCommon Issues:")
        for issue, count in sorted(stats.common_issues.items(), key=lambda x: -x[1])[:10]:
            print(f"  {issue}: {count}")

    return 0


def register(subparsers):
    """Register metadata commands with argparse."""
    # Tags
    tags_p = subparsers.add_parser("tags", help="Manage memory tags")
    tags_p.add_argument("--db", default="anamnesis.db", help="Database path")
    tags_p.add_argument("--memory-id", "-m", dest="memory_id",
                        help="Memory ID for tag operations")
    tags_p.add_argument("--type", "-t", choices=["episode", "fact"], help="Memory type")
    tags_p.add_argument("--add", help="Add a tag to memory")
    tags_p.add_argument("--remove", help="Remove a tag from memory")
    tags_p.add_argument("--find", "-f", help="Find memories with tag")
    tags_p.add_argument("--auto-tag", help="Auto-generate tags for memory ID")
    tags_p.add_argument("--auto-all", action="store_true",
                        help="Auto-tag all untagged memories")
    tags_p.add_argument("--apply", action="store_true", help="Apply auto-generated tags")
    tags_p.add_argument("--stats", "-s", action="store_true", help="Show tag statistics")
    tags_p.add_argument("--limit", type=int, default=20, help="Max results to show")
    tags_p.add_argument("--min-confidence", type=float, default=0.3,
                        help="Min confidence for auto-tagging")
    tags_p.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Link
    link_p = subparsers.add_parser("link", help="Manage memory links")
    link_p.add_argument("--db", default="anamnesis.db", help="Database path")
    link_p.add_argument("--create", action="store_true", help="Create a new link")
    link_p.add_argument("--source", help="Source memory ID")
    link_p.add_argument("--source-type", default="episode", choices=["episode", "fact"])
    link_p.add_argument("--target", help="Target memory ID")
    link_p.add_argument("--target-type", default="episode", choices=["episode", "fact"])
    link_p.add_argument("--link-type", default="related_to",
                        help="Link type: related_to, continues, contradicts, supersedes,"
                             " see_also, derived_from, references")
    link_p.add_argument("--strength", type=float, default=1.0, help="Link strength (0-1)")
    link_p.add_argument("--note", help="Optional note for the link")
    link_p.add_argument("--delete", help="Delete a link by ID")
    link_p.add_argument("--show", help="Show links for a memory ID")
    link_p.add_argument("--type", "-t", choices=["episode", "fact"],
                        help="Memory type for --show")
    link_p.add_argument("--path",
                        help="Find path: source_id:source_type:target_id:target_type")
    link_p.add_argument("--auto-link", action="store_true",
                        help="Auto-create links from metadata")
    link_p.add_argument("--stats", "-s", action="store_true", help="Show link statistics")
    link_p.add_argument("--cleanup", action="store_true", help="Clean up orphaned links")
    link_p.add_argument("--limit", type=int, default=20, help="Max links to show")

    # Quality
    quality_p = subparsers.add_parser("quality", help="Assess memory quality")
    quality_p.add_argument("--db", default="anamnesis.db", help="Database path")
    quality_p.add_argument("--memory-id", "-m", dest="memory_id",
                           help="Score specific memory")
    quality_p.add_argument("--type", "-t", choices=["episode", "fact"], help="Memory type")
    quality_p.add_argument("--limit", type=int, default=1000,
                           help="Max memories to analyze")
    quality_p.add_argument("--lowest", action="store_true",
                           help="Show lowest quality memories")
    quality_p.add_argument("--highest", action="store_true",
                           help="Show highest quality memories")
    quality_p.add_argument("--issues", action="store_true", help="Show common issues")
    quality_p.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    return {
        "tags": cmd_tags,
        "link": cmd_link,
        "quality": cmd_quality,
    }
