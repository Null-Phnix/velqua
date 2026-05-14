"""Maintenance commands: forget, maintain, health."""

from ...forgetting.manager import ForgettingManager
from ..base import make_context, print_header, print_health_summary


def cmd_forget(args):
    """Run forgetting cycle."""
    print_header("Forgetting Cycle")

    ctx = make_context(args)

    manager = ForgettingManager(
        episodic_store=ctx.episodic,
        semantic_store=ctx.semantic,
    )

    if args.dry_run:
        print("DRY RUN - No changes will be made")

    stats = manager.run_forgetting_cycle(dry_run=args.dry_run)

    print("\nForgetting cycle results:")
    print(f"  Memories checked: {stats.memories_checked}")
    print(f"  Memories decayed: {stats.memories_decayed}")
    print(f"  Memories garbage collected: {stats.memories_garbage_collected}")
    print(f"  Memories compressed: {stats.memories_compressed}")
    print(f"  Facts checked: {stats.facts_checked}")
    print(f"  Facts garbage collected: {stats.facts_garbage_collected}")
    print(f"  Duration: {stats.duration_seconds:.2f}s")

    return 0


def cmd_maintain(args):
    """Run full maintenance cycle on memory system."""
    print_header("Memory Maintenance")

    ctx = make_context(args)

    manager = ForgettingManager(
        episodic_store=ctx.episodic,
        semantic_store=ctx.semantic,
        gc_threshold=args.gc_threshold,
        compress_threshold=args.compress_threshold,
    )

    # Show pre-maintenance health
    print("Pre-maintenance health:")
    health = manager.get_system_health()
    print_health_summary(health)

    if args.dry_run:
        print("\n[DRY RUN - No changes will be made]\n")

    # Run forgetting cycle
    print("\nRunning forgetting cycle...")
    stats = manager.run_forgetting_cycle(dry_run=args.dry_run)

    print("\nMaintenance results:")
    print(f"  Memories checked: {stats.memories_checked}")
    print(f"  Decayed (importance reduced): {stats.memories_decayed}")
    print(f"  Compressed (details reduced): {stats.memories_compressed}")
    print(f"  Garbage collected (removed): {stats.memories_garbage_collected}")
    print(f"  Facts checked: {stats.facts_checked}")
    print(f"  Facts garbage collected: {stats.facts_garbage_collected}")
    print(f"  Duration: {stats.duration_seconds:.2f}s")

    # Show post-maintenance health if not dry run
    if not args.dry_run:
        print("\nPost-maintenance health:")
        health = manager.get_system_health()
        print_health_summary(health)

    # Show at-risk memories if requested
    if args.show_at_risk:
        print("\nAt-risk memories (will be garbage collected soon):")
        episodes = ctx.episodic.list_all(limit=1000)
        at_risk = []
        for ep in episodes:
            mem_health = manager.get_memory_health(ep)
            if mem_health.is_at_risk:
                at_risk.append((ep, mem_health))

        if at_risk:
            for ep, mem_health in at_risk[:10]:
                topic = ep.topic or 'Untitled'
                days = (f"{mem_health.days_until_forgotten:.1f} days"
                        if mem_health.days_until_forgotten else "N/A")
                print(f"  [{mem_health.current_strength:.2f}] {topic[:50]} (forgotten in: {days})")
            if len(at_risk) > 10:
                print(f"  ... and {len(at_risk) - 10} more")
        else:
            print("  No at-risk memories found.")

    return 0


def cmd_health(args):
    """Show memory system health."""
    print_header("Memory System Health")

    ctx = make_context(args)

    manager = ForgettingManager(
        episodic_store=ctx.episodic,
        semantic_store=ctx.semantic,
    )

    health = manager.get_system_health()
    print_health_summary(health)

    # Show health distribution as visual bar
    total = health["total_episodes"]
    if total > 0:
        print("\nHealth Distribution:")
        bar_width = 40

        categories = [
            ("Healthy", health["healthy"]),
            ("Aging", health["aging"]),
            ("At Risk", health["at_risk"]),
            ("Forgotten", health["forgotten"]),
        ]

        for name, count in categories:
            pct = (count / total) * 100
            bar_len = int((count / total) * bar_width)
            bar = "\u2588" * bar_len
            print(f"  {name:12} {bar:40} {count:4} ({pct:5.1f}%)")

    # Show access stats
    access_stats = ctx.backend.get_access_stats()
    print("\nAccess Activity:")
    print(f"  Total episode accesses: {access_stats['total_episode_accesses']}")
    print(f"  Recent accesses (7d): {access_stats['recent_episode_accesses']}")

    # Show recommendations
    print("\nRecommendations:")
    if health["at_risk"] > 0:
        print(f"  - {health['at_risk']} memories at risk. Consider reviewing important ones.")
    if health["forgotten"] > 0:
        print(f"  - {health['forgotten']} memories below threshold. Run 'maintain' to clean up.")
    if health["low_confidence_facts"] > health["high_confidence_facts"]:
        print("  - Many low-confidence facts. Consider fact verification.")
    if health["total_episodes"] == 0:
        print("  - No episodes in database. Import some data to get started.")
    if health["at_risk"] == 0 and health["forgotten"] == 0:
        print("  - Memory system is healthy. No immediate action needed.")

    # Detailed memory health if requested
    if args.detailed:
        print("\nDetailed Memory Health:")
        episodes = ctx.episodic.list_all(limit=args.limit)
        for ep in episodes:
            mem_health = manager.get_memory_health(ep)
            topic = ep.topic or 'Untitled'
            status = "RISK" if mem_health.is_at_risk else "OK"
            print(f"  [{status:4}] [{mem_health.current_strength:.2f}] {topic[:50]}")
            if args.verbose:
                print(f"         {mem_health.recommendation}")

    return 0


def cmd_rebuild_fts(args):
    """Rebuild FTS5 full-text search indexes."""
    print_header("FTS5 Index Rebuild")

    ctx = make_context(args)
    counts = ctx.backend.rebuild_fts()

    print(f"  Episodes re-indexed: {counts['episodes']}")
    print(f"  Facts re-indexed: {counts['facts']}")
    print("\nFTS indexes rebuilt successfully.")
    return 0


def register(subparsers):
    """Register maintenance commands with argparse."""
    # Rebuild FTS
    rebuild_p = subparsers.add_parser("rebuild-fts", help="Rebuild FTS5 search indexes")
    rebuild_p.add_argument("--db", default="anamnesis.db", help="Database path")

    # Forget
    forget_p = subparsers.add_parser("forget", help="Run forgetting cycle")
    forget_p.add_argument("--db", default="anamnesis.db", help="Database path")
    forget_p.add_argument("--dry-run", action="store_true", help="Don't make changes")

    # Maintain
    maintain_p = subparsers.add_parser("maintain", help="Run full maintenance cycle")
    maintain_p.add_argument("--db", default="anamnesis.db", help="Database path")
    maintain_p.add_argument("--dry-run", action="store_true", help="Don't make changes")
    maintain_p.add_argument("--gc-threshold", type=float, default=0.05,
                            help="Garbage collection threshold (0.0-1.0)")
    maintain_p.add_argument("--compress-threshold", type=float, default=0.2,
                            help="Compression threshold (0.0-1.0)")
    maintain_p.add_argument("--show-at-risk", action="store_true",
                            help="Show at-risk memories after maintenance")

    # Health
    health_p = subparsers.add_parser("health", help="Show memory system health")
    health_p.add_argument("--db", default="anamnesis.db", help="Database path")
    health_p.add_argument("--detailed", "-d", action="store_true",
                          help="Show detailed per-memory health")
    health_p.add_argument("--limit", type=int, default=20,
                          help="Max memories to show in detailed view")
    health_p.add_argument("--verbose", "-v", action="store_true",
                          help="Show recommendations for each memory")

    return {
        "rebuild-fts": cmd_rebuild_fts,
        "forget": cmd_forget,
        "maintain": cmd_maintain,
        "health": cmd_health,
    }
