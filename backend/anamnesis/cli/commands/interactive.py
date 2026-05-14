"""Interactive commands: shell, demo, async."""

from ...anamnesis import Anamnesis
from ...forgetting.manager import ForgettingManager
from ...integration import InjectionMode, MemoryInjector
from ...models import EmotionalValence
from ..base import make_context, print_header, valence_to_str


def cmd_shell(args):
    """Run interactive shell mode."""
    print_header("Anamnesis Interactive Shell")

    print("Welcome to the Anamnesis memory system!")
    print("Type 'help' for available commands, 'quit' to exit.\n")

    ctx = make_context(args)

    def shell_help():
        print("""
Available commands:
  search <query>     - Search memories
  recent [n]         - Show n recent episodes (default: 5)
  facts [type]       - List facts (optionally filter by type)
  emotions           - Show emotional distribution
  stats              - Show database statistics
  access             - Show access statistics
  inject <query>     - Generate context injection
  forget             - Run forgetting cycle (dry-run)
  health             - Show memory system health
  maintain           - Run maintenance cycle (dry-run)
  help               - Show this help
  quit/exit          - Exit shell
""")

    def shell_search(query):
        if not query:
            print("Usage: search <query>")
            return

        results = ctx.episodic.search(query, limit=5)
        if results:
            print(f"\nFound {len(results)} episodes:")
            for ep in results:
                date = ep.started_at.strftime('%Y-%m-%d') if ep.started_at else 'N/A'
                valence = valence_to_str(ep.overall_valence)
                print(f"  [{date}] [{valence}] {ep.topic or 'Untitled'}")
                if ep.summary:
                    print(f"    {ep.summary[:80]}...")
        else:
            print("No episodes found.")

        facts = ctx.semantic.search(query, limit=5)
        if facts:
            print(f"\nFound {len(facts)} facts:")
            for f in facts:
                print(f"  [{f.fact_type}] {f.content[:60]}")

    def shell_recent(n_str):
        n = int(n_str) if n_str else 5
        recent = ctx.episodic.get_recent(days=90, limit=n)
        if recent:
            print(f"\nRecent episodes ({len(recent)}):")
            for ep in recent:
                date = ep.started_at.strftime('%Y-%m-%d') if ep.started_at else 'N/A'
                print(f"  [{date}] {ep.topic or 'Untitled'}")
        else:
            print("No recent episodes.")

    def shell_facts(type_filter):
        if type_filter:
            facts = ctx.semantic.get_by_type(type_filter, limit=20)
        else:
            facts = ctx.semantic.list_all(limit=20)

        if facts:
            print(f"\nFacts ({len(facts)}):")
            for f in facts:
                conf = f"[{f.confidence:.0%}]" if f.confidence < 1.0 else ""
                print(f"  [{f.fact_type}] {f.content[:60]} {conf}")
        else:
            print("No facts found.")

    def shell_emotions():
        all_episodes = ctx.episodic.list_all(limit=1000)
        total = len(all_episodes)
        if total == 0:
            print("No episodes.")
            return

        counts = {}
        for valence in EmotionalValence:
            counts[valence] = sum(1 for ep in all_episodes if ep.overall_valence == valence)

        print(f"\nEmotional Distribution ({total} episodes):")
        for valence in [EmotionalValence.VERY_POSITIVE, EmotionalValence.POSITIVE,
                        EmotionalValence.NEUTRAL, EmotionalValence.NEGATIVE,
                        EmotionalValence.VERY_NEGATIVE]:
            count = counts.get(valence, 0)
            pct = (count / total) * 100
            bar = "\u2588" * int(pct / 5)
            label = valence.name.replace("_", " ").title()
            print(f"  {label:15} {bar:20} {count:3} ({pct:5.1f}%)")

    def shell_stats():
        stats = ctx.backend.get_stats()
        print(f"\nDatabase: {args.db}")
        print(f"  Episodes: {stats['episodes']}")
        print(f"  Facts: {stats['facts']}")
        print(f"  Conversations: {stats['conversations']}")

    def shell_access():
        stats = ctx.backend.get_access_stats()
        print("\nAccess Statistics:")
        print(f"  Total episode accesses: {stats['total_episode_accesses']}")
        print(f"  Total fact accesses: {stats['total_fact_accesses']}")
        if stats['most_accessed_episodes']:
            print("\n  Most accessed episodes:")
            for ep in stats['most_accessed_episodes'][:3]:
                topic = ep.get('topic', 'Untitled') or 'Untitled'
                print(f"    [{ep['access_count']:3}x] {topic[:40]}")

    def shell_inject(query):
        if not query:
            print("Usage: inject <query>")
            return

        injector = MemoryInjector(
            episodic_store=ctx.episodic,
            semantic_store=ctx.semantic,
            budget='minimal',
        )

        result = injector.inject_for_query(query, mode=InjectionMode.SYSTEM_PROMPT)
        print(f"\nContext ({result.total_tokens} tokens,"
              f" {result.episodes_used} episodes, {result.facts_used} facts):")
        print("-" * 40)
        text = result.injected_text
        if len(text) > 500:
            text = text[:500] + "\n..."
        print(text)
        print("-" * 40)

    def shell_forget():
        manager = ForgettingManager(episodic_store=ctx.episodic, semantic_store=ctx.semantic)
        stats = manager.run_forgetting_cycle(dry_run=True)
        print("\nForgetting cycle (dry-run):")
        print(f"  Memories checked: {stats.memories_checked}")
        print(f"  Would decay: {stats.memories_decayed}")
        print(f"  Would GC: {stats.memories_garbage_collected}")

    def shell_health():
        manager = ForgettingManager(episodic_store=ctx.episodic, semantic_store=ctx.semantic)
        health = manager.get_system_health()
        total = health["total_episodes"]
        print("\nMemory System Health:")
        print(f"  Episodes: {total}")
        if total > 0:
            print(f"    Healthy: {health['healthy']} ({health['healthy']/total*100:.1f}%)")
            print(f"    Aging: {health['aging']}")
            print(f"    At Risk: {health['at_risk']}")
            print(f"    Forgotten: {health['forgotten']}")
        print(f"  Facts: {health['total_facts']}")
        print(f"    High confidence: {health['high_confidence_facts']}")
        print(f"    Low confidence: {health['low_confidence_facts']}")

    def shell_maintain():
        manager = ForgettingManager(episodic_store=ctx.episodic, semantic_store=ctx.semantic)
        print("\nRunning maintenance (dry-run)...")
        stats = manager.run_forgetting_cycle(dry_run=True)
        print("\nMaintenance results:")
        print(f"  Memories checked: {stats.memories_checked}")
        print(f"  Would decay: {stats.memories_decayed}")
        print(f"  Would compress: {stats.memories_compressed}")
        print(f"  Would GC: {stats.memories_garbage_collected}")
        print(f"  Facts would GC: {stats.facts_garbage_collected}")
        print("\nRun 'anamnesis maintain' outside shell to apply changes.")

    # Main REPL loop
    while True:
        try:
            user_input = input("anamnesis> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        parts = user_input.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        handlers = {
            'help': lambda: shell_help(),
            'search': lambda: shell_search(arg),
            'recent': lambda: shell_recent(arg),
            'facts': lambda: shell_facts(arg),
            'emotions': lambda: shell_emotions(),
            'stats': lambda: shell_stats(),
            'access': lambda: shell_access(),
            'inject': lambda: shell_inject(arg),
            'forget': lambda: shell_forget(),
            'health': lambda: shell_health(),
            'maintain': lambda: shell_maintain(),
        }

        if cmd in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        elif cmd in handlers:
            handlers[cmd]()
        else:
            print(f"Unknown command: {cmd}. Type 'help' for available commands.")

    return 0


def cmd_demo(args):
    """Run interactive demo."""
    print_header("Anamnesis Demo")

    ana = Anamnesis(db_path=args.db)
    stats = ana.get_stats()

    print(f"Database: {args.db}")
    print(f"Episodes: {stats['episodes']}, Facts: {stats['facts']}")
    print()

    if stats['episodes'] == 0 and stats['facts'] == 0:
        print("Database is empty. Import some data first:")
        print("  anamnesis import /path/to/claude/export --type claude")
        return 1

    print("Demo queries:")
    print()

    # Demo 1: Recent memories
    print("1. Recent memories:")
    recent = ana.recall_timeframe("recent", days_back=30)
    for ep in recent[:3]:
        date = ep.started_at.strftime('%Y-%m-%d') if ep.started_at else 'N/A'
        print(f"   [{date}] {ep.topic or 'Untitled'}")
    print()

    # Demo 2: Search
    if stats['facts'] > 0:
        print("2. Known facts:")
        fact_summary = ana.get_fact_summary()
        for ftype, facts in fact_summary.items():
            if facts:
                print(f"   {ftype}: {facts[0][:60]}...")
                break
        print()

    # Demo 3: Context injection
    print("3. Context injection sample:")
    ctx = make_context(args)

    injector = MemoryInjector(
        episodic_store=ctx.episodic,
        semantic_store=ctx.semantic,
        budget='minimal',
    )

    result = injector.inject_for_query(
        query="general context",
        mode=InjectionMode.SYSTEM_PROMPT,
    )

    text = result.injected_text
    if len(text) > 500:
        text = text[:500] + "\n   ..."
    print(f"   {text}")
    print()

    print("Try these commands:")
    print("  anamnesis search 'your query'")
    print("  anamnesis facts")
    print("  anamnesis inject 'query' --format markdown")

    return 0


def cmd_async(args):
    """Show async operations status."""
    print_header("Async Operations")
    print("Async module was removed in v0.2.0 (unused, will rebuild when needed).")
    return 0


def register(subparsers):
    """Register interactive commands with argparse."""
    # Shell
    shell_p = subparsers.add_parser("shell", help="Run interactive shell")
    shell_p.add_argument("--db", default="anamnesis.db", help="Database path")

    # Demo
    demo_p = subparsers.add_parser("demo", help="Run interactive demo")
    demo_p.add_argument("--db", default="anamnesis.db", help="Database path")

    # Async (stub)
    async_p = subparsers.add_parser("async", help="Async operations status and demo")
    async_p.add_argument("--db", default="anamnesis.db", help="Database path")
    async_p.add_argument("--status", action="store_true", help="Show async status and usage")
    async_p.add_argument("--demo", action="store_true", help="Run async demo")

    return {
        "shell": cmd_shell,
        "demo": cmd_demo,
        "async": cmd_async,
    }
