"""Analysis commands: stats, analytics, emotions, access."""

from ...analytics import MemoryAnalyzer
from ...models import EmotionalValence
from ..base import make_context, parse_emotion, print_header


def cmd_stats(args):
    """Show database statistics."""
    print_header("Memory Statistics")

    ctx = make_context(args)

    print(f"Database: {args.db}")
    print("\nCounts:")
    print(f"  Episodes: {ctx.episodic.count()}")
    print(f"  Facts: {ctx.semantic.count()}")
    print(f"  Raw conversations: {ctx.backend.get_stats()['conversations']}")

    # Episode date range
    all_episodes = ctx.episodic.list_all(limit=1000)
    if all_episodes:
        dates = [ep.started_at for ep in all_episodes if ep.started_at]
        if dates:
            print("\nEpisode date range:")
            print(f"  Oldest: {min(dates).strftime('%Y-%m-%d')}")
            print(f"  Newest: {max(dates).strftime('%Y-%m-%d')}")

    # Fact types
    all_facts = ctx.semantic.list_all(limit=1000)
    if all_facts:
        types = {}
        for f in all_facts:
            types[f.fact_type] = types.get(f.fact_type, 0) + 1
        print("\nFact types:")
        for ftype, count in sorted(types.items()):
            print(f"  {ftype}: {count}")

    # Recent episodes
    recent = ctx.episodic.get_recent(days=30, limit=5)
    if recent:
        print("\nRecent episodes:")
        for ep in recent:
            date = ep.started_at.strftime('%Y-%m-%d') if ep.started_at else 'Unknown'
            topic = ep.topic or 'Untitled'
            print(f"  [{date}] {topic[:50]}")

    return 0


def cmd_access(args):
    """Show memory access statistics."""
    print_header("Memory Access Statistics")

    ctx = make_context(args)
    stats = ctx.backend.get_access_stats()

    print("Total Accesses:")
    print(f"  Episodes: {stats['total_episode_accesses']}")
    print(f"  Facts: {stats['total_fact_accesses']}")

    print("\nRecently Accessed (last 7 days):")
    print(f"  Episodes: {stats['recent_episode_accesses']}")
    print(f"  Facts: {stats['recent_fact_accesses']}")

    if stats['most_accessed_episodes']:
        print("\nMost Accessed Episodes:")
        for ep in stats['most_accessed_episodes'][:5]:
            topic = ep.get('topic', 'Untitled') or 'Untitled'
            print(f"  [{ep['access_count']:3}x] {topic[:50]} (importance: {ep['importance']:.2f})")

    if stats['most_accessed_facts']:
        print("\nMost Accessed Facts:")
        for fact in stats['most_accessed_facts'][:5]:
            print(f"  [{fact['access_count']:3}x] {fact['content'][:50]}...")

    return 0


def cmd_emotions(args):
    """Show emotional valence statistics and episodes."""
    print_header("Emotional Memory Analysis")

    ctx = make_context(args)

    all_episodes = ctx.episodic.list_all(limit=1000)
    total = len(all_episodes)

    if total == 0:
        print("No episodes in database.")
        return 0

    # Count by valence
    counts = {}
    for valence in EmotionalValence:
        counts[valence] = 0

    for ep in all_episodes:
        if ep.overall_valence in counts:
            counts[ep.overall_valence] += 1

    print(f"Total episodes: {total}\n")
    print("Emotional Distribution:")

    max_count = max(counts.values()) if counts.values() else 1
    bar_width = 30

    for valence in [EmotionalValence.VERY_POSITIVE, EmotionalValence.POSITIVE,
                    EmotionalValence.NEUTRAL, EmotionalValence.NEGATIVE,
                    EmotionalValence.VERY_NEGATIVE]:
        count = counts.get(valence, 0)
        pct = (count / total) * 100 if total > 0 else 0
        bar_len = int((count / max_count) * bar_width) if max_count > 0 else 0
        bar = "\u2588" * bar_len
        label = valence.name.replace("_", " ").title()
        print(f"  {label:15} {bar:30} {count:3} ({pct:5.1f}%)")

    # Show sample episodes for requested emotion
    if args.emotion:
        valence = parse_emotion(args.emotion)
        if valence:
            episodes = ctx.episodic.get_emotional(valence, limit=args.limit)
            print(f"\n{args.emotion.title()} Episodes ({len(episodes)}):")
            for ep in episodes:
                date = ep.started_at.strftime('%Y-%m-%d') if ep.started_at else 'N/A'
                topic = ep.topic or 'Untitled'
                print(f"  [{date}] {topic}")

    return 0


def cmd_analytics(args):
    """Generate comprehensive memory analytics report."""
    print_header("Memory Analytics Report")

    ctx = make_context(args)
    analyzer = MemoryAnalyzer(ctx.episodic, ctx.semantic, ctx.backend)

    if args.quick:
        stats = analyzer.get_quick_stats()
        print(f"Total Episodes: {stats['total_episodes']}")
        print(f"Total Facts: {stats['total_facts']}")
        print(f"Average Importance: {stats['avg_importance']:.2f}")
        print(f"Average Confidence: {stats['avg_confidence']:.2f}")

        print("\nFact Types:")
        for ft, count in stats['fact_types'].items():
            print(f"  {ft}: {count}")

        print("\nEmotional Distribution:")
        for emotion, count in stats['emotions'].items():
            print(f"  {emotion}: {count}")
        return 0

    # Full report
    print("Generating comprehensive report...")
    report = analyzer.generate_report()

    print(f"\nGenerated: {report.generated_at.strftime('%Y-%m-%d %H:%M')}")
    print(f"Memory Span: {report.memory_span_days} days")
    print(f"Total Episodes: {report.total_episodes}")
    print(f"Total Facts: {report.total_facts}")

    # Memory Health
    print("\n--- Memory Health ---")
    total_memories = (report.healthy_memories + report.aging_memories +
                      report.at_risk_memories + report.forgotten_memories)
    if total_memories > 0:
        print(f"  Healthy (>70%):   {report.healthy_memories:4}"
              f" ({report.healthy_memories/total_memories*100:5.1f}%)")
        print(f"  Aging (20-70%):   {report.aging_memories:4}"
              f" ({report.aging_memories/total_memories*100:5.1f}%)")
        print(f"  At Risk (5-20%):  {report.at_risk_memories:4}"
              f" ({report.at_risk_memories/total_memories*100:5.1f}%)")
        print(f"  Forgotten (<5%):  {report.forgotten_memories:4}"
              f" ({report.forgotten_memories/total_memories*100:5.1f}%)")

    # Topics
    print("\n--- Top Topics ---")
    for i, topic in enumerate(report.top_topics[:5], 1):
        print(f"  {i}. {topic.topic} ({topic.count} episodes,"
              f" avg importance: {topic.avg_importance:.2f})")
    print(f"\nTopic Diversity: {report.topic_diversity:.2f}")

    # Emotional Balance
    print("\n--- Emotional Patterns ---")
    for es in report.emotion_distribution:
        trend_icon = {"increasing": "\u2191", "decreasing": "\u2193",
                      "stable": "\u2192"}.get(es.trend, "?")
        print(f"  {es.valence.name:15} {es.count:3}"
              f" ({es.percentage:5.1f}%) {trend_icon} {es.trend}")
    print(f"\nEmotional Balance: {report.emotional_balance:+.2f} (-1 negative, +1 positive)")

    # Temporal
    print("\n--- Activity ---")
    print(f"  Peak Period: {report.temporal_stats.peak_period}")
    print(f"  Activity Trend: {report.temporal_stats.activity_trend}")

    # Quality
    print("\n--- Quality Metrics ---")
    print(f"  Avg Episode Importance: {report.avg_episode_importance:.2f}")
    print(f"  Avg Fact Confidence: {report.avg_fact_confidence:.2f}")

    print("\n  Fact Types:")
    for ft, count in report.facts_by_type.items():
        print(f"    {ft}: {count}")

    # Most Important
    if report.most_important:
        print("\n--- Most Important Memories ---")
        for mem in report.most_important[:5]:
            print(f"  [{mem['importance']:.2f}] {mem['topic']} ({mem['valence']})")

    # Most Accessed
    if report.most_accessed:
        print("\n--- Most Accessed Memories ---")
        for mem in report.most_accessed[:5]:
            print(f"  [{mem['access_count']} hits] {mem['topic']}")

    return 0


def register(subparsers):
    """Register analysis commands with argparse."""
    # Stats
    stats_p = subparsers.add_parser("stats", help="Show database statistics")
    stats_p.add_argument("--db", default="anamnesis.db", help="Database path")

    # Access
    access_p = subparsers.add_parser("access", help="Show memory access statistics")
    access_p.add_argument("--db", default="anamnesis.db", help="Database path")

    # Emotions
    emotions_p = subparsers.add_parser("emotions", help="Show emotional valence statistics")
    emotions_p.add_argument("--db", default="anamnesis.db", help="Database path")
    emotions_p.add_argument("--emotion", help="Show episodes of specific emotion")
    emotions_p.add_argument("--limit", type=int, default=10, help="Max episodes to show")

    # Analytics
    analytics_p = subparsers.add_parser("analytics", help="Generate memory analytics report")
    analytics_p.add_argument("--db", default="anamnesis.db", help="Database path")
    analytics_p.add_argument("--quick", "-q", action="store_true", help="Quick stats only")

    return {
        "stats": cmd_stats,
        "access": cmd_access,
        "emotions": cmd_emotions,
        "analytics": cmd_analytics,
    }
