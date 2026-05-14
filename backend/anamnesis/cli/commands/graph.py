"""Graph commands: graph, contradictions."""

from pathlib import Path

from ...graph import GraphExporter
from ..base import make_context, print_header


def cmd_graph(args):
    """Export memory graph in various formats."""
    print_header("Graph Export")

    ctx = make_context(args)

    exporter = GraphExporter(ctx.episodic, ctx.semantic)

    print("Building memory graph...")
    graph = exporter.build_graph(
        include_facts=not args.no_facts,
        include_topics=not args.no_topics,
    )

    stats = exporter.get_stats(graph)
    print("\nGraph Statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Total edges: {stats['total_edges']}")
    print(f"  Nodes by type: {stats['nodes_by_type']}")
    print(f"  Edges by type: {stats['edges_by_type']}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "json":
        content = exporter.to_json(graph)
    elif args.format == "graphml":
        content = exporter.to_graphml(graph)
    elif args.format == "dot":
        content = exporter.to_dot(graph)
    else:
        print(f"Unknown format: {args.format}")
        return 1

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\nExported to: {output_path}")
    print(f"Format: {args.format}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    return 0


def cmd_contradictions(args):
    """Find and show contradictions in facts."""
    print_header("Contradiction Detection")

    ctx = make_context(args)
    facts = ctx.semantic.list_all(limit=1000)
    print(f"Analyzing {len(facts)} facts for contradictions...")

    from ...consolidation.contradiction import ContradictionDetector
    detector = ContradictionDetector()

    contradictions_found = []
    checked_pairs = set()

    for fact in facts:
        other_facts = [f for f in facts if f.id != fact.id]
        results = detector.find_contradictions(fact, other_facts, threshold=args.threshold)

        for result in results:
            if result.existing_fact:
                pair = tuple(sorted([fact.id, result.existing_fact.id]))
                if pair not in checked_pairs:
                    checked_pairs.add(pair)
                    contradictions_found.append((fact, result))

    if not contradictions_found:
        print("\nNo contradictions found.")
        return 0

    print(f"\nFound {len(contradictions_found)} potential contradictions:\n")

    for i, (fact, result) in enumerate(contradictions_found[:20], 1):
        print(f"{i}. [{result.contradiction_type}] (confidence: {result.confidence:.0%})")
        print(f"   NEW:      {fact.content[:60]}")
        print(f"   EXISTING: {result.existing_fact.content[:60]}")
        print(f"   Reason:   {result.explanation}")
        print()

    if len(contradictions_found) > 20:
        print(f"... and {len(contradictions_found) - 20} more")

    if args.resolve:
        print("\nTo resolve contradictions, you can:")
        print("  - Update facts manually")
        print("  - Use 'anamnesis facts' to review")
        print("  - Use the API to supersede old facts")

    return 0


def register(subparsers):
    """Register graph commands with argparse."""
    # Graph
    graph_p = subparsers.add_parser("graph", help="Export memory graph for visualization")
    graph_p.add_argument("--db", default="anamnesis.db", help="Database path")
    graph_p.add_argument("--output", "-o", default="memory_graph.json",
                         help="Output file path")
    graph_p.add_argument("--format", "-f", choices=["json", "graphml", "dot"],
                         default="json", help="Output format")
    graph_p.add_argument("--no-facts", action="store_true", help="Exclude fact nodes")
    graph_p.add_argument("--no-topics", action="store_true", help="Exclude topic clustering")

    # Contradictions
    contra_p = subparsers.add_parser("contradictions", help="Find contradictions in facts")
    contra_p.add_argument("--db", default="anamnesis.db", help="Database path")
    contra_p.add_argument("--threshold", type=float, default=0.5,
                          help="Confidence threshold (0.0-1.0)")
    contra_p.add_argument("--resolve", action="store_true", help="Show resolution tips")

    return {
        "graph": cmd_graph,
        "contradictions": cmd_contradictions,
    }
