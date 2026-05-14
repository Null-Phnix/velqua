"""Query commands: search, facts, inject, semantic, reindex."""

from pathlib import Path

from ...anamnesis import Anamnesis
from ...integration import FormatStyle, InjectionMode, MemoryInjector, get_preset
from ...retrieval import SemanticSearchService
from ..base import make_context, parse_emotion, print_header, valence_to_str


def cmd_search(args):
    """Search memories."""
    # Handle emotion-only search
    if args.emotion and not args.query:
        return cmd_search_by_emotion(args)

    print_header(f"Search: '{args.query}'")

    ana = Anamnesis(db_path=args.db)
    results = ana.remember(args.query, limit=args.limit)

    episodes = results.get('episodes', [])
    facts = results.get('facts', [])

    # Filter by emotion if specified
    if args.emotion:
        valence = parse_emotion(args.emotion)
        if valence is not None:
            episodes = [ep for ep in episodes if ep.overall_valence == valence]

    if episodes:
        print(f"Episodes ({len(episodes)}):")
        for ep in episodes:
            date = ep.started_at.strftime('%Y-%m-%d') if ep.started_at else 'N/A'
            topic = ep.topic or 'Untitled'
            valence_str = valence_to_str(ep.overall_valence)
            print(f"  [{date}] [{valence_str}] {topic}")
            if ep.summary:
                print(f"    {ep.summary[:100]}...")
        print()

    if facts:
        print(f"Facts ({len(facts)}):")
        for fact in facts:
            print(f"  [{fact.fact_type}] {fact.content}")

    if not episodes and not facts:
        print("No results found.")

    return 0


def cmd_search_by_emotion(args):
    """Search by emotion only (no query)."""
    valence = parse_emotion(args.emotion)
    if valence is None:
        print(f"Unknown emotion: {args.emotion}")
        print("Valid options: very_positive, positive, neutral, negative, very_negative")
        return 1

    print_header(f"Episodes with emotion: {args.emotion}")

    ctx = make_context(args)
    episodes = ctx.episodic.get_emotional(valence, limit=args.limit)

    if episodes:
        for ep in episodes:
            date = ep.started_at.strftime('%Y-%m-%d') if ep.started_at else 'N/A'
            topic = ep.topic or 'Untitled'
            print(f"  [{date}] {topic}")
            if ep.summary:
                print(f"    {ep.summary[:100]}...")
    else:
        print(f"  No episodes found with {args.emotion} emotion.")

    return 0


def cmd_facts(args):
    """List facts."""
    print_header("Known Facts")

    ctx = make_context(args)

    if args.type:
        facts = ctx.semantic.get_by_type(args.type, limit=50)
        print(f"Facts of type '{args.type}':")
    else:
        facts = ctx.semantic.list_all(limit=50)
        print("All facts:")

    if not facts:
        print("  No facts found.")
        return 0

    # Group by type
    by_type = {}
    for f in facts:
        if f.fact_type not in by_type:
            by_type[f.fact_type] = []
        by_type[f.fact_type].append(f)

    for ftype, type_facts in sorted(by_type.items()):
        print(f"\n  {ftype.upper()} ({len(type_facts)}):")
        for f in type_facts[:10]:
            conf = f"[{f.confidence:.0%}]" if f.confidence < 1.0 else ""
            print(f"    - {f.content[:70]} {conf}")
        if len(type_facts) > 10:
            print(f"    ... and {len(type_facts) - 10} more")

    return 0


def cmd_inject(args):
    """Generate context injection for a query."""
    print_header(f"Context Injection: '{args.query}'")

    ctx = make_context(args)

    style_map = {
        'markdown': FormatStyle.MARKDOWN,
        'xml': FormatStyle.XML,
        'natural': FormatStyle.NATURAL,
        'bullet': FormatStyle.BULLET,
        'minimal': FormatStyle.MINIMAL,
    }
    style = style_map.get(args.format.lower(), FormatStyle.MARKDOWN)

    mode_map = {
        'system': InjectionMode.SYSTEM_PROMPT,
        'user': InjectionMode.USER_CONTEXT,
        'preface': InjectionMode.ASSISTANT_PREFACE,
    }
    mode = mode_map.get(args.mode.lower(), InjectionMode.SYSTEM_PROMPT)

    injector = MemoryInjector(
        episodic_store=ctx.episodic,
        semantic_store=ctx.semantic,
        budget=get_preset('standard') if args.tokens <= 1000 else get_preset('extensive'),
        format_style=style,
    )

    result = injector.inject_for_query(query=args.query, mode=mode)

    print(f"Mode: {result.mode.value}")
    print(f"Episodes used: {result.episodes_used}")
    print(f"Facts used: {result.facts_used}")
    print(f"Tokens: ~{result.total_tokens}")
    print(f"Budget used: {result.budget_used:.0%}")
    print()
    print("-" * 60)
    print(result.injected_text)
    print("-" * 60)

    return 0


def cmd_reindex(args):
    """Generate embeddings for vector search."""
    print_header("Reindex for Vector Search")

    ctx = make_context(args)

    episodes = ctx.episodic.list_all(limit=10000)
    facts = ctx.semantic.list_all(limit=10000)

    print(f"Found {len(episodes)} episodes and {len(facts)} facts")

    if len(episodes) == 0 and len(facts) == 0:
        print("Nothing to index.")
        return 0

    from ...retrieval.hybrid import HybridRetriever

    db_dir = Path(args.db).parent
    vector_dir = db_dir / "vectors"
    vector_dir.mkdir(exist_ok=True)

    retriever = HybridRetriever(sqlite_backend=ctx.backend)
    retriever._ensure_embedder()
    retriever._ensure_vector_store(str(vector_dir))

    if not retriever.embedder:
        print("ERROR: Could not load embedder. Install sentence-transformers:")
        print("  pip install sentence-transformers")
        return 1

    if not retriever.vector_store:
        print("ERROR: Could not initialize vector store.")
        return 1

    print(f"Using embedder: {retriever.embedder}")
    print(f"Vector store: {vector_dir}")
    print()

    def progress(current, total):
        if current % 10 == 0 or current == total:
            print(f"  Indexed {current}/{total}...")

    print("Indexing memories...")
    retriever.index_all(episodes, facts, batch_size=50, progress_callback=progress)

    stats = retriever.get_stats()
    print("\nIndexing complete:")
    print(f"  Vector search: {stats['vector_search']}")
    print(f"  Vectors indexed: {stats['vector_count']}")

    return 0


def cmd_semantic(args):
    """Semantic search using embeddings."""
    print_header("Semantic Search")

    ctx = make_context(args)

    vector_path = None
    if args.persist:
        db_dir = Path(args.db).parent
        vector_path = str(db_dir / "vectors")

    service = SemanticSearchService(
        episodic_store=ctx.episodic,
        semantic_store=ctx.semantic,
        vector_persist_path=vector_path,
        use_transformers=args.use_transformers,
        fts_weight=args.fts_weight,
        vector_weight=args.vector_weight,
    )

    if args.index:
        print("Indexing memories...")
        print(f"  Using embedder: {service.embedder.model_name}")
        print(f"  Embedding dimension: {service.embedder.dimension}")

        service.index_all(batch_size=args.batch_size)

        stats = service.get_stats()
        print("\nIndexing complete:")
        print(f"  Episodes indexed: {stats['episodes_indexed']}")
        print(f"  Facts indexed: {stats['facts_indexed']}")
        return 0

    if args.stats:
        stats = service.get_stats()

        print("Semantic Search Status:")
        print(f"  Embedder: {stats['embedder']}")
        print(f"  Dimension: {stats['embedding_dimension']}")
        print(f"  Episodes indexed: {stats['episodes_indexed']}")
        print(f"  Facts indexed: {stats['facts_indexed']}")
        print(f"  Fully indexed: {stats['indexed']}")

        if not stats['indexed'] and stats['episodes_indexed'] == 0:
            print("\nTip: Run 'anamnesis semantic --index' to enable vector search")

        return 0

    if not args.query:
        print("Usage: anamnesis semantic <query> [options]")
        print("\nOptions:")
        print("  --index         Index all memories for vector search")
        print("  --stats         Show indexing statistics")
        print("  --fts-only      Use only full-text search (no vectors)")
        return 0

    print(f"Query: {args.query}")
    print(f"Mode: {'FTS only' if args.fts_only else 'Hybrid (FTS + Vector)'}")
    print()

    use_hybrid = not args.fts_only

    if args.type in (None, "episode"):
        print(f"Episodes (top {args.limit}):")
        results = service.search_episodes(args.query, limit=args.limit, use_hybrid=use_hybrid)

        if not results:
            print("  No episodes found.")
        else:
            for i, (ep, score) in enumerate(results, 1):
                date = ep.started_at.strftime('%Y-%m-%d') if ep.started_at else 'N/A'
                topic = ep.topic or 'Untitled'
                print(f"  {i}. [{score:.2f}] [{date}] {topic}")
                if args.verbose and ep.summary:
                    print(f"       {ep.summary[:80]}...")
        print()

    if args.type in (None, "fact"):
        print(f"Facts (top {args.limit}):")
        results = service.search_facts(args.query, limit=args.limit, use_hybrid=use_hybrid)

        if not results:
            print("  No facts found.")
        else:
            for i, (fact, score) in enumerate(results, 1):
                print(f"  {i}. [{score:.2f}] [{fact.fact_type}] {fact.content[:60]}")

    return 0


def register(subparsers):
    """Register query commands with argparse."""
    # Search
    search_p = subparsers.add_parser("search", help="Search memories")
    search_p.add_argument("query", nargs="?", default="",
                          help="Search query (optional if --emotion provided)")
    search_p.add_argument("--db", default="anamnesis.db", help="Database path")
    search_p.add_argument("--limit", type=int, default=10, help="Max results")
    search_p.add_argument("--emotion",
                          help="Filter by emotion (positive, negative, neutral, very_positive, very_negative)")

    # Facts
    facts_p = subparsers.add_parser("facts", help="List known facts")
    facts_p.add_argument("--db", default="anamnesis.db", help="Database path")
    facts_p.add_argument("--type", help="Filter by fact type")

    # Inject
    inject_p = subparsers.add_parser("inject", help="Generate context injection")
    inject_p.add_argument("query", help="Query for context")
    inject_p.add_argument("--db", default="anamnesis.db", help="Database path")
    inject_p.add_argument("--format", default="markdown",
                          choices=["markdown", "xml", "natural", "bullet", "minimal"],
                          help="Output format")
    inject_p.add_argument("--mode", default="system",
                          choices=["system", "user", "preface"],
                          help="Injection mode")
    inject_p.add_argument("--tokens", type=int, default=1000, help="Token budget")

    # Reindex
    subparsers.add_parser("reindex", help="Generate embeddings for vector search") \
        .add_argument("--db", default="anamnesis.db", help="Database path")

    # Semantic
    semantic_p = subparsers.add_parser("semantic", help="Semantic search with embeddings")
    semantic_p.add_argument("query", nargs="?", help="Search query")
    semantic_p.add_argument("--db", default="anamnesis.db", help="Database path")
    semantic_p.add_argument("--index", action="store_true", help="Index all memories")
    semantic_p.add_argument("--stats", "-s", action="store_true", help="Show indexing stats")
    semantic_p.add_argument("--type", "-t", choices=["episode", "fact"],
                            help="Memory type to search")
    semantic_p.add_argument("--limit", type=int, default=5, help="Max results")
    semantic_p.add_argument("--fts-only", action="store_true", help="Use only FTS (no vectors)")
    semantic_p.add_argument("--fts-weight", type=float, default=0.3, help="FTS weight (0-1)")
    semantic_p.add_argument("--vector-weight", type=float, default=0.7, help="Vector weight (0-1)")
    semantic_p.add_argument("--persist", action="store_true", help="Persist vector store")
    semantic_p.add_argument("--use-transformers", action="store_true",
                            help="Use sentence-transformers")
    semantic_p.add_argument("--batch-size", type=int, default=50, help="Indexing batch size")
    semantic_p.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    return {
        "search": cmd_search,
        "facts": cmd_facts,
        "inject": cmd_inject,
        "reindex": cmd_reindex,
        "semantic": cmd_semantic,
    }
