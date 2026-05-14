"""Import commands: import, batch-import."""

import json
import sqlite3
from pathlib import Path

from ...anamnesis import Anamnesis
from ..base import print_header


def cmd_import(args):
    """Import data from a source."""
    print_header("Import Data")

    ana = Anamnesis(db_path=args.db)

    if args.type == "claude":
        print(f"Importing Claude export from: {args.source}")
        result = ana.import_claude_export(args.source)
        print(f"  Imported {result['conversations']} conversations")
        print(f"  Imported {result['facts']} facts from Claude memories")
    else:
        print(f"Unknown source type: {args.type}")
        return 1

    stats = ana.get_stats()
    print("\nDatabase now contains:")
    print(f"  Episodes: {stats['episodes']}")
    print(f"  Facts: {stats['facts']}")
    print(f"  Raw conversations: {stats['raw_conversations']}")
    return 0


def cmd_batch_import(args):
    """Batch import with progress and resume support."""
    print_header("Batch Import")

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Source not found: {source_path}")
        return 1

    # Track progress file
    progress_file = Path(args.db).parent / f".import_progress_{source_path.stem}.json"
    imported_ids = set()

    # Resume from previous run if available
    if progress_file.exists() and args.resume:
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
            imported_ids = set(progress_data.get("imported_ids", []))
        print(f"Resuming from previous import ({len(imported_ids)} already imported)")

    ana = Anamnesis(db_path=args.db)

    if args.type == "claude":
        from ...loaders.claude_loader import ClaudeExportLoader
        loader = ClaudeExportLoader(str(source_path))

        # Load all conversations
        print("Loading conversations from export...")
        conversations = list(loader.load_conversations())
        print(f"Found {len(conversations)} conversations")

        # Filter out already imported
        to_import = [c for c in conversations if c.id not in imported_ids]
        print(f"To import: {len(to_import)}")

        if not to_import:
            print("All conversations already imported.")
            return 0

        # Import in batches with progress
        batch_size = args.batch_size
        imported_count = 0
        error_count = 0

        for i in range(0, len(to_import), batch_size):
            batch = to_import[i:i + batch_size]

            for convo in batch:
                try:
                    ana._import_conversation(convo)
                    imported_ids.add(convo.id)
                    imported_count += 1
                except (ValueError, KeyError, OSError, sqlite3.Error) as e:
                    if args.verbose:
                        print(f"  Error importing {convo.id}: {e}")
                    error_count += 1

            # Save progress
            with open(progress_file, 'w') as f:
                json.dump({"imported_ids": list(imported_ids)}, f)

            # Progress indicator
            total_done = i + len(batch)
            pct = (total_done / len(to_import)) * 100
            print(f"  Progress: {total_done}/{len(to_import)} ({pct:.1f}%)")

        # Import memories/facts if available
        facts = list(loader.load_memories())
        fact_count = 0
        for fact in facts:
            try:
                ana.semantic.save(fact)
                fact_count += 1
            except (ValueError, KeyError, sqlite3.Error):
                pass

        # Cleanup progress file
        if progress_file.exists():
            progress_file.unlink()

        print("\nImport complete:")
        print(f"  Conversations imported: {imported_count}")
        print(f"  Facts imported: {fact_count}")
        print(f"  Errors: {error_count}")

        if not args.skip_consolidate:
            print("\nTip: Run 'consolidate' to process imported conversations")

    return 0


def register(subparsers):
    """Register import commands with argparse."""
    import_p = subparsers.add_parser("import", help="Import data from a source")
    import_p.add_argument("source", help="Path to data source")
    import_p.add_argument("--type", default="claude", choices=["claude"], help="Source type")
    import_p.add_argument("--db", default="anamnesis.db", help="Database path")

    batch_import_p = subparsers.add_parser("batch-import", help="Batch import with resume support")
    batch_import_p.add_argument("source", help="Path to data source")
    batch_import_p.add_argument("--type", default="claude", choices=["claude"], help="Source type")
    batch_import_p.add_argument("--db", default="anamnesis.db", help="Database path")
    batch_import_p.add_argument("--batch-size", type=int, default=50, help="Batch size")
    batch_import_p.add_argument("--resume", action="store_true", help="Resume from previous import")
    batch_import_p.add_argument("--skip-consolidate", action="store_true", help="Skip consolidation hint")
    batch_import_p.add_argument("--verbose", "-v", action="store_true", help="Show detailed progress")

    return {
        "import": cmd_import,
        "batch-import": cmd_batch_import,
    }
