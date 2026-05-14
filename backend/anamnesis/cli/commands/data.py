"""Data commands: export, backup, restore, archive."""

import json
from datetime import datetime
from pathlib import Path

from ...archive import ArchiveManager, ArchiveRule
from ...models import EmotionalValence, Episode, Fact, FactType
from ..base import make_context, print_header


def cmd_export(args):
    """Export memories to JSON file."""
    print_header("Export Memories")

    ctx = make_context(args)

    export_data = {
        "version": "1.0",
        "exported_at": datetime.now().isoformat(),
        "source_db": str(args.db),
    }

    if args.type in ('all', 'episodes'):
        episodes = ctx.episodic.list_all(limit=10000)
        export_data["episodes"] = []
        for ep in episodes:
            export_data["episodes"].append({
                "id": ep.id,
                "summary": ep.summary,
                "topic": ep.topic,
                "started_at": ep.started_at.isoformat() if ep.started_at else None,
                "ended_at": ep.ended_at.isoformat() if ep.ended_at else None,
                "overall_valence": (ep.overall_valence.value
                                    if hasattr(ep.overall_valence, 'value')
                                    else ep.overall_valence),
                "importance": ep.importance,
                "source_id": ep.source_id,
                "messages": ep.messages,
                "metadata": ep.metadata,
            })
        print(f"  Episodes: {len(export_data['episodes'])}")

    if args.type in ('all', 'facts'):
        facts = ctx.semantic.list_all(limit=10000)
        export_data["facts"] = []
        for f in facts:
            export_data["facts"].append({
                "id": f.id,
                "content": f.content,
                "fact_type": f.fact_type,
                "confidence": f.confidence,
                "importance": f.importance,
                "source_episodes": f.source_episodes,
                "first_learned": f.first_learned.isoformat() if f.first_learned else None,
                "last_confirmed": f.last_confirmed.isoformat() if f.last_confirmed else None,
                "confirmation_count": f.confirmation_count,
                "is_superseded": f.is_superseded,
                "metadata": f.metadata,
            })
        print(f"  Facts: {len(export_data['facts'])}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"\nExported to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    return 0


def cmd_backup(args):
    """Create a full database backup."""
    import shutil

    print_header("Backup Database")

    source = Path(args.db)
    if not source.exists():
        print(f"Database not found: {source}")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{source.stem}_backup_{timestamp}{source.suffix}"
    backup_path = Path(args.output) if args.output else source.parent / backup_name

    shutil.copy2(source, backup_path)

    print(f"Source: {source}")
    print(f"Backup: {backup_path}")
    print(f"Size: {backup_path.stat().st_size / 1024:.1f} KB")

    vectors_dir = source.parent / "vectors"
    if vectors_dir.exists():
        vectors_backup = backup_path.parent / f"vectors_backup_{timestamp}"
        shutil.copytree(vectors_dir, vectors_backup)
        print(f"Vectors: {vectors_backup}")

    return 0


def cmd_restore(args):
    """Restore from a backup or import JSON export."""
    print_header("Restore/Import Memories")

    source = Path(args.source)
    if not source.exists():
        print(f"Source not found: {source}")
        return 1

    if source.suffix == '.json':
        return _restore_from_json(args, source)
    else:
        return _restore_from_backup(args, source)


def _restore_from_json(args, source: Path):
    """Import from JSON export."""
    print(f"Importing from JSON: {source}")

    with open(source, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ctx = make_context(args)

    episodes_imported = 0
    facts_imported = 0

    if "episodes" in data:
        for ep_data in data["episodes"]:
            if ctx.episodic.get(ep_data["id"]):
                continue

            ep = Episode(
                id=ep_data["id"],
                summary=ep_data.get("summary", ""),
                topic=ep_data.get("topic"),
                started_at=(datetime.fromisoformat(ep_data["started_at"])
                            if ep_data.get("started_at") else None),
                ended_at=(datetime.fromisoformat(ep_data["ended_at"])
                          if ep_data.get("ended_at") else None),
                overall_valence=EmotionalValence(ep_data.get("overall_valence", 0)),
                importance=ep_data.get("importance", 0.5),
                source_id=ep_data.get("source_id"),
                messages=ep_data.get("messages", []),
                metadata=ep_data.get("metadata", {}),
            )
            ctx.episodic.save(ep)
            episodes_imported += 1

    if "facts" in data:
        for f_data in data["facts"]:
            if ctx.semantic.get(f_data["id"]):
                continue

            fact = Fact(
                id=f_data["id"],
                content=f_data["content"],
                fact_type=f_data.get("fact_type", FactType.GENERAL),
                confidence=f_data.get("confidence", 0.8),
                importance=f_data.get("importance", 0.5),
                source_episodes=f_data.get("source_episodes", []),
                first_learned=(datetime.fromisoformat(f_data["first_learned"])
                               if f_data.get("first_learned") else None),
                last_confirmed=(datetime.fromisoformat(f_data["last_confirmed"])
                                if f_data.get("last_confirmed") else None),
                confirmation_count=f_data.get("confirmation_count", 1),
                is_superseded=f_data.get("is_superseded", False),
                metadata=f_data.get("metadata", {}),
            )
            ctx.semantic.save(fact)
            facts_imported += 1

    print("\nImported:")
    print(f"  Episodes: {episodes_imported}")
    print(f"  Facts: {facts_imported}")

    return 0


def _restore_from_backup(args, source: Path):
    """Restore from SQLite backup."""
    import shutil

    dest = Path(args.db)
    print(f"Restoring from backup: {source}")
    print(f"Destination: {dest}")

    if dest.exists() and not args.force:
        print("\nDestination exists! Use --force to overwrite.")
        return 1

    shutil.copy2(source, dest)
    print("\nRestored successfully.")

    return 0


def cmd_archive(args):
    """Archive or manage archived memories."""
    print_header("Memory Archive")

    ctx = make_context(args)

    archive_path = str(Path(args.db).parent / "anamnesis_archive.db")
    manager = ArchiveManager(ctx.episodic, ctx.semantic, archive_path)

    if args.auto:
        rule = ArchiveRule(
            name="auto",
            min_age_days=args.age_days,
            max_importance=args.max_importance,
            max_access_count=args.max_access,
        )

        print("Auto-archive with rule:")
        print(f"  Min age: {rule.min_age_days} days")
        print(f"  Max importance: {rule.max_importance}")
        print(f"  Max access count: {rule.max_access_count}")

        results = manager.auto_archive(rule=rule, dry_run=args.dry_run)

        if args.dry_run:
            print(f"\n[DRY RUN] Would archive {len(results['candidates'])} memories:")
            for candidate in results['candidates'][:10]:
                print(f"  [{candidate['type']}]"
                      f" {candidate.get('topic') or candidate.get('content', '')[:30]}")
            if len(results['candidates']) > 10:
                print(f"  ... and {len(results['candidates']) - 10} more")
        else:
            print(f"\nArchived {results['episodes_archived']} episodes")
            print(f"Archived {results['facts_archived']} facts")

    elif args.restore:
        entry = manager.get_archived(args.restore)
        if not entry:
            print(f"Archive entry not found: {args.restore}")
            return 1

        if entry.entry_type == "episode":
            restored = manager.restore_episode(args.restore)
        else:
            restored = manager.restore_fact(args.restore)

        if restored:
            print(f"Restored: {restored.id}")
        else:
            print("Restore failed")
            return 1

    elif args.list:
        entries = manager.list_archived(limit=args.limit)

        if not entries:
            print("No archived entries found.")
            return 0

        print(f"Archived entries ({len(entries)}):\n")
        for entry in entries:
            date = entry.archived_at.strftime('%Y-%m-%d') if entry.archived_at else 'N/A'
            print(f"  [{entry.entry_type}] {entry.id}")
            print(f"    Topic: {entry.topic or entry.summary[:40]}")
            print(f"    Archived: {date} ({entry.archive_reason})")
            print()

    else:
        stats = manager.get_stats()

        print("Archive Statistics:")
        print(f"  Total archived: {stats.total_archived}")
        print(f"  Episodes: {stats.episodes_archived}")
        print(f"  Facts: {stats.facts_archived}")
        print(f"  Archive size: {stats.total_size_kb:.1f} KB")
        print(f"  Est. space saved: {stats.space_saved_estimate_kb:.1f} KB")

        if stats.oldest_entry:
            print(f"  Oldest: {stats.oldest_entry.strftime('%Y-%m-%d')}")
        if stats.newest_entry:
            print(f"  Newest: {stats.newest_entry.strftime('%Y-%m-%d')}")

    return 0


def register(subparsers):
    """Register data commands with argparse."""
    # Export
    export_p = subparsers.add_parser("export", help="Export memories to JSON")
    export_p.add_argument("--db", default="anamnesis.db", help="Database path")
    export_p.add_argument("--output", "-o", default="anamnesis_export.json",
                          help="Output file path")
    export_p.add_argument("--type", choices=["all", "episodes", "facts"], default="all",
                          help="What to export")

    # Backup
    backup_p = subparsers.add_parser("backup", help="Create database backup")
    backup_p.add_argument("--db", default="anamnesis.db", help="Database path")
    backup_p.add_argument("--output", "-o", help="Backup file path (default: auto-generated)")

    # Restore
    restore_p = subparsers.add_parser("restore", help="Restore from backup or JSON export")
    restore_p.add_argument("source", help="Backup file or JSON export to restore from")
    restore_p.add_argument("--db", default="anamnesis.db", help="Target database path")
    restore_p.add_argument("--force", "-f", action="store_true",
                           help="Overwrite existing database")

    # Archive
    archive_p = subparsers.add_parser("archive", help="Manage memory archive")
    archive_p.add_argument("--db", default="anamnesis.db", help="Database path")
    archive_p.add_argument("--auto", "-a", action="store_true",
                           help="Auto-archive based on rules")
    archive_p.add_argument("--restore", "-r", help="Restore archived entry by ID")
    archive_p.add_argument("--list", "-l", action="store_true", help="List archived entries")
    archive_p.add_argument("--limit", type=int, default=20, help="Max entries to list")
    archive_p.add_argument("--dry-run", action="store_true",
                           help="Show what would be archived")
    archive_p.add_argument("--age-days", type=int, default=90,
                           help="Min age for auto-archive")
    archive_p.add_argument("--max-importance", type=float, default=0.5,
                           help="Max importance for auto-archive")
    archive_p.add_argument("--max-access", type=int, default=2,
                           help="Max access count for auto-archive")

    return {
        "export": cmd_export,
        "backup": cmd_backup,
        "restore": cmd_restore,
        "archive": cmd_archive,
    }
