"""
Anamnesis CLI - Command-line interface for the memory system.

Usage:
    anamnesis import <source> [--type TYPE] [--db PATH]
    anamnesis consolidate [--db PATH] [--limit N]
    anamnesis stats [--db PATH]
    anamnesis search <query> [--db PATH] [--limit N]
    anamnesis facts [--db PATH] [--type TYPE]
    anamnesis inject <query> [--db PATH] [--format FORMAT] [--tokens N]
    anamnesis forget [--db PATH] [--dry-run]
    anamnesis demo [--db PATH]
"""

import argparse
import sys

from .commands import (
    analysis,
    consolidate,
    data,
    graph,
    import_cmd,
    interactive,
    maintenance,
    metadata,
    query,
    structure,
)

# All command modules that register subparsers
COMMAND_MODULES = [
    import_cmd,
    consolidate,
    query,
    maintenance,
    analysis,
    structure,
    data,
    metadata,
    interactive,
    graph,
]


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Anamnesis - Memory System for AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Collect command handlers from all modules
    commands = {}
    for module in COMMAND_MODULES:
        commands.update(module.register(subparsers))

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
