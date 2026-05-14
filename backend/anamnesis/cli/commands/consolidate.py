"""Consolidation command."""

import json
from datetime import datetime

from ...consolidation.pipeline import ConsolidationPipeline
from ...models import Conversation, ConversationMessage
from ..base import make_context, print_header


def cmd_consolidate(args):
    """Run consolidation pipeline on raw conversations."""
    print_header("Consolidation Pipeline")

    ctx = make_context(args)

    initial_episodes = ctx.episodic.count()
    initial_facts = ctx.semantic.count()

    # Get total count of unprocessed
    total_unprocessed = len(ctx.backend.get_unprocessed_conversations(limit=100000))
    print(f"Total unprocessed conversations: {total_unprocessed}")

    if total_unprocessed == 0:
        print("Nothing to consolidate.")
        return 0

    # Process in batches
    batch_size = args.batch_size
    limit = args.limit if args.limit > 0 else total_unprocessed

    pipeline = ConsolidationPipeline(
        episodic_store=ctx.episodic,
        semantic_store=ctx.semantic,
    )

    total_processed = 0
    total_success = 0
    batch_num = 0

    while total_processed < limit:
        batch_num += 1
        unprocessed = ctx.backend.get_unprocessed_conversations(
            limit=min(batch_size, limit - total_processed)
        )

        if not unprocessed:
            break

        if args.verbose:
            print(f"\nBatch {batch_num}: Processing {len(unprocessed)} conversations...")

        for convo_dict in unprocessed:
            # Reconstruct Conversation object
            messages_raw = convo_dict.get('messages', [])
            if isinstance(messages_raw, str):
                messages_raw = json.loads(messages_raw)

            if not messages_raw:
                ctx.backend.mark_conversation_processed(convo_dict['id'])
                total_processed += 1
                continue

            messages = []
            for m in messages_raw:
                ts = None
                if m.get('timestamp'):
                    try:
                        ts = datetime.fromisoformat(m['timestamp'].replace('Z', ''))
                    except (ValueError, TypeError):
                        pass
                messages.append(ConversationMessage(
                    id=m.get('id', ''),
                    role=m.get('role', 'user'),
                    content=m.get('content', ''),
                    timestamp=ts,
                ))

            created_at = None
            if convo_dict.get('created_at'):
                try:
                    created_at = datetime.fromisoformat(convo_dict['created_at'])
                except (ValueError, TypeError):
                    pass

            convo = Conversation(
                id=convo_dict['id'],
                name=convo_dict.get('name'),
                summary=convo_dict.get('summary'),
                messages=messages,
                created_at=created_at,
            )

            result = pipeline.consolidate(convo)
            if result.success:
                total_success += 1

            ctx.backend.mark_conversation_processed(convo_dict['id'])
            total_processed += 1

            # Progress indicator
            if total_processed % 10 == 0:
                pct = (total_processed / min(limit, total_unprocessed)) * 100
                print(f"  Progress: {total_processed}/{min(limit, total_unprocessed)} ({pct:.1f}%)")

    final_episodes = ctx.episodic.count()
    final_facts = ctx.semantic.count()

    print("\nConsolidation complete:")
    print(f"  Conversations processed: {total_processed}")
    print(f"  Successfully consolidated: {total_success}")
    print(f"  New episodes: {final_episodes - initial_episodes}")
    print(f"  New facts: {final_facts - initial_facts}")
    return 0


def register(subparsers):
    """Register consolidation command with argparse."""
    cons_p = subparsers.add_parser("consolidate", help="Run consolidation pipeline")
    cons_p.add_argument("--db", default="anamnesis.db", help="Database path")
    cons_p.add_argument("--limit", type=int, default=0, help="Max conversations to process (0=all)")
    cons_p.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    cons_p.add_argument("--verbose", "-v", action="store_true", help="Show detailed progress")

    return {"consolidate": cmd_consolidate}
