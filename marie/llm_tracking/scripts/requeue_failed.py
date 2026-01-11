#!/usr/bin/env python
"""
Requeue Failed LLM Tracking Events.

This script queries failed events from PostgreSQL and republishes them to
RabbitMQ for reprocessing by the worker.

Usage:
    python -m marie.llm_tracking.scripts.requeue_failed \\
        --config config/service/marie-gateway-4.0.0.yml \\
        --status failed \\
        --limit 100 \\
        --dry-run
"""

import argparse
import json
import logging
import sys
from typing import Optional

import pika
import yaml

from marie.llm_tracking.config import configure_from_yaml, get_settings
from marie.llm_tracking.storage.postgres import PostgresStorage

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def requeue_failed_events(
    config_path: str,
    status: str = "failed",
    limit: int = 100,
    dry_run: bool = False,
) -> int:
    """
    Query failed events from Postgres and republish to RabbitMQ.

    Args:
        config_path: Path to YAML config file
        status: Event status to query ('failed' or 'pending')
        limit: Max events to requeue
        dry_run: If True, only print what would be done

    Returns:
        Number of events requeued (or would be requeued in dry-run mode)
    """
    # Load config from YAML (required!)
    config = load_config(config_path)
    llm_tracking_config = config.get("llm_tracking", {})
    storage_config = config.get("storage", {})

    if not llm_tracking_config:
        raise ValueError(f"No llm_tracking config found in {config_path}")

    # Initialize settings from YAML
    configure_from_yaml(llm_tracking_config, storage_config)
    settings = get_settings()

    if not settings.POSTGRES_URL:
        raise ValueError("Postgres URL not configured in llm_tracking config")
    if not settings.RABBITMQ_URL:
        raise ValueError("RabbitMQ URL not configured in llm_tracking config")

    # Initialize Postgres
    postgres = PostgresStorage()
    postgres.start()

    # Query events by status
    conn = postgres._get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, trace_id, event_type FROM {postgres._table_name}
                WHERE status = %s
                ORDER BY created_at ASC
                LIMIT %s
                """,
                (status, limit),
            )
            events = cur.fetchall()
    finally:
        postgres._close_connection(conn)

    if not events:
        print(f"No events with status='{status}' found")
        postgres.stop()
        return 0

    print(f"Found {len(events)} events to requeue")

    if dry_run:
        for event_id, trace_id, event_type in events:
            print(
                f"  Would requeue: {event_id} (trace: {trace_id}, type: {event_type})"
            )
        postgres.stop()
        return len(events)

    # Connect to RabbitMQ and republish
    connection = pika.BlockingConnection(pika.URLParameters(settings.RABBITMQ_URL))
    channel = connection.channel()

    # Ensure exchange exists
    channel.exchange_declare(
        exchange=settings.RABBITMQ_EXCHANGE,
        exchange_type="topic",
        durable=True,
    )

    requeued_count = 0
    for event_id, trace_id, event_type in events:
        try:
            message = json.dumps(
                {
                    "event_id": str(event_id),
                    "trace_id": str(trace_id),
                    "event_type": event_type,
                }
            )
            channel.basic_publish(
                exchange=settings.RABBITMQ_EXCHANGE,
                routing_key=settings.RABBITMQ_ROUTING_KEY,
                body=message,
                properties=pika.BasicProperties(
                    delivery_mode=pika.DeliveryMode.Persistent,
                    headers={"x-retry-count": 0},  # Reset retry count
                ),
            )

            # Reset status to pending
            postgres.mark_pending(str(event_id))
            requeued_count += 1
            print(f"Requeued: {event_id}")
        except Exception as e:
            logger.error(f"Failed to requeue {event_id}: {e}")

    connection.close()
    postgres.stop()
    print(f"Successfully requeued {requeued_count} events")
    return requeued_count


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Requeue failed LLM tracking events from Postgres to RabbitMQ"
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to YAML config file (e.g., config/service/marie-gateway-4.0.0.yml)",
    )
    parser.add_argument(
        "--status",
        default="failed",
        choices=["failed", "pending"],
        help="Status of events to requeue (default: failed)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of events to requeue (default: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making changes",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        count = requeue_failed_events(
            config_path=args.config,
            status=args.status,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        sys.exit(0 if count >= 0 else 1)
    except Exception as e:
        logger.error(f"Failed to requeue events: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
