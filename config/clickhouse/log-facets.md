# Store log facet fields

Research uses stored `event_type`, `job_tag`, and `event_source` columns to avoid
reading the complete `LogAttributes` map for common filters. Counts still cover
the full selected date range and apply the same search conditions. Event Source
remains available in queries without becoming a default sidebar facet.

The Docker initialization SQL and Helm schema job add these columns. New logs
populate them automatically. Existing installations need a one-time backfill:

```bash
clickhouse-client --multiquery < config/clickhouse/materialize-log-facets.sql
```

Run from the Marie-AI repository with your ClickHouse client connection options.
The script waits for the local server to complete the backfill. It reads the
existing attribute maps to populate the three columns; schedule that work for
large installations. Replicated installations should run it on each shard and
verify replicas have completed the mutation.

Studio detects these columns and their materialized expressions. It retains the
map query path before the schema update and checks again after one minute.
Adding columns alone leaves older parts calculating values from maps until
the backfill or a merge writes them. See [ClickHouse column operations](https://clickhouse.com/docs/reference/statements/alter/column).
