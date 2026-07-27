# Marie-AI SQL Schema

This directory contains the SQL schema definitions for the Marie-AI scheduler database.

## Schema Overview

All tables are created in the `marie_scheduler` schema (configurable via `{schema}` placeholder).

### File Naming Convention

Files use 3-digit numbered prefixes for explicit load ordering:
- `001_schema.sql` - Schema creation
- `002_enums.sql` - Enum types
- `003_version.sql` - Version tracking
- `005_job.sql` - Main job table
- etc.

Files are auto-discovered and loaded in lexical order by
`AsyncJobRepository.create_tables()`: numbered files in this directory first,
then files in `lease/`. The ordered SQL files are the deployment contract; do
not add separate ordering metadata that can drift from the artifacts.

### Directory Structure

```
schema/
├── 001_schema.sql ... 043_*.sql   # Core schema files (auto-loaded)
├── lease/                          # Lease management functions
│   ├── 001_lease_jobs_by_id.sql
│   └── ...
├── dev/                            # Development/test files (NOT auto-loaded)
│   ├── cleanup.sql
│   └── ...
└── monitoring/                     # Monitoring views (NOT auto-loaded)
    └── ...
```

## Shared Tables (Source of Truth)

These tables are defined here and also used by Marie-Studio:

| Table | File | Description |
|-------|------|-------------|
| `job` | 005_job.sql | Main job queue (partitioned) |
| `dag` | 007_dag.sql | DAG workflow definitions |
| `queue` | 004_queue.sql | Queue configuration |
| `job_dependencies` | 017_job_dependencies.sql | Job dependency tracking |
| `job_history` | 006_job_history.sql | Job state change history |
| `dag_history` | 008_dag_history.sql | DAG state change history |
| `schedule` | 009_schedule.sql | Cron-based scheduling |
| `subscription` | 010_subscription.sql | Event subscriptions |
| `archive` | 011_archive.sql | Archived jobs |
| `llm_queue_fabric_config` | 066_llm_queue_scheduler.sql | Runtime Fabric scoped LLM dispatch scheduler policy |
| `llm_queue_pool` | 066_llm_queue_scheduler.sql | Configured LLM dispatch pools used as DRR lanes |
| `version` | 003_version.sql | Schema version tracking |

## When Making Changes

### Adding New Columns

1. Create a new numbered file (e.g., `044_add_new_column.sql`)
2. Use idempotent pattern:
   ```sql
   DO $$ BEGIN
       ALTER TABLE {schema}.table_name ADD COLUMN new_column TYPE;
   EXCEPTION
       WHEN duplicate_column THEN NULL;
   END $$;
   ```
3. Build and deploy a Marie-AI image containing `config/psql`
4. Notify Marie-Studio team to update Prisma schema

### Adding New Tables

1. Create a new numbered file
2. Use `CREATE TABLE IF NOT EXISTS {schema}.table_name`
3. Add appropriate indexes and comments

### Modifying Existing Tables

1. **Never** modify existing migration files once deployed
2. Create a new migration file with `ALTER TABLE` statements
3. Use idempotent patterns (IF NOT EXISTS, EXCEPTION handling)

## Sync with Marie-Studio

Marie-Studio uses Prisma ORM and maintains its own schema definitions. When this schema changes:

1. Marie-AI changes are deployed first
2. Marie-Studio runs `prisma db pull` to introspect changes
3. Or manually updates Prisma models to match

### Schema Separation

- **marie_scheduler**: Shared tables (owned by Marie-AI)
- **marie_studio**: Studio-only tables (owned by Marie-Studio)

Marie-AI is not aware of `marie_studio` tables. This allows Marie-AI to run independently.

## Deployment Ownership

`config/psql` is the canonical SQL source. Helm charts must not carry
copied schema SQL; Kubernetes deployments should point `MARIE_PSQL_DIR`
and `MARIE_SCHEMA_DIR` at the packaged image paths, or at an explicitly
managed external runtime mount.

`config/psql/schema` is the only active scheduler schema tree. Do not create
parallel `schema-v*` or environment-specific schema directories.

## Canonical Function Definitions

Deployed SQL history is immutable. When a function must replace a definition
from an earlier artifact, add a new forward SQL file instead of editing the old
file. Schema version 74 gives each logical function its own final artifact:

- `lease/012_job_update_trigger_function.sql`
- `lease/013_resolve_dag_state.sql`
- `lease/014_release_expired_leases.sql`
- `lease/015_activate_from_lease.sql` for the overload family

Tests and reviews should use those files as the current contracts. The earlier
definitions remain only because existing deployments may already have applied
them.

### `activate_from_lease()` compatibility

| Signature | Status | Removal date |
|-----------|--------|--------------|
| `(uuid[], uuid[], text, interval, text)` returning `(job_id, run_attempt_id)` | Current regular-dispatch contract; introduced by build `4ed95423` | No removal planned |
| `(uuid[], text, interval, text)` returning `(job_id, run_attempt_id)` | Current control-flow contract and rolling-upgrade contract for builds containing `c4048865` or a descendant | No removal planned while control-flow activation uses it |
| `(uuid[], text, interval)` returning `uuid[]` | Legacy rolling-upgrade contract for builds older than `c4048865` | Fleet convergence date on builds containing `c4048865`, plus 30 days |

The package version remained `5.0.0` across these changes, so it is not a safe
retirement signal. Inventory every running gateway's `git-commit` value from
its `/status` response. Record the last-old-gateway removal date during the
rollout; the three-argument overload may be removed by a later forward SQL file
only after the resulting 30-day rollback window has expired. If any gateway
reports `unknown`, retain the overload. An exact calendar date is intentionally
unset until the fleet inventory supplies the convergence date.

## Testing Changes

```bash
# Drop and recreate schema (development only!)
psql -c "DROP SCHEMA marie_scheduler CASCADE; CREATE SCHEMA marie_scheduler;"

# Run Marie-AI to apply schema
marie server --start --uses config/service/marie.yml
```

## Placeholder Substitution

All SQL files use `{schema}` placeholder which is replaced at runtime with the actual schema name (default: `marie_scheduler`).

```sql
-- In file:
CREATE TABLE {schema}.job (...)

-- At runtime (after substitution):
CREATE TABLE marie_scheduler.job (...)
```
