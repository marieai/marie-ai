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

Files are auto-discovered and loaded in numeric order by `job_repository.py`.

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
3. Sync to runtime: `cp *.sql /mnt/data/marie-ai/config/psql/schema/`
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
