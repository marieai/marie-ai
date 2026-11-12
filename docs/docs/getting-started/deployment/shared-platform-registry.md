---
sidebar_position: 12
---

# Share organization, tenant and repository records

Marie-AI and Studio use one platform registry in the existing PostgreSQL database:

| Schema | Records |
| --- | --- |
| `marie_runtime` | Organization, tenant hierarchy, registered Git repositories, fabric/gateway definitions and storage connections. |
| `marie_scheduler` | Jobs, DAGs, execution leases, KB catalog and current indexed output. |
| `marie_studio` | Studio application records, users/memberships, workspace bindings and UI state. |

Organization and tenant IDs remain the same in both systems. Studio manages these
records through its existing UI and Prisma delegates, mapped to `marie_runtime`.
The backend SQL bundle owns their physical definitions. Shared table/type models
are externally managed in Prisma, so future Studio diffs do not recreate them.

These are schemas within one database. Moving tables does not replicate identities
between different databases. Consumers must use the same control-plane registry.
Existing workspace checkouts, executor path mounts and fabric bindings continue to
resolve execution in each environment.

## Existing installations

The transfer file is `tools/runtime/shared-platform-handoff.sql` in Marie-AI.
It moves the three tables and their shared enum types in place, preserving IDs,
table/type identities, foreign keys, organization settings and tenant hierarchy.
Repository visibility changes to the existing runtime visibility type; Studio's
visibility type remains available for its other application records.

1. Stop Studio API and the gateway, and allow active repository/datasource writers
   to finish. Preserve the database using the normal backup procedure.
2. Check Studio migration history before changing enum ownership. The prepared
   `20261109_code_repository_type` migration targets the legacy Studio enum. If it
   is pending, either apply it through the existing migration process before the
   transfer, or reconcile that single migration afterward as described below.
   Do not edit already applied migration files or blindly replay old Studio SQL
   after the shared enums have moved.
3. Run the entire transfer file against the same database used by both services.
   For an existing configured `psql` connection, from the Marie-AI checkout:

   ```bash
   psql -v ON_ERROR_STOP=1 -f tools/runtime/shared-platform-handoff.sql
   ```

4. Verify that each shared table now exists only in `marie_runtime`:

   ```sql
   SELECT n.nspname AS schema_name, c.relname AS table_name, c.oid
   FROM pg_class c
   JOIN pg_namespace n ON n.oid = c.relnamespace
   WHERE n.nspname IN ('marie_studio', 'marie_runtime')
     AND c.relname IN ('organization', 'tenant', 'git_repositories')
     AND c.relkind = 'r'
   ORDER BY c.relname, n.nspname;
   ```

   Expect exactly three runtime rows. The OIDs must match the pre-transfer table
   OIDs. Incoming Studio foreign keys and dependent views follow the moved objects.
5. Deploy the complete matching backend SQL bundle, including
   `102_shared_platform_registry.sql` and `103_kb_code_navigation.sql`, to the SQL
   directory the gateway actually uses. Normal scheduler startup installs/replays
   it. The current schema revision is 103. An unresolved legacy catalog causes
   bootstrap to fail before creating a second set of shared tables.
6. Regenerate and build Studio's database package before starting Studio API:

   ```bash
   pnpm --filter @marie/db exec prisma generate
   pnpm --filter @marie/db build
   ```

7. Restart the matching gateway and Studio API. Extraction and KB executors only
   need reloading if their deployed source includes changed registry readers.

The transfer uses one transaction and rejects partial catalogs, populated runtime
duplicates and unexpected private dependencies. It can replace known empty runtime
bootstrap tables with the existing legacy objects; unknown dependencies cause
rollback rather than cascading deletion. If it fails, resolve the stated conflict
before retrying. Do not delete populated catalogs to get past the guard.

This operation moves only organization, tenant and repository registration. It does
not move KB catalog data. If a legacy Studio KB catalog also needs transfer, run
the separate [KB ownership handoff](scheduler-database-upgrades.md) after this
shared-platform handoff; that procedure now reads tenant authority from runtime.

## Pending code-repository enum migration

The transfer adds the `code` value to the same repository asset enum while retaining
its identity. If `20261109_code_repository_type` was not already recorded as applied,
verify that its complete effect is present:

```sql
SELECT 'code'::marie_runtime.repository_asset_type;
```

After verifying this and the transfer, an operator using Prisma migration history
can record the equivalent completed change from Studio's database package:

```bash
pnpm --filter @marie/db exec prisma migrate resolve --applied 20261109_code_repository_type
```

This updates migration bookkeeping; it does not run the old statement against the
removed Studio enum. Do this only for the exact pending migration whose full
effect has been verified. Other historical migration differences require their
own reconciliation. The transfer itself does not alter Prisma's migration ledger.

## Fresh installations and validation

For standalone Marie-AI, numbered SQL installs runtime registry 093 followed by
the final shared layout 102 without requiring Studio tables. If provisioning Studio
from its historical migration chain, finish that chain and the shared-platform
handoff before starting either application's writers. Current Studio models then
use the shared objects; they do not create another identity catalog.

Disposable PostgreSQL checks cover repeat installation, transfer/replay, foreign
keys, hierarchy, enum identity, duplicate rejection and rollback. An isolated
generated Prisma client checks both directions of shared reads/writes. Tenant
rename/reparent regressions exercise the actual router against shared tables and
verify that a descendant failure rolls back the parent update.
