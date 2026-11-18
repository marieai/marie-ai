---
sidebar_position: 13
---

# Connect repository providers and reuse their repositories

Settings → Integrations → Git Providers owns provider connections. Named GitHub,
GitLab and Gitea instances extend that existing registry; discovery and KB forms
select connection IDs from it. Connecting an external Gitea instance does not
replace the platform's System Git service.

## Upgrade an existing installation

This release moves `git_provider_connections` from `marie_studio` to
`marie_runtime` and adds immutable discovery snapshots. It preserves connection
UUIDs, OAuth tokens, invalid states and incoming publish-target foreign keys.
Organization credentials use references; personal OAuth accounts remain personal.
The canonical scheduler bundle is version 105.

1. Back up the shared database using the existing procedure. Stop Studio API and
   the gateway, and allow active repository operations to finish before transfer.
2. From the matching Marie-AI checkout, apply the **whole** handoff file with the
   normal configured PostgreSQL connection:

   ```bash
   psql -v ON_ERROR_STOP=1 -f tools/runtime/git-provider-handoff.sql
   ```

   Studio carries the identical SQL at
   `packages/@marie/db/prisma/migrations/20261117_git_provider_connections/migration.sql`.
   Use the normal migration-history procedure when reconciling a manually applied
   file; do not run a second independent table transfer.
3. Verify table ownership without selecting credentials:

   ```sql
   SELECT to_regclass('marie_studio.git_provider_connections') AS old_registry,
          to_regclass('marie_runtime.git_provider_connections') AS shared_registry,
          to_regclass('marie_runtime.git_repository_discovery') AS snapshots;
   SELECT provider, connection_key, count(*)
   FROM marie_runtime.git_provider_connections
   GROUP BY provider, connection_key;
   ```

   The old registry should be null. If both registries existed before transfer,
   the script refuses to guess which one owns existing identities. Reconcile that
   state before restarting; never drop either table as a shortcut.
4. Deploy the matching SQL bundle, including 104 and 105, to the gateway's actual
   configured SQL mount. Updating the checkout does not update an external
   `/mnt/data/marie-ai/config/psql` copy.
5. Regenerate Studio's Prisma client with the repository's existing build process.
   Restart Studio API, the gateway and the CPU repository executor. The gateway's
   consumer subscriptions must include `repository_management`; its ordinary
   builtin planner dispatches to `repository_executor://repository_operation`.
   OCR and embedding executors do not need changes for provider discovery.

Shared table definitions live in Marie-AI's SQL bundle and identical Helm files.
The application handoff alone contains references to the former Studio table.
Do not copy those application migration references into scheduler baseline SQL.

## Provision credentials in Integrations

A platform administrator provisions an organization service credential and its
approved provider endpoint together. This boundary prevents an organization
manager from naming an unrelated API process variable and sending its value to a
new endpoint. Organization managers can select, validate and disconnect approved
connections, discover repositories, and approve organization availability.

The initial native worker supports `env:NAME` and `env://NAME` references. Configure
the referenced value in both Studio API and the repository worker before saving.
The value stays out of browser responses, job metadata, Git URLs and command
arguments. Other secret backends require a scoped native implementation before
being enabled for shared workers. Personal named connections accept the user's
submitted token, rather than arbitrary server secret references.

System Git uses the existing configured Gitea service and the current user's
**default personal System Git account**. A shared discovery account on the same
URL is not a substitute. Managed source registration checks write access.
External configuration imports name a new private destination in System Git;
upstream reads and destination writes use their separate authorizations.

## Discover, register and attach

1. Open repository discovery from Integrations or a KB's Sources tab. Choose the
   organization connection and request **Discover**. It submits a normal
   `repository_management` DAG. Discovery never indexes content or enrolls new
   repositories automatically.
2. Select repositories across pages and filters. Review and explicitly approve
   organization availability, including any private content. Registration
   revalidates provider repository IDs and returns individual outcomes.
3. Choose **Prepare now** or defer preparation until first KB use. Preparation
   creates an organization-scoped checkout under the worker's configured roots.
   Selecting a deferred registration in the KB editor submits its normal
   preparation workflow; saving is enabled after preparation completes.
4. Use ready registrations in one or more KBs in the selected tenant. Each
   repository/KB binding has its own branch, include/exclude paths, optional first
   refresh and schedule. Already-added bindings retain their existing policy.
5. A source save, refresh submission and schedule activation have separate
   outcomes. Retry a failed refresh submission without recreating its source.
   Schedule failures require schedule review and never start an unrequested
   refresh. Active KB schedules use the existing Automation/Sensors machinery.

Discovery reads through a confirmed terminal provider page before selecting a
snapshot. Partial failures retain the previous complete snapshot. An older DAG
cannot replace a newer completed scan. Repository operations use existing DAG
leases, history and retry controls; they do not maintain another job-state table.

A disconnected or invalid connection withholds private evidence that depends on
it while retaining captures and audit/run metadata. Known repository-access
revocation also requires successful per-repository verification before reuse.
A transient provider outage does not establish revocation.

## Reuse existing checkouts and topology

Configuration **Register server path** verifies the existing checkout against the
user's authorized System Git repository and selected branch. It does not rewrite
`origin`, switch branches or move files. Existing legacy registrations remain
available; a display-only edit does not silently adopt them. Use the explicit
verification option when linking an existing registration.

Configuration file access runs in Studio API; repository capture runs in the CPU
worker. Mount checkouts at the expected paths in each consumer. Existing KB
`repository_roots` and `path_mounts` remain authoritative for worker access.
Use `/manage/workspace-topology` for existing source/worktree relationships,
drift inspection, workspace setup and guarded repair/sync.

Individual public HTTP(S) Git URLs and existing server-path KB registrations remain
available without a provider. Inline credentials are rejected. Private URL
registration uses an approved organization provider connection in this version;
standalone SSH credential and host-key provisioning is not implemented.

An import records its provider destination ID as soon as creation succeeds and
checks that identity and private visibility again before pushing. A crash between
provider creation and recording its ID requires explicit destination review;
retry will not trust a description marker or overwrite an unrelated destination.
Generic preparation cannot complete a failed import.

The local tests cover disposable PostgreSQL, local Git history, native Zoekt
capture and production UI fixtures. They do not establish live provider
permissions, deployed mounts, live scheduled execution or a migrated application
session. Qualify those after applying the migration and restarting the services.
