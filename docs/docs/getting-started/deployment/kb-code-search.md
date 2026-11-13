# Configure repository capture and exact code search

Repository sources capture a configured Git branch and search its eligible files with Zoekt. They can share a knowledge base with document and plugin sources. Exact code search is a separate operation from the KB's vector or document full-text search; it requires no embeddings or Tree-sitter navigation index.

## Prepare the CPU worker and Zoekt

Install the pinned binaries with `tools/kb/install-zoekt.sh /usr/local/bin` in the CPU worker image and private Zoekt service image. The pinned sourcegraph/zoekt revision is `d64299b4420d0f0a46082c26d34ee956184558ed`. Its Go module selects the required toolchain through `GOTOOLCHAIN=auto`; do not install an unpinned `latest` build.

Mount one persistent directory at `/var/lib/marie/kb-code` in the repository worker and the private Zoekt service. The worker writes shards into its `served` child directory. Start Zoekt with:

```bash
mkdir -p /var/lib/marie/kb-code/served
zoekt-webserver -rpc -html=false -listen :6070 -index /var/lib/marie/kb-code/served
```

Keep this unauthenticated internal API off public networks. Only Marie's authenticated KB gateway should reach it. Zoekt contains staged and current shards; the gateway always selects an authorized exact build.

Use the standalone CPU service `config/service/kb/marie-kb-code-4.0.0.yml`, or merge [the example service configuration](../../../../config/service/kb-code-search.example.yml) into the existing deployment. The standalone service uses the normal PostgreSQL/discovery configuration and takes its startup key from `MARIE_API_KEY`. Include `kb_repository_refresh` in an explicit gateway `job_scheduler_kwargs.queue_names` subscription. Set `gateway.with.kb.code_executor` to `repository_executor`, deploy the CPU `RepositoryExecutor`, and reuse the existing `storage.psql` block. Keep the current vector/text executor configuration. The scheduler's registered `kb_repository_refresh` planner sends work to `repository_executor://refresh_repository`; sensors and the gateway do not fetch or index Git content.

## Register a source

Organization, tenant and registered repositories are shared in `marie_runtime`. Follow the [shared platform transfer](shared-platform-registry.md) before starting services against an older Studio-owned registry. Backend schema 102 includes the `code` asset type; the transfer also adds it when moving legacy enums. Do not replay the historical Studio enum migration after transfer.

In a knowledge base's **Sources** tab, choose **Add repository**. Select a registered repository or use **Register a code repository** to register an existing checkout using the normal repository registry. Organization repository management permissions apply to registration. The checkout must be available inside the CPU worker's configured `repository_roots`. `path_mounts` translates existing registered Studio paths when hosts use different mount prefixes; it does not create a second repository registration or copy credentials.

Choose the branch, include/exclude globs, and optional daily schedule. When a KB has no execution workspace yet, saving its first repository source binds it to your authenticated active workspace in the same catalog transaction. A missing or inaccessible active workspace is rejected. Later source edits preserve the assigned workspace. A source represents one repository/branch; add separate sources for another repository or branch. Globs use case-sensitive full-path matching; `*` also spans directories. Use `src/*` to include a subtree. A local example is:

```json
{
  "repositoryId": "40000000-0000-4000-8000-000000000001",
  "branch": "main",
  "include": ["src/*", "lib/*"],
  "exclude": ["*/generated/*"],
  "fetch": true,
  "dirty": false
}
```

Fetch uses the checkout's registered `origin` and existing Git credential setup. No provider token is stored on the KB source. **Include local uncommitted files** explicitly captures actual worktree bytes and digests, requires the selected checked-out branch, and disables fetch. An unchanged commit hash does not skip changed dirty files.

Secret paths (`.env*`, `secrets/`, `*.pem`, `*.key`) are always excluded. VCS/indexer control directories (`.hg`, `.svn`, `.sourcegraph`) are also counted as omissions so their ignore files cannot silently change the captured file set. Symlinks and submodules are not followed; binary and files above 1 MiB are excluded. Captures report these omissions. Zoekt receives the exact staged file set with its directory ignore defaults disabled. The adapter bounds scans to 100,000 members and 128 MiB of eligible file bytes; exceeding a scan budget fails the refresh and preserves the previous result. Git operations have a 120-second timeout, and indexing has a 300-second timeout.

Save the source and choose **Refresh** for the initial capture. Saving a schedule leaves it inactive; activate it explicitly. Schedules reuse existing cron/timezone preview, occurrence identities, scheduler deduplication and retries. Pause an active schedule before editing it. Manual and scheduled refresh use the same source identity. `replace` permits a later run after terminal work; it does not cancel an active run or create a durable queue of manual refresh requests.

A completed refresh reports the actual captured commit. If the branch advances during indexing, the successful result remains commit A and records the newer observed input. An active source schedule admits a follow-up through the existing scheduler; an unscheduled source shows updates pending until its next manual refresh. A Git/input change is not reported as an indexing failure. A failed follow-up occurrence falls back to normal cron admission; an acknowledged pending key cannot block later daily runs. Failure of the optional post-build HEAD check does not discard the successfully selected capture.

## Search and inspect

In **Retrieval**, select repository sources and choose **Literal**, **Regex** (Zoekt/RE2), or **File path**. Each result shows its source, captured branch/commit, path and line. Searches are case-sensitive and bounded to 50 results per request and a two-second engine budget; truncation is disclosed. Narrow a broad pattern or source selection when a limit is reached.

In **Workspace assistant → Search code**, select the repository sources to explore. Opening a match loads the full captured file in the shared source inspector and focuses the matching line. The reader supports files up to 2 MiB; the capture admission limit remains 1 MiB. Search highlights, Go to line, Back/Forward and selected excerpts use the original captured bytes. Quotes are bounded to 8,000 Unicode characters; there is no fixed 100-line limit. Attach a selected range or the matched code to a question through the same evidence contract as document passages.

The assistant's **Query** mode combines text, quoted phrases, `AND`, `OR`, `NOT`, `-` exclusions and parentheses. For example, `file:store.py lang:Python (settings_digest OR fingerprint)` searches matching Python files in the selected sources. Filters narrow that selection:

| Filter | Matches |
| --- | --- |
| `repo:"Marie AI"` | Selected source or registered repository name/identity |
| `file:\.py$` | File path pattern using RE2 |
| `lang:Python` | Indexed programming language |
| `rev:HEAD` | Current capture; a retained commit prefix also selects its saved capture |
| `sym:settings_digest` | Exact captured symbol or qualified definition name |
| `context:"Engineering"` | Selected knowledge base name or ID |
| `content:"exit boot"` | File content phrase |

**Match case** is on by default. Enable **Use regular expressions** to interpret all content terms as RE2 patterns, including quoted terms. Quote regex patterns containing spaces or parentheses, for example `file:store.py content:"settings_(digest|other)"`. Turn regex off for literal phrases. File patterns always use RE2. **Exact text**, **Regex** and **File paths** remain available as separate modes; existing search links retain their meaning.

Use `/` to focus search, arrows and Tab to select a completion, and Ctrl/Command + `/` to open the syntax guide. Enter or **Search code** submits the query. The history button opens a filterable list and restores each query's mode, case and regex settings. Browser history storage keeps at most 50 searches, separated by signed-in user, organization, workspace and selected KB/source set.

Query expressions allow at most 1,000 characters, 64 terms, 12 grouping levels, eight expanded alternatives and 128 expanded terms. Searches keep the selected server-owned capture scope, with a two-second native engine budget per branch and an eight-second budget per source. Result, symbol-definition and parser limits are disclosed; a negative symbol filter only admits files whose parser records establish the symbol's absence. Revision filters search current or retained captures and do not fetch arbitrary Git revisions.

The assistant sidebar holds the search title, description and workspace actions. The shell hamburger collapses that sidebar on desktop and opens its navigation drawer on mobile. Use **Maximize** in the source panel to expand the captured-code workspace.

**Files** browses the actual capture manifest, and **Symbols** shows the file outline. Both stay beside the editor. **Explore** opens a resizable bottom panel; F12 opens definitions and Shift+F12 opens reference matches for the selected symbol. Opening a candidate preserves the results and reading position, including page and scroll position when moving between files. Choose **This repository** or **Selected repositories** to explore the authorized code sources across the selected knowledge bases. Results retain source, capture, path and line identity. Pagination restarts explicitly if captures, availability or result partitions change.

The captured revision selector opens only retained captures containing that file. **History** shows actual Git commits reachable from the selected captured commit. An **Open capture** action is available only when a retained file digest matches that commit's blob; dirty worktree bytes are never presented as the committed version. Git metadata failure leaves the captured file readable. Source/repository/configuration permissions are checked again before returning file or navigation results. Removing a source from the conversation selection closes its inspector and removes its navigation locations and attachment controls.

## Add syntax navigation

The same repository refresh also derives bounded navigation artifacts from the captured bytes using the existing extraction plugin's `TreeSitterProvider`. Apply backend schema 103 (`kb_code_navigation`) through the normal scheduler bootstrap. These are source/build lookup records in `marie_scheduler`; organization, tenant and repository registration remain in `marie_runtime`.

The CPU repository worker needs `tree-sitter-language-pack==1.12.5` and the existing extraction plugin package on its Python import path. In a source deployment, include `packages/marie-plugin-document-extraction` in `PYTHONPATH` alongside existing runtime paths; install the pinned grammar package into that worker's environment. Make the language grammars available in its runtime cache before activation. The document extraction and vector workers do not acquire a new navigation responsibility. If the package or grammar is unavailable, the capture still supports exact search and original-range reads; navigation reports unavailable coverage.

Definitions use exact names or qualified names from syntax artifacts. References use whole-word Zoekt matches, including declarations and variable uses, and remain available without a grammar. These are heuristic candidates, not compiler-resolved bindings. Each query returns at most 50 candidates with at most 8,000 characters of candidate fields and 2,000 characters of context per row. Available totals exclude lines too large to quote; matched and omitted counts remain separate. Definition coverage includes parser gaps across the repository, rather than only the opened file. Parsing retains at most 1,000 records and 128,000 serialized characters per file; oversized names are omitted and reported as limited. Partial parses remain marked on their candidates, including when a record limit is reached. Unsupported and incomplete coverage is shown in the inspector.

Attach captured code to the current chat message to enable `kb_code_navigate` and `kb_code_read`. These tools can use that attachment and candidate references returned during the same turn; arbitrary paths and repository IDs are not accepted. All model iterations share a limit of eight calls, 50 candidates and 32,000 serialized response characters. Each read reuses authenticated source checks and the 8,000-character quote budget; full-file browsing is independent of that model evidence allowance. Claims retain citations to the returned original code ranges. The tools are offered only with tool-capable model providers.

Existing captures acquire navigation on their next normal refresh; the parser artifact digest participates in the unchanged check. No repository job is submitted merely by opening the inspector. Locally qualified parsing, API, chat and browser fixture behavior does not activate or qualify a live repository deployment; the CPU worker, gateway routing, shared volume and chosen tenant source still need their normal deployment configuration.

## Refresh and cleanup behavior

One current complete index is selected per source. Daily checks compare captured file digests and scope/processing settings, and require a usable index before skipping indexing. Zoekt performs a complete branch/shard build for changed eligible content; no file-incremental indexing claim is made. Failed or incomplete scans/builds keep the previous authorized result. Scheduler state, DAG identity and existing timestamps remain the execution authority; there is no new lifecycle table or unbounded daily archive. The worker keeps up to five current and superseded captures per source by default; set `code_search.retained_captures` from 1 to 50 to adjust this bound.

The worker stages unique owned shards, verifies loaded repository/branch/commit and document coverage through Zoekt's real API, and then selects the handle. `coverage_timeout_seconds` defaults to 90 seconds, longer than Zoekt's one-minute rescan fallback. It is configurable from 1 to 600 seconds. Local inotify-capable shared mounts are locally qualified; verify actual cross-host/NFS event and visibility behavior before deployment. Scoped shared read locks and exclusive selection/cleanup locks cover backend calls so a reader cannot resolve a handle that is retired before its request reaches Zoekt. The refresh lock additionally fences direct/recovered executions against concurrent effective writes. Changing a source repository/branch/path scope temporarily makes the old capture unavailable until a matching refresh succeeds. Current and retained captures remain available for digest-checked original-file inspection while their configuration is authorized. Superseded captures outside the retention limit, and abandoned owned attempts, are retired on the next successful refresh.

Removing a repository source or its KB withdraws access and removes only its owned index files. The registered repository and Git history are preserved. Removal during an in-flight refresh stays pending and is retryable after the refresh stops. A cleanup failure leaves removal pending so it can be retried. A retirement failure after a complete successful selection reports cleanup pending and is retried during normal subsequent refresh; it does not misreport the selected build as failed. Tenant/repository disablement blocks further source delivery; administrative removal performs physical cleanup.

Local fixture/service qualification does not establish a deployed gateway, tenant activation, scheduler-to-remote-worker delivery, remote Git credential setup, or shared-volume behavior across production hosts. Verify those deployment-specific paths before activation.
