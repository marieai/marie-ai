---
sidebar_position: 13
---

# LLM dispatcher setup

Use this guide to configure queued LLM dispatch for a regular gateway.
Test fixtures and a fixture manifest are not required for installation.

For a fresh application database, start with
[LLM dispatch provisioning](../getting-started/deployment/scheduler-database-upgrades.md#provision-llm-dispatch-on-a-fresh-database).
Ordinary schema installation creates a baseline fabric row, but provider endpoints,
dispatch pools, and routing rules require deployment configuration.

## Provision a database policy with SQL

Use this example for an initial `default` fabric configuration after installing
the matching scheduler schema. Keep queued producers and submissions stopped.
If the database contains existing application data, take the backups described in
the provisioning guide first. The transaction below refuses to overwrite an
active policy, registered dispatch metadata, or existing pools.

The example provisions four pools in the `default` fabric with the document-size
routing and DRR settings used by the previous installation:

| Pool | Effective pages | DRR quantum | Minimum concurrency | Maximum concurrency | Maximum burst per visit |
| --- | --- | ---: | ---: | ---: | ---: |
| `document-small` | 1–4 | 4 | 1 | 4 | 2 |
| `document-medium` | 5–16 | 2 | 1 | 4 | 1 |
| `document-large` | 17+ | 1 | 0 | 4 | 1 |
| `default` | Catch-all | 1 | 0 | 4 | Unset |

The fabric permits four concurrent dispatch items in total, shared by all pools.
Each pool can use the available shared capacity up to its own maximum. Routing
uses `document.effective_page_count`; requests without matching page-count facts
use the catch-all pool. The DRR quantum controls each pool's scheduling weight.

This example uses the local AIMock provider at `http://127.0.0.1:4010/v1`, with
loopback access enabled, matching the previous setup. The replica includes
`"credential_env": "LLM_PROVIDER_API_KEY"`; supply that variable in the gateway
environment, or remove the field when the provider does not require authentication.
Never put the credential value in SQL.
The address must be reachable from the gateway; `127.0.0.1` refers to the host or
container where the gateway runs.

For another OpenAI-compatible provider, change `base_url` in the replica
registration. Set `allow_loopback` and `allow_private` to match the intended
provider address. The logical inference target is the endpoint group `primary`;
its physical provider is replica `primary-1`. Pools bind to that group and its
revision. This format is supported by both the gateway and Studio's Inference
Targets editor.

If you already provisioned the earlier flat `endpoints` example, use
[Convert the earlier flat endpoint example](#convert-the-earlier-flat-endpoint-example)
instead of rerunning the initial provisioning transaction.

Execute the complete transaction in the application database, using a role
authorized to write the scheduler configuration tables:

```sql
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $bootstrap$
DECLARE
    dispatch_metadata JSONB := $policy$
    {
      "llm_dispatch": {
        "schema_version": 1,
        "endpoint_groups": [
          {
            "group_id": "primary",
            "revision": "r1",
            "replicas": [
              {
                "replica_id": "primary-1",
                "base_url": "http://127.0.0.1:4010/v1",
                "credential_env": "LLM_PROVIDER_API_KEY",
                "allow_loopback": true,
                "execution_limit": 4,
                "execution_bytes": 67108864
              }
            ]
          }
        ],
        "limits": {
          "max_execution_items": 4,
          "max_execution_bytes": 67108864
        }
      }
    }
    $policy$::JSONB;
BEGIN
    IF dispatch_metadata #>> '{llm_dispatch,endpoint_groups,0,replicas,0,base_url}'
       = 'https://your-provider.example/v1' THEN
        RAISE EXCEPTION 'Replace the provider base_url placeholder before provisioning';
    END IF;

    PERFORM 1
    FROM marie_scheduler.llm_queue_fabric_config
    WHERE fabric_group_id = 'default'
    FOR UPDATE;

    IF EXISTS (
        SELECT 1 FROM marie_scheduler.llm_queue_fabric_config
        WHERE fabric_group_id = 'default'
          AND (active_policy_generation IS NOT NULL OR metadata ? 'llm_dispatch')
    ) OR EXISTS (
        SELECT 1 FROM marie_scheduler.llm_queue_pool
        WHERE fabric_group_id = 'default'
    ) THEN
        RAISE EXCEPTION 'Fabric already has policy or pools; use a reviewed policy update';
    END IF;

    INSERT INTO marie_scheduler.llm_queue_fabric_config (
        fabric_group_id, policy, total_concurrent_dispatch, enabled, metadata
    )
    VALUES ('default', 'drr', 4, TRUE, dispatch_metadata)
    ON CONFLICT (fabric_group_id) DO UPDATE SET
        policy = EXCLUDED.policy,
        total_concurrent_dispatch = EXCLUDED.total_concurrent_dispatch,
        enabled = EXCLUDED.enabled,
        metadata = marie_scheduler.llm_queue_fabric_config.metadata || EXCLUDED.metadata;

    INSERT INTO marie_scheduler.llm_queue_pool (
        fabric_group_id, pool_id, display_name, quantum,
        min_concurrent, max_concurrent, max_burst_per_visit,
        enabled, sort_order, metadata
    )
    SELECT
        'default', preset.pool_id, preset.display_name, preset.quantum,
        preset.min_concurrent, 4, preset.max_burst_per_visit,
        TRUE, preset.sort_order,
        jsonb_build_object(
            'preset', 'document_size',
            'llm_dispatch', jsonb_build_object(
                'schema_version', 1,
                'endpoint_group_id', 'primary',
                'revision', 'r1',
                'execution_bytes', 67108864
            ),
            'admission', jsonb_build_object(
                'schema_version', 1,
                'priority', preset.priority,
                'accepting', TRUE,
                'match', preset.match
            )
        ) || CASE WHEN preset.pool_id = 'default'
             THEN '{"role":"catch_all"}'::JSONB ELSE '{}'::JSONB END
    FROM (VALUES
        ('default', 'Default', 1, 0, NULL::INTEGER, 0, 1000000,
         '{}'::JSONB),
        ('document-small', 'Document Small', 4, 1, 2, 20, 10,
         '{"document.effective_page_count":{"gte":1,"lte":4}}'::JSONB),
        ('document-medium', 'Document Medium', 2, 1, 1, 30, 20,
         '{"document.effective_page_count":{"gte":5,"lte":16}}'::JSONB),
        ('document-large', 'Document Large', 1, 0, 1, 40, 30,
         '{"document.effective_page_count":{"gte":17}}'::JSONB)
    ) AS preset (
        pool_id, display_name, quantum, min_concurrent,
        max_burst_per_visit, sort_order, priority, match
    );
END;
$bootstrap$;

COMMIT;
```

If the transaction fails, issue `ROLLBACK` before retrying. This example creates
three document-size admission rules and one catch-all rule. All four pools use
the registered `primary` endpoint within a shared 64 MiB execution budget.
Adapt the rules and capacity to the deployment. The submitted LLM request still selects
its model. Setting only `llm_queue_pool.endpoint_url` does not register a trusted
dispatch endpoint; the metadata above provides the endpoint binding.

Check the staged configuration without returning provider credentials or URLs:

```sql
SELECT fabric_group_id, policy, total_concurrent_dispatch,
       metadata ? 'llm_dispatch' AS has_dispatch_policy,
       active_policy_generation
FROM marie_scheduler.llm_queue_fabric_config
WHERE fabric_group_id = 'default';

SELECT pool_id, enabled, quantum, min_concurrent, max_concurrent,
       max_burst_per_visit,
       metadata #>> '{admission,priority}' AS admission_priority,
       metadata #> '{admission,match}' AS admission_match,
       metadata #>> '{llm_dispatch,endpoint_group_id}' AS endpoint_group_id,
       metadata #>> '{llm_dispatch,revision}' AS revision
FROM marie_scheduler.llm_queue_pool
WHERE fabric_group_id = 'default'
ORDER BY sort_order, pool_id;

SELECT target ->> 'group_id' AS endpoint_group_id,
       target ->> 'revision' AS revision,
       replica ->> 'replica_id' AS replica_id,
       replica ->> 'credential_env' AS credential_env,
       replica ->> 'execution_limit' AS execution_limit
FROM marie_scheduler.llm_queue_fabric_config fabric
CROSS JOIN LATERAL jsonb_array_elements(
    fabric.metadata #> '{llm_dispatch,endpoint_groups}'
) AS targets(target)
CROSS JOIN LATERAL jsonb_array_elements(target -> 'replicas') AS replicas(replica)
WHERE fabric.fabric_group_id = 'default';
```

Expect four pool rows bound to `primary` at revision `r1` and one replica,
`primary-1`. Studio should list `primary` under Inference Targets. A null
`active_policy_generation` and "Policy not applied" are expected before first startup;
"Destination registration complete" confirms the stored configuration, not a
running gateway or resolved credentials.

Restart the gateway with `LLM_QUEUE_ENABLED=true`, `LLM_QUEUE_FABRIC_GROUP_ID=default`,
reachable Valkey configured through `LLM_QUEUE_URL`, any named provider credential
required by your endpoint registration,
and `llm_queue.llm_dispatch.policy_source: database` in its YAML configuration.
When no active generation exists, startup automatically validates and activates
the first policy from the SQL configuration using fenced ownership and drain
checks. A separate Studio save or manual activation is not required. Look for
`LLM routing policy activated: fabric=default generation=1` in the startup log,
then verify that an active generation exists and the dispatcher reports it.

For optional explicit activation of later saved policy changes, use an API key
authorized for `runtime-routing-admin` on this fabric. Set `MARIE_GATEWAY_URL`
to your gateway's HTTP address and supply
`MARIE_GATEWAY_API_KEY` through your approved credential mechanism:

```bash
curl --fail-with-body --silent --show-error \
  --request POST \
  --header "Authorization: Bearer ${MARIE_GATEWAY_API_KEY:?Set an authorized gateway API key}" \
  "${MARIE_GATEWAY_URL:?Set the gateway HTTP address}/api/llm-dispatch/policy/activate?fabric_group_id=default"
```

The activation workflow validates admission rules and builds the immutable
snapshot and digest. Do not manually insert policy-generation rows or set
`active_policy_generation` to an arbitrary value. Confirm that the database shows
an active generation, verify that the dispatcher reports the same generation,
then enable matching queued producers and resume submissions. The policy-activation
endpoint is unavailable while the gateway runs with `LLM_QUEUE_ENABLED=false`.

### Convert the earlier flat endpoint example

Use this only if you already ran the earlier four-pool SQL example, which stored
`metadata.llm_dispatch.endpoints`. The backend accepts that format, but Studio's
Inference Targets editor manages `endpoint_groups`.

Keep queued submissions stopped and take a database backup first. The transaction
below preserves the registered provider settings, credential binding, and four
pools while converting their bindings to endpoint group `primary`. It refuses an
active policy or a configuration that differs from the earlier example. An
already converted configuration also refuses this one-time conversion.

```sql
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $convert$
DECLARE
    fabric_metadata JSONB;
    active_generation BIGINT;
    dispatch_policy JSONB;
    registered_endpoint JSONB;
BEGIN
    SELECT metadata, active_policy_generation
    INTO fabric_metadata, active_generation
    FROM marie_scheduler.llm_queue_fabric_config
    WHERE fabric_group_id = 'default'
    FOR UPDATE;

    IF NOT FOUND OR active_generation IS NOT NULL THEN
        RAISE EXCEPTION 'Conversion requires an existing, unactivated default fabric';
    END IF;

    dispatch_policy := fabric_metadata -> 'llm_dispatch';
    IF jsonb_typeof(dispatch_policy -> 'endpoints') IS DISTINCT FROM 'array' THEN
        RAISE EXCEPTION 'Expected the earlier flat endpoints configuration';
    END IF;
    IF jsonb_array_length(dispatch_policy -> 'endpoints') <> 1
       OR (dispatch_policy ? 'endpoint_groups'
           AND dispatch_policy -> 'endpoint_groups' <> '[]'::JSONB) THEN
        RAISE EXCEPTION 'Conversion requires exactly one flat endpoint and no groups';
    END IF;
    registered_endpoint := dispatch_policy #> '{endpoints,0}';
    IF registered_endpoint ->> 'endpoint_id' IS DISTINCT FROM 'primary' THEN
        RAISE EXCEPTION 'Expected the earlier primary endpoint';
    END IF;

    PERFORM 1 FROM marie_scheduler.llm_queue_pool
    WHERE fabric_group_id = 'default' FOR UPDATE;
    IF (SELECT count(*) FROM marie_scheduler.llm_queue_pool
        WHERE fabric_group_id = 'default') <> 4
       OR EXISTS (
           SELECT 1 FROM marie_scheduler.llm_queue_pool
           WHERE fabric_group_id = 'default'
             AND (pool_id NOT IN ('default', 'document-small', 'document-medium', 'document-large')
                  OR metadata #>> '{llm_dispatch,endpoint_id}' IS DISTINCT FROM 'primary'
                  OR metadata #>> '{llm_dispatch,revision}' IS DISTINCT FROM 'r1')
       ) THEN
        RAISE EXCEPTION 'Expected the earlier four pools bound to primary revision r1';
    END IF;

    dispatch_policy := (dispatch_policy - 'endpoints') || jsonb_build_object(
        'endpoint_groups', jsonb_build_array(jsonb_build_object(
            'group_id', 'primary',
            'revision', 'r1',
            'replicas', jsonb_build_array(
                (registered_endpoint - 'endpoint_id')
                || jsonb_build_object('replica_id', 'primary-1')
            )
        ))
    );
    UPDATE marie_scheduler.llm_queue_fabric_config
    SET metadata = jsonb_set(fabric_metadata, '{llm_dispatch}', dispatch_policy)
    WHERE fabric_group_id = 'default';

    UPDATE marie_scheduler.llm_queue_pool
    SET metadata = jsonb_set(
        metadata, '{llm_dispatch}',
        ((metadata -> 'llm_dispatch') - 'endpoint_id')
        || jsonb_build_object('endpoint_group_id', 'primary')
    )
    WHERE fabric_group_id = 'default';
END;
$convert$;

COMMIT;
```

After conversion, rerun the read-only verification queries above, supply the
credential referenced by the policy in the gateway environment, restart the
gateway, and verify the active policy generation. If you are provisioning a fresh database,
use only the initial provisioning transaction; no conversion is needed.

## Configure the gateway

The SQL provisioning workflow above uses a database policy. In your existing
gateway YAML, keep this block under `with`:

```yaml
llm_queue:
  fabric_group_id: default
  llm_dispatch:
    policy_source: database
```

For a mounted installation, the gateway configuration may be
`/mnt/data/marie-ai/config/service/marie-gateway-4.0.0.yml`.
The selector tells the gateway to load the selected fabric's dispatch policy
from PostgreSQL. It reuses the PostgreSQL connection from `job_scheduler_kwargs`
unless a separate connection is configured under `llm_queue.scheduler`.
Keep the provider and pool settings in the database when using this mode:

| Configuration | Location |
| --- | --- |
| Fabric ID and `policy_source: database` | Existing gateway YAML |
| Provider URL, `credential_env` name, address permissions, and execution limits | Fabric policy metadata provisioned through SQL |
| Four pools, DRR weights, capacity, endpoint bindings, and admission rules | Pool and fabric rows provisioned through SQL |
| Active immutable policy generation | Created by the policy-activation workflow |
| Credential values, `LLM_QUEUE_ENABLED`, and `LLM_QUEUE_URL` | Gateway environment |

Set `LLM_QUEUE_ENABLED=true` and point `LLM_QUEUE_URL` at the intended Valkey
store. Supply the credential named by `credential_env` if that field is present.
Use one consistent fabric ID in the gateway and producers. Enabled startup
requires a complete registered policy; missing credentials, invalid policy,
inconsistent fabric identity, or unavailable storage refuses startup. A disabled
queue selects direct batch execution.

### Save configuration and restart

In Studio's **LLM Scheduler**, click **Save** after editing a target, scheduler,
or pool. This writes the settings to PostgreSQL immediately, including when the
gateway is offline. The success message and pending-change banner explain:
**Saved. Changes take effect after the gateway restarts.** There is no second
save or scheduling step. Each Save replaces the previous pending configuration
with the latest saved settings.

The running gateway continues using its active immutable policy revision. On
restart, it loads the saved configuration and resolves its credentials before
using the old policy's credentials. It validates the configuration, acquires
exclusive fenced ownership, closes old admission gates, and requires durable
PostgreSQL routes and the Valkey queue to be drained. It then replaces idle
endpoint/route registrations, creates or reuses an immutable revision, updates
the active pointer and consumes the pending change in one PostgreSQL transaction.
No schema migration is needed. An incomplete or disabled configuration is saved
without scheduling an enabled runtime policy; complete setup before starting
queued dispatch.

Missing credentials, another dispatcher owner, invalid or changed configuration,
or outstanding work blocks startup activation. A failed preflight preserves the
pending change and active revision. It can leave admission gates closed. If work
must drain, choose **Cancel pending change** and restart the existing active
configuration; normal dispatcher startup restores its admission gates. Drain the
work, click **Save** again, then restart. Existing work is never discarded to
force a configuration change. Fabric storage limits remain immutable.

This also lets you repair a credential binding that prevents startup: save the
correct environment variable name, or remove it for a provider that requires no
authentication, then restart. Credential values still belong in the gateway
environment. If startup activation commits but a later runtime failure occurs,
the new revision remains active; correct that runtime failure and restart.
PostgreSQL and Valkey do not share a distributed transaction.

Each replica offers **Test connection**, an optional, timestamped TCP reachability
check from the gateway's network context. It uses the selected private-network and
loopback permissions and a three-second timeout. It sends no HTTP or inference
request and does not check credentials, TLS negotiation, models, or inference
readiness. The gateway must be available to run it. Results are not persisted or
used as health, save, activation, or startup conditions; a target can come online
after a failed check.

### Alternative: a static YAML policy

`config/service/llm-dispatch.example.yml` is a static policy fragment with
endpoints, lanes, and limits supplied in YAML. Use it as a reference only when
choosing `llm_dispatch.policy_source: static`; static is also the default when
the selector is omitted. Database mode loads the persisted policy instead of
the static endpoint and lane definitions. You do not need to copy that example
into your gateway configuration for the SQL provisioning workflow.

This is a fresh-install setup. The store initializes current fabric policy in
`llm:dispatch:{fabric:<id>}:`; producers attach to that policy. Startup does not
scan historical queue namespaces or require a legacy migration procedure.
### Reserve gateway ports on Linux

The regular gateway uses HTTP port `51000` and gRPC port `52000`. These ports
fall within the usual Linux ephemeral range (`32768–60999`), so a browser or
another outbound client can occupy them before the gateway starts. Reserve the
configured gateway ports on the host where the gateway binds them.

Check the host's allocation range and existing reservations:

```bash
sysctl net.ipv4.ip_local_port_range net.ipv4.ip_local_reserved_ports
```

If the reserved-port list is empty, persist both gateway ports and apply the
setting immediately:

```bash
printf '%s\n' 'net.ipv4.ip_local_reserved_ports = 51000,52000' \
  | sudo tee /etc/sysctl.d/99-marie-gateway-ports.conf

sudo sysctl -p /etc/sysctl.d/99-marie-gateway-ports.conf
sysctl net.ipv4.ip_local_reserved_ports
```

If the host already reserves ports, include those ports or ranges alongside
`51000,52000` in the configuration value. Writing this setting replaces the
entire reservation list. Use the actual gateway ports if your deployment uses
different ones, and keep any existing sysctl configuration consistent with this
file. Reservations exclude ports from automatic outbound allocation while
allowing explicit server binds; see the [Linux kernel IP sysctl documentation](https://docs.kernel.org/networking/ip-sysctl.html#ip-variables).

Reservations do not close existing connections. If startup still reports
`Address already in use`, inspect all TCP states rather than only listeners:

```bash
ss -tanp '( sport = :51000 or sport = :52000 )'
```

An outbound connection can block a wildcard gateway bind even when no server is
listening. Close the conflicting connection or stop its owning application,
then retry once the port is free. A closed connection may remain in `TIME_WAIT`
until the kernel releases it.

See [LLM tracking and observability](./llm-tracking.md) for telemetry configuration.

## Resolve startup failures

For missing policy, pool, credential, fabric, or store configuration, see
[LLM dispatch provisioning and startup diagnostics](../getting-started/deployment/scheduler-database-upgrades.md#provision-llm-dispatch-on-a-fresh-database).

## Run optional qualification

For automated deployment and failure-recovery testing, follow the
[LLM dispatcher qualification runbook](https://github.com/marieai/marie-ai/blob/develop-agents/tools/stress/llm-dispatcher-qualification.md)
at `tools/stress/llm-dispatcher-qualification.md` in your source checkout.
That runbook covers the isolated test services, fixture manifest, load and
recovery scenarios, and cleanup. These steps are optional for normal setup.
