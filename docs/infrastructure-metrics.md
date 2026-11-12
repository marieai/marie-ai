# Scope infrastructure telemetry

The Studio Infrastructure page reads exported metrics from ClickHouse. Select a Runtime Fabric and either all its assigned gateways or one gateway. Execution-disabled gateways remain available for monitoring.

## Use the existing gateway connection

Register the gateway's normal host, port and credentials in Studio. No additional gateway ID environment variable is required. Studio reads the existing runtime instance identity from `/api/operations/health` through that connection and associates matching telemetry with the existing gateway record.

Observed runtime instances are retained automatically in the gateway record's metadata, preserving scoped history across restarts. Distinct registered ports are discovered independently. A runtime instance reported by multiple registrations is treated as ambiguous; correct duplicate gateway endpoints instead of assigning another manual ID.

Keep the gateway's existing Runtime Fabric configuration aligned with Studio. The same telemetry works with queued dispatch disabled. Preserve the existing OTel metrics exporter and collector configuration. No ClickHouse schema migration is required. Metrics that predate runtime identity labels remain stored and are excluded from scoped queries with an explanatory notice.

## Interpret the page

- Request counts, latency and endpoint charts follow the selected date range and retain separate process counter epochs.
- Current executor capacity uses successful observations from the last 180 seconds. Missing measurements display as unavailable.
- The reporting table shows gateway names, last report times, heartbeat status and capacity freshness separately. A healthy heartbeat can accompany missing capacity data; the page qualifies partial totals. Repeated exports retain the original capacity observation time.
- Gateways reporting the same etcd cluster and effective storage prefix count as one shared pool. Distinct pools are summed; historical buckets apply the same deduplication.
- Fabric scope uses current gateway assignments. Samples from removed gateways remain stored but do not appear under the fabric's current inventory.

Infrastructure telemetry requires an active global administrator. Selecting a gateway outside the fabric is rejected before connecting to gateways or reading telemetry.

## Diagnose missing telemetry

Events use the gateway connection selected by Studio. Infrastructure also requires metrics to reach ClickHouse. A working event stream confirms the connection but does not establish metrics ingestion.

If Studio cannot discover a runtime identity, the page identifies the affected gateway connection. Check its saved endpoint and credentials and refresh. Known process history remains available when a gateway is offline. Newly started processes are associated on the next successful discovery; no separate identity setting is needed. Allow one metrics export interval, normally 60 seconds, for new measurements.
