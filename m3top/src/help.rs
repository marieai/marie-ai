#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HelpTopic {
    Pools,
    Requests,
    Fabric,
    Scheduler,
    Runtime,
    ExecutionHistory,
    Keys,
}

impl HelpTopic {
    pub const ALL: [Self; 7] = [
        Self::Pools,
        Self::Requests,
        Self::Fabric,
        Self::Scheduler,
        Self::Runtime,
        Self::ExecutionHistory,
        Self::Keys,
    ];

    pub fn shifted(self, delta: isize) -> Self {
        let index = Self::ALL
            .iter()
            .position(|topic| *topic == self)
            .unwrap_or(0);
        Self::ALL[(index as isize + delta).rem_euclid(Self::ALL.len() as isize) as usize]
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Pools => "Pools",
            Self::Requests => "Requests",
            Self::Fabric => "Fabric",
            Self::Scheduler => "Scheduler",
            Self::Runtime => "Runtime",
            Self::ExecutionHistory => "History",
            Self::Keys => "Keys",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Pools => {
                "How queued work shares endpoint capacity. Select a pool to see its accounting below the table."
            }
            Self::Requests => {
                "A bounded sample of request metadata. This list can be smaller than the runtime totals."
            }
            Self::Fabric => {
                "Dispatcher ownership, physical endpoints, and how jobs are routed into pools."
            }
            Self::Scheduler => {
                "Read-only activated scheduler configuration. Select a pool and press Enter for all settings and admission conditions."
            }
            Self::Runtime => {
                "The summary above every dispatcher tab, stream status, and how to read missing values."
            }
            Self::ExecutionHistory => {
                "Retained worker lifecycle events. Select a row and press x to export its diagnostics as Markdown."
            }
            Self::Keys => {
                "Keyboard shortcuts for m3top. Close help to use these on the underlying screen."
            }
        }
    }

    pub fn entries(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::ExecutionHistory => &[
                (
                    "selected event",
                    "The selected row identifies a particular worker event and attempt. A later retry may have changed the current job state.",
                ),
                (
                    "x / Markdown export",
                    "Fetch diagnostics for the selected task and event, including recorded errors and tracebacks, current scheduler state, worker and scheduler history, and durable attempts. The saved path remains in a popup until you close it.",
                ),
                (
                    "c / y / X",
                    "In the export popup, c copies the file path and y copies the Markdown report. Esc or Enter closes it. Uppercase X reopens the last export from any screen. Clipboard requests over SSH need terminal support for OSC52.",
                ),
                (
                    "export directory",
                    "Reports go into m3top-reports under the directory where m3top was started. Use --export-dir PATH to choose a different directory. Each export creates a new file.",
                ),
                (
                    "scope and limits",
                    "Exports include up to 250 newest retained records per source, plus the selected event. Missing or truncated information is marked. Raw task payloads and runtime environment objects are excluded.",
                ),
                (
                    "gateway access",
                    "Full exports require the updated gateway and runtime-observability access for the selected fabric. A failed request creates no report.",
                ),
                (
                    "n / p",
                    "Move through server pages. The export follows the selected row, including tasks other than the original job used to open the DAG history.",
                ),
            ],
            Self::Pools => &[
                (
                    "pool",
                    "A logical queue with its own scheduling weight, concurrency limits, and endpoint-group binding.",
                ),
                (
                    "target",
                    "Physical replica and listening port, such as aimock-local:4010. When no unique replica is published, the group and replica count are shown instead.",
                ),
                (
                    "policy / drr",
                    "DRR means Deficit Round Robin: pools take turns earning credits and spending them to dispatch requests of different estimated costs.",
                ),
                (
                    "queued",
                    "Current ready-queue depth. Requests already claimed or executing are counted separately.",
                ),
                (
                    "oldest",
                    "Age of the oldest ready request reported by the gateway. A dash means no age was supplied.",
                ),
                (
                    "in flight / use",
                    "Current reserved request slots / pool maximum. The compact use bar shows this ratio as a percentage. A reservation may remain held while an outcome is unknown.",
                ),
                (
                    "min/max/burst",
                    "Minimum protected concurrency / maximum concurrent requests / maximum launches during one scheduler visit. These are request slots, not threads or tokens.",
                ),
                (
                    "weight",
                    "The DRR quantum: credits added per scheduling visit. Higher weights give more scheduling credit, subject to request costs and capacity limits.",
                ),
                (
                    "credits",
                    "The current DRR deficit balance available to pay request costs. Zero credits does not imply a disabled pool.",
                ),
                (
                    "reserved",
                    "Current reserved request items and payload bytes. This is capacity accounting, not token use or GPU memory.",
                ),
                (
                    "accepted / completed",
                    "Gateway-published totals for work routed to the pool and completed work. These are counters, not per-second rates; routing diagnostics can count work units rather than individual calls.",
                ),
                (
                    "protected / borrowed",
                    "Reserved slots within the pool's configured minimum / reserved slots above that minimum. Both contribute to current reservations.",
                ),
                (
                    "ready / claimed / executing",
                    "Ready requests await a claim; claimed requests have been reserved by a dispatcher; executing requests have started an endpoint call. Published state counts appear in the selected-pool panel.",
                ),
                (
                    "outcome_unknown",
                    "The endpoint call's outcome is unresolved. Its reservation can remain held to prevent unsafe duplicate execution.",
                ),
                (
                    "projection pending",
                    "Durable job routes that have not yet been projected into the runtime queue. These can exist before requests appear in the ready queue.",
                ),
                (
                    "drain refs",
                    "References that still keep a pool in use. The total may combine active database routes and runtime queue/processing records; it is not a unique-request count.",
                ),
                (
                    "charged / refunded / committed",
                    "Scheduling cost charged to claims / cost returned for definitely-unsent work / scheduler committed-charge accounting. These are fairness units, not currency or token billing.",
                ),
                (
                    "head cost",
                    "Estimated scheduling cost of the request at the front of the ready queue. DRR credits must cover this cost before it can be claimed.",
                ),
                (
                    "skipped",
                    "Published counts of scheduler visits that could not launch work, grouped by reason. These counts do not mean requests were dropped.",
                ),
                (
                    "waiting on",
                    "The published reason the pool cannot dispatch now. This can describe endpoint capacity or health even when the pool's own queue is empty.",
                ),
                (
                    "empty / disabled",
                    "No ready request is available / the pool or its eligible replicas are disabled.",
                ),
                (
                    "lane_capacity / global_capacity",
                    "The pool's concurrent-request limit / the dispatcher's shared concurrent-request limit is reached.",
                ),
                (
                    "replica_capacity",
                    "Eligible replicas lack request slots or payload-byte capacity for the next request. Check endpoint reservations as well as the pool maximum.",
                ),
                (
                    "insufficient_credit",
                    "The head request costs more than the current DRR credit balance. Further scheduling visits can add credit.",
                ),
                (
                    "replica_unavailable",
                    "No eligible replica is available, for example because it is disabled or its admission gate is closed.",
                ),
                (
                    "circuit_cooldown",
                    "Endpoint calls are blocked after a failure until the circuit's next probe is eligible. See Fabric for the cause and next-probe time.",
                ),
                (
                    "probe_unresolved",
                    "An endpoint recovery probe still has an unresolved outcome. Normal dispatch remains gated until it settles.",
                ),
            ],
            Self::Requests => &[
                (
                    "stage",
                    "The request's published lifecycle stage. The selected-request panel shows the full stage name; the table uses short tokens.",
                ),
                (
                    "[PEND]",
                    "ready, queued, or pending: work is waiting for a dispatcher claim.",
                ),
                (
                    "[RTRY] / [GATE]",
                    "retry_wait: delayed until eligible again / gated: admission or dispatch is blocked.",
                ),
                (
                    "[CLM] / [DISP]",
                    "claimed: reserved by a dispatcher / dispatching: being handed to an endpoint.",
                ),
                (
                    "[EXEC] / [RPLY]",
                    "executing: the endpoint call is active / replying: its result is being returned.",
                ),
                (
                    "[DONE] / [FAIL] / [?]",
                    "completed / failed, expired, or abandoned / an unrecognized lifecycle stage. Terminal rows appear only when published in the sample.",
                ),
                (
                    "request / pool / model / dispatcher",
                    "Request identity, assigned logical pool, requested model, and the dispatcher handling the request.",
                ),
                (
                    "waited / running",
                    "Published queue-wait age / in-flight age. These are request ages, not average latency metrics.",
                ),
                (
                    "admitted / deadline / eligible",
                    "When the request entered the system / when it expires / the earliest time it may be retried. 'In' means future; 'ago' means the timestamp has passed.",
                ),
                (
                    "source",
                    "store: queue-state metadata / dispatcher: the dispatcher's local observation. This identifies the observation source, not the caller.",
                ),
                (
                    "attempt / endpoint / execution",
                    "Attempt identifier, target endpoint identifier, and execution sequence when published. These help correlate a request with its execution.",
                ),
                (
                    "payload / cost",
                    "Payload size in bytes / estimated scheduling cost. Request bodies, prompts, and replies are not displayed.",
                ),
                (
                    "last error",
                    "The published error category or diagnostic from the latest request observation.",
                ),
                (
                    "sample / filters",
                    "The title shows visible rows / published sample rows. Search and pool filters narrow this sample; they do not change dispatch or load additional history.",
                ),
                (
                    "/ search / c pool",
                    "Search matches request, pool, model, stage, and dispatcher identifiers. Press c to cycle the pool filter, including all pools.",
                ),
            ],
            Self::Fabric => &[
                (
                    "dispatcher / role",
                    "Dispatcher process identity and its published ownership role. An owner holds the lease used to coordinate dispatch.",
                ),
                (
                    "[RUN] / [DRN] / [STOP] / [ERR]",
                    "Running / draining existing work / not running / running with a recorded last error. An ERR label alone does not mean the process has stopped.",
                ),
                (
                    "gen o/c",
                    "Owner-lease generation / local configuration generation. These identify ownership and configuration revisions, not request counts.",
                ),
                (
                    "items",
                    "Published processed-item counter for the dispatcher. It is not the number of requests currently queued.",
                ),
                (
                    "in flight / fails / last error",
                    "Current in-flight requests / dispatcher concurrency cap; published execution-failure counter; most recent recorded error. A past error can remain visible after the call ends.",
                ),
                (
                    "runtime_error / recorded error",
                    "runtime_error is the gateway fallback for an unrecognized error category. The selected dispatcher panel explains known categories and shows the last execution-failure age when published. The category can change separately from that timestamp. Errors remain recorded after successful calls; gateway/provider logs may have more detail.",
                ),
                (
                    "seen by",
                    "The full ID of the selected dispatcher. Selecting a different dispatcher changes its observation rows in the Endpoints panel.",
                ),
                (
                    "source",
                    "An endpoint-group ID identifies the group/replica observation. 'dispatcher' identifies the selected dispatcher's own endpoint observation. The same endpoint can appear in both sources.",
                ),
                (
                    "endpoint / port / > marker",
                    "Physical replica identifier / actual listening port. A triangle before the name marks the replica selected by the published group observation.",
                ),
                (
                    "[OK] circuit closed",
                    "The circuit allows normal calls. Available capacity and other admission gates still apply.",
                ),
                (
                    "[OPEN] circuit open",
                    "Normal calls are blocked after endpoint failures. The next probe time indicates when a recovery check becomes eligible.",
                ),
                (
                    "[HALF] / [?] circuit",
                    "Half-open: recovery is being tested with limited probing / unknown: no recognized circuit state was published.",
                ),
                (
                    "free/max / reserved",
                    "Currently available request slots / replica execution limit; held request reservations. Free slots do not override an open circuit. N/A means the observation omitted the value.",
                ),
                (
                    "cause",
                    "Published endpoint failure category or waiting reason. call_timeout means the call exceeded its timeout; connect_refused means the connection was refused.",
                ),
                (
                    "next probe",
                    "Relative time until, or since, the published recovery-probe eligibility timestamp. A past timestamp does not prove a probe succeeded or that the endpoint is healthy.",
                ),
                (
                    "matched / rejected",
                    "Published routing counts by admission source or rejection reason. Automatic uses configured rules; operator_override records an explicit routing override.",
                ),
                (
                    "recent routes / effective pages",
                    "Recent job-to-pool decisions with their source and one reported fact: effective page count. This is the number of distinct requested pages, or total document pages when no subset is requested. It does not describe the complete admission rule.",
                ),
                (
                    "admission rules",
                    "Rules match trusted submission facts: total, requested, or effective pages; workload kind and mode; pipeline stage; request source. Every condition in a rule must match. Only enabled, accepting rules participate; the matching rule with the lowest priority number wins, with a catch-all fallback.",
                ),
                (
                    "routing versus scheduling",
                    "Admission chooses the pool. DRR weight, protected slots, concurrency limits, and visit burst limits control when queued work can dispatch. The recent-route rows do not publish the full matched conditions or all submission facts.",
                ),
            ],
            Self::Scheduler => &[
                (
                    "policy source",
                    "The activated policy generation, with dispatcher application status. Draft edits in Studio are excluded. Older gateways expose runtime-observed settings only; unavailable admission rules remain unknown.",
                ),
                (
                    "execution / accepting",
                    "Execution enabled permits dispatch and draining existing work. Accepting permits new automatic admissions. A pool can stop accepting while continuing to drain.",
                ),
                (
                    "weight / min / max / burst",
                    "DRR credits earned per visit / protected minimum request slots / maximum concurrent requests / maximum launches per visit. Default burst uses the scheduler's default; N/A means not published.",
                ),
                (
                    "priority / admission rule",
                    "All trusted-fact conditions must match. Lower numeric priorities are evaluated first. A catch-all handles remaining unmatched work. These conditions choose the pool before dispatch scheduling.",
                ),
                (
                    "target / bytes cap",
                    "Logical inference target bound to the pool / maximum bytes of payloads reserved concurrently. Targets list physical replicas, addresses, execution limits, and payload limits.",
                ),
                (
                    "Enter / Esc / t",
                    "Open the selected pool's complete configuration / return to the Scheduler tab / switch between target and pool selection. Within pool details, j/k or arrows scroll fields and Home returns to the top.",
                ),
                (
                    "awaiting application / partial",
                    "Dispatchers have not confirmed the activated generation / the gateway truncated configuration rows. Neither indicates draft settings are already running.",
                ),
            ],
            Self::Runtime => &[
                (
                    "waiting / in flight",
                    "Published pending-request count / currently in-flight requests. These totals can exceed the bounded sample shown on Requests.",
                ),
                (
                    "dispatchers",
                    "Running dispatchers / registered dispatchers. The header health marker reflects these counts; it does not certify endpoint health.",
                ),
                (
                    "accepted / completed",
                    "Sum of available pool counters in the snapshot. They describe reported work totals, not throughput rates or necessarily individual endpoint calls.",
                ),
                (
                    "exec failures",
                    "Published execution-failure counter. Failures can be retried, so this is not necessarily a count of permanently failed jobs.",
                ),
                (
                    "producer drops / malformed",
                    "Requests or replies dropped because the producer is offline / requests dropped because their metadata is malformed.",
                ),
                (
                    "projection pending",
                    "Routes awaiting projection into runtime queues. This warning can explain work that is accepted but not yet visible as a queued request.",
                ),
                (
                    "policy syncing",
                    "Observed policy generation differs from the desired generation. The runtime has not yet reported the desired admission policy as synchronized.",
                ),
                (
                    "sample capped / pool list truncated",
                    "The request sample reached its limit / the gateway omitted some pool rows. Treat visible rows as a partial view.",
                ),
                (
                    "observation stale",
                    "The gateway marked dispatcher observations stale. Snapshot arrival and dispatcher freshness are separate signals.",
                ),
                (
                    "live / subscribed / retrying",
                    "Snapshots are arriving / the stream subscription is open / the connection is being retried. Subscribed without a snapshot can indicate missing runtime-observability access for this fabric.",
                ),
                (
                    "age / PAUSED",
                    "Header age is time since the last successful displayed update. Pausing freezes the display while the subscription stays open; reconnect requests a fresh stream.",
                ),
                (
                    "N/A / dash / zero",
                    "N/A means unavailable, not zero. A dash means no applicable value or timestamp was supplied. An explicit zero is a reported zero.",
                ),
                (
                    "units / counters",
                    "ms, s, m, h, and d are time units; KiB and MiB are payload sizes. Counters follow their source's retention/reset scope and are not per-second rates.",
                ),
            ],
            Self::Keys => &[
                (
                    "1-9 / 0",
                    "Target, Overview, Executors, Executor, Scheduler, Jobs, DAGs, Throughput, Observe / LLM Dispatcher.",
                ),
                (
                    "Tab / Shift-Tab",
                    "Change the focused pane. On LLM Dispatcher, switch Pools / Requests / Fabric / Scheduler.",
                ),
                (
                    "P / R / F / S on LLM Dispatcher",
                    "Jump to Pools / Requests / Fabric / Scheduler. Use Shift with the letter; lowercase p pauses and r reconnects.",
                ),
                (
                    "j/k / arrows / Enter",
                    "Move the selection / load or describe the selected row.",
                ),
                (
                    "? / Esc",
                    "Open help for the current dispatcher tab / close an overlay or return one screen.",
                ),
                (
                    "p / r",
                    "Pause operational updates / refresh. On LLM Dispatcher, r reconnects the event stream.",
                ),
                (
                    "/ / c on Requests",
                    "Search request metadata / cycle the pool filter.",
                ),
                (
                    "a on Target / s",
                    "Add a gateway / open source details on Overview, Executor, or Scheduler.",
                ),
                ("x on Executors", "Toggle executor groups and replicas."),
                (
                    "x on Execution history",
                    "Export the selected event's job diagnostics to a Markdown file. Use --export-dir PATH to choose where reports are saved.",
                ),
                (
                    "f/a/c/s on work lists",
                    "Cycle state, attention, queue, and sort filters.",
                ),
                (
                    "/ on Jobs or DAGs / n,p",
                    "Search the list / move to the next or previous server page where paging is supported.",
                ),
                (
                    "e / t in job and DAG views",
                    "Open worker execution history from a detail screen / toggle the DAG timeline and job table.",
                ),
                (
                    "w on Throughput / e,f,a,h on Observe",
                    "Cycle the throughput window / select events, flow, attempts, or health.",
                ),
                ("q / Ctrl-C", "Quit m3top."),
            ],
        }
    }
}
