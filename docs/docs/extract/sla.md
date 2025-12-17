---
sidebar_position: 7
---

# Marie Scheduler: SLA-Based Job Priority Design

## Overview

This document outlines the SLA-aware job priority system implemented in the Marie Scheduler. It provides a robust, scalable, and transparent approach for prioritizing jobs based on SLA targets using a 15-minute interval bucket model.

---

## SLA Fields

The `job` table includes several fields that support SLA-based prioritization:

* **`soft_sla`**: The preferred time by which the job should complete. This is a best-effort target.
* **`hard_sla`**: The absolute deadline the job must complete by. Missing this does not cancel the job, but it significantly increases its urgency.
* **`sla_interval`**: An optional field representing the allowable duration after `start_after` to finish the job.
* **`duration`**: Tracks how long the job has been running. This is updated periodically.
* **`priority`**: A dynamically calculated integer that reflects job urgency. Lower values indicate higher urgency.

---

## Priority Strategy

Job priority is calculated in real-time based on the current time relative to the job's SLA deadlines, measured in 15-minute buckets:

* Jobs that have missed their `hard_sla` receive the highest urgency.
* Jobs that have missed their `soft_sla` are considered urgent, but less so than hard SLA breaches.
* Jobs with an upcoming `soft_sla` are assigned moderate urgency, increasing as the deadline approaches.
* Jobs without SLA data are deprioritized.

### 15-Minute Bucket Model

* Every 15 minutes past a deadline decreases the priority score.
* Every 15 minutes before a deadline increases the score, capped to maintain reasonable urgency windows.
* This design allows for fine-grained control and fair scheduling pressure.

---

## Refreshing Priorities

A background process is responsible for periodically recalculating job priorities. This is essential to:

* Promote overdue jobs automatically
* Adjust scheduling urgency as SLA deadlines approach
* Enable SLA-aware job selection in the scheduler

The function responsible for this uses timestamp arithmetic to compute priority values in real time. It updates all jobs in `created` or `retry` state.

---

## Future Enhancements

* SLA aggregation at the DAG level
* SLA classification tags such as `missed_hard`, `missed_soft`, `sla_pending`, `no_sla`
* Metrics and alerts for SLA pressure and violations
* Escalation workflows for auto-retries, dead-letter queues, or rerouting

---

## Conclusion

This SLA-aware priority system ensures that Marie Scheduler intelligently and fairly executes jobs according to contractual and operational deadlines. It is extendable, efficient, and compliant with enterprise-grade scheduling guarantees.
