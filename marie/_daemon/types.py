from typing import NamedTuple, Optional, Sequence

import marie.check as check
from marie.utils.error import SerializableErrorInfo


class DaemonHeartbeat(
    NamedTuple(
        "_DaemonHeartbeat",
        [
            ("timestamp", float),
            ("daemon_type", str),
            ("daemon_id", Optional[str]),
            ("errors", Optional[Sequence[SerializableErrorInfo]]),
        ],
    ),
):
    def __new__(
        cls,
        timestamp: float,
        daemon_type: str,
        daemon_id: Optional[str],
        errors: Optional[Sequence[SerializableErrorInfo]] = None,
    ):
        errors = check.opt_sequence_param(
            errors, "errors", of_type=SerializableErrorInfo
        )

        return super(DaemonHeartbeat, cls).__new__(
            cls,
            timestamp=check.float_param(timestamp, "timestamp"),
            daemon_type=check.str_param(daemon_type, "daemon_type"),
            daemon_id=check.opt_str_param(daemon_id, "daemon_id"),
            errors=errors,
        )


class DaemonStatus(
    NamedTuple(
        "_DaemonStatus",
        [
            ("daemon_type", str),
            ("required", bool),
            ("healthy", Optional[bool]),
            ("last_heartbeat", Optional[DaemonHeartbeat]),
        ],
    )
):
    """Daemon statuses are derived from daemon heartbeats and instance configuration to provide an
    overview about the daemon's liveness.
    """

    def __new__(
        cls,
        daemon_type: str,
        required: bool,
        healthy: Optional[bool],
        last_heartbeat: Optional[DaemonHeartbeat],
    ):
        return super(DaemonStatus, cls).__new__(
            cls,
            daemon_type=check.str_param(daemon_type, "daemon_type"),
            required=check.bool_param(required, "required"),
            healthy=check.opt_bool_param(healthy, "healthy"),
            last_heartbeat=check.opt_inst_param(
                last_heartbeat, "last_heartbeat", DaemonHeartbeat
            ),
        )
