from enum import Enum


class WorkState(Enum):
    """Represents the state of a work item in a system.

    The WorkState class is an enumeration that defines the possible states
    that a work item can be in. Each state is represented by a unique string
    value.

    Attributes:
        CREATED (str): Represents the state of a work item that has been created.
        RETRY (str): Represents the state of a work item that is pending retry.
        ACTIVE (str): Represents the state of a work item that is active and being processed.
        COMPLETED (str): Represents the state of a work item that has been successfully completed.
        EXPIRED (str): Represents the state of a work item that has expired and cannot be processed further.
        CANCELLED (str): Represents the state of a work item that has been cancelled.
        FAILED (str): Represents the state of a work item that has failed to complete.

    """

    CREATED = "created"
    RETRY = "retry"
    ACTIVE = "active"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    FAILED = "failed"

    def is_terminal(self) -> bool:
        """Return whether or not this status is terminal.

        A terminal status is one that cannot transition to any other status.

        Returns:
            True if this status is terminal, otherwise False.
        """
        return self in [
            WorkState.COMPLETED,
            WorkState.EXPIRED,
            WorkState.CANCELLED,
            WorkState.FAILED,
        ]

    @staticmethod
    def terminal_states() -> list['WorkState']:
        """Return a list of terminal states."""
        return [
            WorkState.COMPLETED,
            WorkState.EXPIRED,
            WorkState.CANCELLED,
            WorkState.FAILED,
        ]
