from typing import TYPE_CHECKING, Literal

from textual.message import Message

if TYPE_CHECKING:
    from textual.widget import Widget


class WidgetMounted(Message):
    def __init__(self, widget: "Widget") -> None:
        super().__init__()
        self.widget = widget


class EtcdConnectionChange(Message):
    def __init__(
        self, connection: str, status: Literal["connected", "disconnected", "pending"]
    ) -> None:
        super().__init__()
        self.connection = connection
        self.status = status


class EtcdChangeEvent(Message):
    def __init__(self, service: str, event: str) -> None:
        super().__init__()
        self.service = service
        self.event = event
