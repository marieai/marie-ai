from textual.app import ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Label


class TopBar(Container):
    connection_status = reactive("disconnected", init=False, always_update=True)
    host = reactive("0.0.0.0", init=False, always_update=True)

    def __init__(
        self,
        connection_status: str = "disconnected",
        host="0.0.0.0",
        app_version="",
        help="press [b highlight]q[/b highlight] to return",
    ):
        super().__init__()
        self.topbar_title = Label(
            f" [b light_blue]Marie AI[/b light_blue] :fox: [light_blue]v{app_version}",
            id="topbar_title",
        )
        self.topbar_host = Label(
            f"[[white]{self.connection_status}[/white]] {self.host}", id="topbar_host"
        )
        self.topbar_help = Label(help, id="topbar_help")

        self.connection_status = connection_status
        self.host = host if host is not None else ""

    def watch_connection_status(self) -> None:
        if self.connection_status:
            self.topbar_host.update(
                f"[[white]{self.connection_status}[/white]] {self.host}"
            )
        else:
            self.topbar_host.update("")

    def compose(self) -> ComposeResult:
        yield self.topbar_title
        yield self.topbar_host
        yield self.topbar_help
