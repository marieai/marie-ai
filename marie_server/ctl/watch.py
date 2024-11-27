import asyncio
import os
import time
from argparse import Namespace

from rich.theme import Theme
from textual import log, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.dom import DOMNode
from textual.lazy import Lazy
from textual.reactive import reactive
from textual.widgets import Footer

from marie.excepts import BadConfigSource
from marie.helper import get_or_reuse_loop
from marie.serve.discovery.resolver import EtcdServiceResolver
from marie_server.ctl.data_catalog import DataCatalog
from marie_server.ctl.editor_collection import EditorCollection
from marie_server.ctl.help_screen import HelpScreen
from marie_server.ctl.messages import EtcdChangeEvent, EtcdConnectionChange
from marie_server.ctl.model import Config
from marie_server.ctl.result_viewer import ResultsViewer
from marie_server.ctl.run_query_bar import RunQueryBar
from marie_server.ctl.topbar import TopBar


class MarieApp(App):
    TITLE = "Marie-AI"
    CSS_PATH = "marie.css"

    full_screen: reactive[bool] = reactive(False)
    sidebar_hidden: reactive[bool] = reactive(False)

    connection_status: reactive[str] = reactive("Establishing Connection...")
    host: reactive[str] = reactive("localhost:ZZZ")

    BINDINGS = [
        Binding(key="q", action="quit", description="Quit"),
        Binding(
            key="question_mark",
            action="show_help_screen",
            description="Show help screen",
            key_display="?",
        ),
    ]

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__()
        log(f"Setting up Marie Ctl {self.TITLE}...")

        theme = Theme(
            {
                "white": "#e9e9e9",
                "green": "#54efae",
                "yellow": "#f6ff8f",
                "dark_yellow": "#cad45f",
                "red": "#fd8383",
                "purple": "#b565f3",
                "dark_gray": "#969aad",
                "highlight": "#91abec",
                "label": "#c5c7d2",
                "b label": "b #c5c7d2",
                "light_blue": "#bbc8e8",
                "b white": "b #e9e9e9",
                "b highlight": "b #91abec",
                "bold red": "b #fd8383",
                "b light_blue": "b #bbc8e8",
                "panel_border": "#6171a6",
                "table_border": "#333f62",
            }
        )

        self.console.push_theme(theme)
        self.console.set_window_title(self.TITLE)
        self.config = config
        self.event_queue = asyncio.Queue()

        self.app_version = "0.0.0"
        self.connection = EtcdConnectionChange(
            f"{self.config.etcd_host}:{self.config.etcd_port}/{self.config.service_name}",
            "disconnected",
        )
        # self.host = "localhost:0000"

    async def on_mount(self) -> None:
        log("Mounting Marie Ctl...")

        self.setup_service_discovery_sync(
            etcd_host=self.config.etcd_host,
            etcd_port=self.config.etcd_port,
            service_name=self.config.service_name,
        )

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        self.top_bar = TopBar(
            connection_status="disconnected",
            host=self.connection.connection,
            app_version=self.app_version,
        )

        self.data_catalog = DataCatalog()
        self.editor_collection = EditorCollection(classes="hide-tabs")
        self.editor: None = None
        editor_placeholder = Lazy(widget=self.editor_collection)
        editor_placeholder.border_title = self.editor_collection.border_title
        editor_placeholder.loading = True

        self.max_results = 10_000
        self.results_viewer = ResultsViewer(
            max_results=self.max_results,
        )

        self.run_query_bar = RunQueryBar(
            max_results=self.max_results, classes="non-responsive"
        )
        self.footer = Footer()
        # lay out the widgets
        yield self.top_bar
        with Horizontal():
            yield self.data_catalog
            with Vertical(id="main_panel"):
                yield editor_placeholder
                yield self.run_query_bar
                yield self.results_viewer
        yield self.footer

    # @on(DatabaseConnected)
    # def initialize_app(self, message: DatabaseConnected) -> None:
    #     self.connection = message.connection
    #     self.post_message(
    #         TransactionModeChanged(new_mode=message.connection.transaction_mode)
    #     )
    #     self.run_query_bar.set_responsive()
    #     self.results_viewer.show_table(did_run=False)
    #     if message.connection.init_message:
    #         self.notify(message.connection.init_message, title="Database Connected.")
    #     else:
    #         self.notify("Database Connected.")
    #     self.update_schema_data()
    #

    @on(EtcdConnectionChange)
    def initialize_app(self, message: EtcdConnectionChange) -> None:
        connection = message.connection
        status = message.status

        self.notify(f"status={status}, connection= {connection}", title="ETCD")
        self.app_version = "0.0.0"

        self.top_bar.host = connection
        self.top_bar.connection_status = status

    @on(EtcdChangeEvent)
    def handle_change_event(self, message: EtcdChangeEvent) -> None:
        self.notify(f"Event : {message.event}", title="ETCD Change Event")

    @property
    def namespace_bindings(self) -> dict[str, tuple[DOMNode, Binding]]:
        """
        Re-order bindings so they appear in the footer with the global bindings first.
        """

        def sort_key(item: tuple[str, tuple[DOMNode, Binding]]) -> int:
            return 0 if item[1][0] == self else 1

        binding_map = {
            k: v for k, v in sorted(super().namespace_bindings.items(), key=sort_key)
        }
        return binding_map

    def action_show_help_screen(self) -> None:
        self.push_screen(HelpScreen(id="help_screen"))

    @work(
        thread=True,
        exclusive=True,
        exit_on_error=False,
        group="connect",
        description="Connecting to ETCD",
    )
    def setup_service_discovery_sync(
        self,
        etcd_host: str,
        etcd_port: int,
        service_name: str,
        watchdog_interval: int = 2,
    ):
        log("Setting up service discovery")
        loop = get_or_reuse_loop()
        time.sleep(1)

        # run setup_service_discovery in the event loop to avoid blocking the main thread
        loop.run_until_complete(
            self.setup_service_discovery(
                etcd_host=etcd_host,
                etcd_port=etcd_port,
                service_name=service_name,
                watchdog_interval=watchdog_interval,
            )
        )

    async def setup_service_discovery(
        self,
        etcd_host: str,
        etcd_port: int,
        service_name: str,
        watchdog_interval: int = 2,
    ):
        """
        Setup service discovery for the gateway.
        :param etcd_host: Optional[str] - The host address of the ETCD service. Default is "0.0.0.0".
        :param etcd_port: Optional[int] - The port of the ETCD service. Default is 2379.
        :param service_name: Optional[str] - The name of the service to discover. Default is "gateway/marie".
        :param watchdog_interval: Optional[int] - The interval in seconds between each service address check. Default is 2.
        :return: None

        """
        log(f"Setting up service discovery : {service_name}")
        log(f"ETCD host : {etcd_host}:{etcd_port}")

        async def _start_watcher():
            if not service_name:
                raise BadConfigSource(
                    "Service name must be provided for service discovery"
                )

            resolver = EtcdServiceResolver(
                etcd_host,
                etcd_port,
                namespace="marie",
                start_listener=False,
                listen_timeout=watchdog_interval,
            )

            self.post_message(
                EtcdConnectionChange(
                    f"{etcd_host}:{etcd_port}/{service_name}", "connected"
                )
            )
            resolver.watch_service(
                service_name,
                lambda service, event: self.post_message(
                    EtcdChangeEvent(service, event)
                ),
            )

        try:
            task = asyncio.create_task(_start_watcher())
            await task  # This raises an exception if the task had an exception
        except Exception as e:
            msg = f"Initialize etcd client failed failed on {etcd_host}:{etcd_port} : error {e}"
            log(msg)
            self.notify(msg, title="Error", severity="error")

    def handle_discovery_event(self, service: str, event: str) -> None:
        """
        Enqueue the event to be processed.
        :param service: The name of the service that is available.
        :param event: The event that triggered the method.
        :return:
        """
        self.notify(f"Service {service} : {event}", title="Service Discovery")


def watch_sever_deployments(args: "Namespace"):
    print(f"Watching server deployments...  {args}")
    # ensure that proper environment variables are set to prevent " Could not load the Qt platform plugin "xcb"
    # in  even when running in headless mode
    # export QT_QPA_PLATFORM=offscreen
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    config = Config()
    config.etcd_host = args.etcd_host
    config.etcd_port = args.etcd_port
    config.service_name = args.service_name

    app = MarieApp(config)
    app.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Watch server deployments")
    parser.add_argument(
        "--etcd-host",
        default="127.0.0.1",
        type=str,
        help="The host address of etcd server to watch",
        metavar="",
    )
    parser.add_argument(
        "--etcd-port",
        dest="etcd_port",
        default=2379,
        type=int,
        help="The port of etcd server to watch",
        metavar="",
    )
    parser.add_argument(
        "--service_name",
        dest="service_name",
        help="The name of the service to watch",
        default="gateway/marie",
    )

    # start the server in development mode
    # https://textual.textualize.io/guide/devtools/
    # textual run --dev watch.py
    # marie server watch --etcd-host 127.0.0.1 --etcd-port 2379

    args = parser.parse_args()
    watch_sever_deployments(args)
