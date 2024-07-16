import asyncio
import os
from argparse import Namespace
from typing import Optional

from rich.theme import Theme
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.dom import DOMNode
from textual.lazy import Lazy
from textual.reactive import reactive
from textual.widgets import Footer

from marie.excepts import RuntimeFailToStart
from marie.helper import get_or_reuse_loop
from marie.logging.logger import MarieLogger
from marie.serve.discovery.resolver import EtcdServiceResolver
from marie_server.ctl.data_catalog import DataCatalog
from marie_server.ctl.editor_collection import EditorCollection
from marie_server.ctl.exception import pretty_print_error
from marie_server.ctl.help_screen import HelpScreen
from marie_server.ctl.messages import EtcdConnected
from marie_server.ctl.model import Config
from marie_server.ctl.result_viewer import ResultsViewer
from marie_server.ctl.run_query_bar import RunQueryBar
from marie_server.ctl.topbar import TopBar


class MarieApp(App):
    TITLE = "Marie-AI"
    CSS_PATH = "marie.css"

    full_screen: reactive[bool] = reactive(False)
    sidebar_hidden: reactive[bool] = reactive(False)

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
        self.logger = MarieLogger(self.__class__.__name__)
        self.logger.info(f"Setting up Marie Ctl {self.TITLE}...")

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

    async def on_mount(self) -> None:
        self.logger.info("Mounting Marie Ctl...")
        self.setup_service_discovery_sync(
            etcd_host=self.config.etcd_host,
            etcd_port=self.config.etcd_port,
        )

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""

        self.connection_status = "Connected"
        self.app_version = "3.0.30"
        self.host = "localhost:8000"

        yield TopBar(
            connection_status=self.connection_status,
            app_version=self.app_version,
            host=self.host,
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
        description="Connecting to DB",
    )
    def setup_service_discovery_sync(
        self,
        etcd_host: Optional[str] = "0.0.0.0",
        etcd_port: Optional[int] = 2379,
        watchdog_interval: Optional[int] = 2,
    ):
        self.logger.info("Setting up service discovery")
        loop = get_or_reuse_loop()
        self.post_message(EtcdConnected(connection="Test Connection"))

    async def setup_service_discovery(
        self,
        etcd_host: Optional[str] = "0.0.0.0",
        etcd_port: Optional[int] = 2379,
        watchdog_interval: Optional[int] = 2,
    ):
        """
         Setup service discovery for the gateway.

        :param etcd_host: Optional[str] - The host address of the ETCD service. Default is "0.0.0.0".
        :param etcd_port: Optional[int] - The port of the ETCD service. Default is 2379.
        :param watchdog_interval: Optional[int] - The interval in seconds between each service address check. Default is 2.
        :return: None

        """
        self.logger.info("Setting up service discovery ")
        # FIXME : This is a temporary solution to test the service discovery
        service_name = "gateway/service_test"

        async def _start_watcher():
            resolver = EtcdServiceResolver(
                etcd_host,
                etcd_port,
                namespace="marie",
                start_listener=False,
                listen_timeout=5,
            )

            self.logger.info(f"checking : {resolver.resolve(service_name)}")
            resolver.watch_service(service_name, self.handle_discovery_event)

        task = asyncio.create_task(_start_watcher())
        try:
            await task  # This raises an exception if the task had an exception
        except Exception as e:
            self.logger.error(
                f"Initialize etcd client failed failed on {etcd_host}:{etcd_port}"
            )
            if isinstance(e, RuntimeFailToStart):
                raise e
            raise RuntimeFailToStart(
                f"Initialize etcd client failed failed on {etcd_host}:{etcd_port}, ensure the etcd server is running."
            )

    def handle_discovery_event(self, service: str, event: str) -> None:
        """
        Enqueue the event to be processed.
        :param service: The name of the service that is available.
        :param event: The event that triggered the method.
        :return:
        """

        self._loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(self.event_queue.put((service, event)))
        )


def watch_sever_deployments(args: "Namespace"):
    print(f"Watching server deployments...  {args}")
    # ensure that proper environment variables are set to prevent " Could not load the Qt platform plugin "xcb"
    # in  even when running in headless mode
    # export QT_QPA_PLATFORM=offscreen
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    config = Config()
    config.etcd_host = args.etcd_host
    config.etcd_port = args.etcd_port

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

    # start the server in development mode
    # https://textual.textualize.io/guide/devtools/
    # textual run --dev watch.py
    # marie server watch --etcd-host 127.0.0.1 --etcd-port 2379

    args = parser.parse_args()
    watch_sever_deployments(args)
