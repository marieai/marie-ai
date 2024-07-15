from argparse import Namespace

from rich.theme import Theme
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.dom import DOMNode
from textual.lazy import Lazy
from textual.reactive import reactive
from textual.widgets import Footer

from marie_server.ctl.data_catalog import DataCatalog
from marie_server.ctl.editor_collection import EditorCollection
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
            action="help",
            description="Show help screen",
            key_display="?",
        ),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__()
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


def watch_sever_deployments(args: "Namespace"):
    print(f"Watching server deployments...  {args}")
    app = MarieApp()
    app.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Watch server deployments")
    args = parser.parse_args()
    watch_sever_deployments(args)
