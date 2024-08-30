from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

from rich.markup import escape
from rich.text import TextType
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Label, Placeholder
from textual_fastdatatable import DataTable
from textual_fastdatatable.backend import AutoBackendType

from marie_server.ctl.messages import WidgetMounted


class ResultsTable(DataTable, inherit_bindings=False):
    DEFAULT_CSS = """
        ResultsTable {
            height: 75%;
            width: 100%;
        }
    """

    def on_mount(self) -> None:
        self.post_message(WidgetMounted(widget=self))


class SeparatorBar(Horizontal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_mount(self) -> None:
        self.add_class("SeparatorBar")


class NodeInfoWidget(Vertical, can_focus=True):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.connection_status = "Connected"
        self.app_version = "3.0.30"
        self.host = "localhost:8000"

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Host: localhost:8000")
            yield Label("Connection Status: Connected")
            yield Label("App Version: 3.0.30")
            yield Label("GPU Status: [red]Not Available[/]")
            yield Label("TPU Status: [red]Not Available[/]")


class PodInfoPanel(Vertical, inherit_bindings=False):
    BORDER_TITLE = "DeploymentsXXX"

    DEFAULT_CSS = """
        NodeInfoPanel {
            height: 25%;
            width: 100%;
        }
    """

    def on_mount(self) -> None:
        # pane_info = TabPane(
        #     f"Pod Info",
        #     NodeInfoWidget(),
        #     id="tab_pod_info",
        # )
        #
        # pane_executors = TabPane(
        #     f"Executors",
        #     Label("Executors: 0"),
        #     id="tab_executors",
        # )
        #
        # self.add_pane(pane_info)
        # self.add_pane(pane_executors)
        pass

    def compose(self) -> ComposeResult:
        with Vertical():
            yield SeparatorBar()

            with Vertical():
                yield Label("Host: localhost:8000")
                yield Label("Connection Status: Connected")
                yield Label("App Version: 3.0.30")
                yield Label("GPU Status: [red]Not Available[/]")
                yield Label("TPU Status: [red]Not Available[/]")

            with Vertical():
                yield Label("")
                yield Label("Executors: 0")
                yield Label("executor: extract")
                yield Label("executor: __dry_run__")


class DataCatalog(Vertical, can_focus=True):
    BORDER_TITLE = "Deployments"

    def __init__(
        self,
        *titles: TextType,
        initial: str = "",
        name: Union[str, None] = None,
        id: Union[str, None] = None,  # noqa: A002
        classes: Union[str, None] = None,
        disabled: bool = False,
        show_files: Path | None = None,
        type_color: str = "#888888",
    ):
        super().__init__(
            *titles,
            # initial=initial,
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
        )
        self.show_files = show_files
        self.type_color = type_color

        self.max_results = 1000
        self.max_col_width = 250

    def on_mount(self) -> None:
        self.max_col_width = self._get_max_col_width()

    def build_table(
        self,
        table_id: str,
        column_labels: List[Tuple[str, str]],
        data: AutoBackendType,
    ) -> ResultsTable:
        formatted_labels = [
            self._format_column_label(col_name, col_type)
            for col_name, col_type in column_labels
        ]
        table = ResultsTable(
            id=table_id,
            column_labels=formatted_labels,  # type: ignore
            data=data,
            max_rows=self.max_results,
            cursor_type="range",
            max_column_content_width=self.max_col_width,
            null_rep="[dim]∅ null[/]",
        )

        # need to manually refresh the table, since activating the tab
        # doesn't consistently cause a new layout calc.
        table.refresh(repaint=True, layout=True)
        return table

    def _format_column_label(self, col_name: str, col_type: str) -> str:
        return f"{escape(col_name)}"
        # return f"{escape(col_name)} [{self.type_color}]{escape(col_type)}[/]"

    def build_top_component(self):
        # Build your top component here
        # This is just a placeholder. Replace it with your actual component creation logic.
        return Placeholder(name="Top Component")

    def compose(self) -> ComposeResult:
        yield self.build_table(
            "nodes_table",
            [
                ("Online", "bool"),
                ("Host", "str"),
                ("Endpoints", "int"),
            ],
            [
                (True, "grpc://0.0.0.0:50517", 1),
                (True, "grpc://0.0.0.0:58264", 3),
                (True, "grpc://0.0.0.0:58264", 3),
                (False, "grpc://0.0.0.0:58264", 3),
            ],
        )

        panel = PodInfoPanel()
        yield panel

    def on_resize(self) -> None:
        # only impacts new tables pushed after the resize
        self.max_col_width = self._get_max_col_width()

    def _get_max_col_width(self) -> int:
        SMALLEST_MAX_WIDTH = 20
        CELL_X_PADDING = 2
        parent_size = getattr(self.parent, "container_size", self.screen.container_size)
        return max(SMALLEST_MAX_WIDTH, parent_size.width // 2 - CELL_X_PADDING)
