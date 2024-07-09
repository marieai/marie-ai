from __future__ import annotations

from pathlib import Path
from typing import Union

from rich.text import TextType
from textual.widgets import Label, TabbedContent, TabPane


class DataCatalog(TabbedContent, can_focus=True):
    BORDER_TITLE = "Data Catalog"

    def __init__(
        self,
        *titles: TextType,
        initial: str = "",
        name: Union[str, None] = None,
        id: Union[str, None] = None,  # noqa: A002
        classes: Union[str, None] = None,
        disabled: bool = False,
        show_files: Path | None = None,
        show_s3: str | None = None,
        type_color: str = "#888888",
    ):
        super().__init__(
            *titles,
            initial=initial,
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
        )
        self.show_files = show_files
        self.show_s3 = show_s3
        self.type_color = type_color

    def on_mount(self) -> None:
        # create dummy child components
        self.add_pane(TabPane("Endpoints", Label("No data available-Nodes")))
        self.add_pane(TabPane("Jobs", Label("No data available")))
