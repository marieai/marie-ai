from __future__ import annotations

from typing import Union

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Button


class RunQueryBar(Horizontal):
    def __init__(
        self,
        *children: Widget,
        name: Union[str, None] = None,
        id: Union[str, None] = None,  # noqa
        classes: Union[str, None] = None,
        disabled: bool = False,
        max_results: int = 10_000,
    ) -> None:
        self.max_results = max_results
        super().__init__(
            *children, name=name, id=id, classes=classes, disabled=disabled
        )

    def compose(self) -> ComposeResult:
        self.run_button = Button(
            "Events",
            id="run_events",
        )
        self.log_button = Button("Logs", id="button_log")

        with Horizontal(id="run_buttons"):
            yield self.run_button
            yield self.log_button

    def on_mount(self) -> None:
        pass

    def set_not_responsive(self) -> None:
        self.add_class("non-responsive")

    def set_responsive(self) -> None:
        self.remove_class("non-responsive")
