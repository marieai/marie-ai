from __future__ import annotations

from typing import Any, Union

from rich.text import TextType
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import ContentSwitcher, Label, TabbedContent, TabPane, Tabs
from textual_textarea import TextEditor

from marie_server.ctl.messages import WidgetMounted


class CodeEditor(TextEditor, inherit_bindings=False):
    class Submitted(Message, bubble=True):
        """Posted when user runs the query.

        Attributes:
            lines: The lines of code being submitted.
            cursor: The position of the cursor
        """

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text


class EditorCollection(TabbedContent, can_focus=True):
    BORDER_TITLE = "Editor Collection"

    def __init__(
        self,
        *titles: TextType,
        initial: str = "",
        name: Union[str, None] = None,
        id: Union[str, None] = None,  # noqa: A002
        classes: Union[str, None] = None,
        disabled: bool = False,
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
        self.type_color = type_color
        self.counter = 0

    @property
    def current_editor(self) -> CodeEditor:
        content = self.query_one(ContentSwitcher)
        active_tab_id = self.active
        if active_tab_id:
            try:
                tab_pane = content.query_one(f"#{active_tab_id}", TabPane)
                return tab_pane.query_one(CodeEditor)
            except NoMatches:
                pass
        all_editors = content.query(CodeEditor)
        return all_editors.first(CodeEditor)

    async def on_mount(self) -> None:
        cache = load_cache()
        if cache is not None:
            for _i, buffer in enumerate(cache.buffers):
                await self.action_new_buffer(state=buffer)
                # we can't load the focus state here, since Tabs
                # really wants to activate the first tab when it's
                # mounted
        else:
            await self.action_new_buffer()
        self.query_one(Tabs).can_focus = False
        # self.current_editor.word_completer = self.word_completer
        # self.current_editor.member_completer = self.member_completer
        self.post_message(WidgetMounted(widget=self))

    async def action_new_buffer(self, state: Union[Any, None] = None) -> None:
        self.counter += 1
        new_tab_id = f"tab-{self.counter}"
        editor = CodeEditor()

        pane = TabPane(
            f"Tab {self.counter}",
            editor,
            id=new_tab_id,
        )
        await self.add_pane(pane)
        if state is not None:
            editor.selection = state.selection
        else:
            self.active = new_tab_id
            try:
                self.current_editor.focus()
            except NoMatches:
                pass
        if self.counter > 1:
            self.remove_class("hide-tabs")
        return editor


def load_cache():
    pass
