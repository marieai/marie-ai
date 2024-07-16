from rich.panel import Panel

from marie.excepts import BaseMarieException


def pretty_print_error(error: BaseMarieException) -> None:
    from rich import print

    print(pretty_error_message(error))


def pretty_error_message(error: BaseMarieException) -> Panel:
    title = hasattr(error, "title") and error.title
    title = title if title else "Marie encountered an error."
    return Panel.fit(
        str(error),
        title=title,
        title_align="left",
        border_style="red",
    )


def pretty_print_warning(title: str, message: str) -> None:
    from rich import print

    from .colors import GREEN

    print(
        Panel.fit(
            message,
            title=title,
            title_align="left",
            border_style=GREEN,
        )
    )
