"""
The Hubble Python Client
"""
import asyncio
import datetime as _datetime
import os as _os
import sys as _sys
from functools import wraps
from typing import Optional

from importlib_metadata import version

from .client.client import Client  # noqa F401
from .excepts import AuthenticationRequiredError
from .utils.auth import Auth  # noqa F401
from .utils.notebook import is_notebook  # noqa F401

try:
    __version__ = version("jina-hubble-sdk")
except Exception:
    __version__ = "v0.0.0"


__windows__ = _sys.platform == 'win32'
__uptime__ = _datetime.datetime.now().isoformat()


def login_required(func):
    """Annotate a function so that it requires login to Jina AI to run.

    Example:

    .. highlight:: python
    .. code-block:: python

        @login_required
        def foo():
            print(1)

    :param levels: required build level to run this function.
    :return: annotated function
    """

    @wraps(func)
    def arg_wrapper(*args, **kwargs):
        try:
            Client(jsonify=True).get_user_info()
            return func(*args, **kwargs)
        except AuthenticationRequiredError:
            import sys

            if sys.__stdin__.isatty():
                from rich import print
                from rich.prompt import Confirm

                print(
                    ':closed_lock_with_key: [yellow bold]You are not logged in to Jina AI. [/]'
                    f'[yellow bold]However, {func!r} requires login to Jina AI[/].'
                )
                if Confirm.ask('Do you want to login now?'):
                    login()
                    return func(*args, **kwargs)
                print(
                    '[yellow]You can login later by running [bold]jina auth login[/] '
                    'or set env variable [bold]JINA_AUTH_TOKEN[/][/].'
                )

            raise AuthenticationRequiredError(
                response={},
                message=f'Jina auth token is not provided or has expired. {func!r} requires login to Jina AI, '
                f'please run `jina auth login -f` or set env variable `JINA_AUTH_TOKEN` with correct token',
            ) from None

    return arg_wrapper


def login(interactive: Optional[bool] = None, **kwargs):
    """This function guides user to login."""

    if interactive is None:
        interactive = is_notebook()

    if interactive:
        Auth.login_notebook(**kwargs)
    else:
        Auth.login_sync(**kwargs)


def notebook_login(**kwargs):
    """This function guides user to log-in via P.A.T. or via browser."""
    Auth.login_notebook(**kwargs)


def logout():
    """Logout."""
    asyncio.run(Auth.logout())


def is_logged_in():
    """Check if user is logged in."""
    return True if Client(jsonify=True).token else False


def get_token(interactive: bool = False) -> Optional[str]:
    """Get token."""
    if _os.environ.get('SHOW_HUBBLE_HINT') == 'ALWAYS':
        token = show_hint(interactive)
    elif _os.environ.get('SHOW_HUBBLE_HINT', 'ONCE') == 'ONCE':
        token = show_hint(interactive)
        _os.environ['SHOW_HUBBLE_HINT'] = 'NEVER'
    else:
        token = Client(jsonify=True).token

    return token


def show_hint(interactive: bool = False) -> Optional[str]:  # noqa: E501
    """
    Show hint if the user is not logged in.

    """
    from rich import print

    try:
        c = Client(jsonify=True)

        user = c.get_user_info(log_error=False).get('data', {})
        name = user.get('nickname') or user.get('name')
        username = user.get('name')

        print(
            f':closed_lock_with_key: [green bold]You are logged in to Jina AI[/] '
            f'as [bold]{name} (username: {username})[/]. '
            f'To log out, use [dim]jina auth logout[/].'
        )
        return c.token
    except AuthenticationRequiredError:
        print(
            ':closed_lock_with_key: [yellow bold]You are not logged in to Jina AI[/]. '
            'To log in, use [bold]jina auth login[/] or set env variable [bold]JINA_AUTH_TOKEN[/].'
        )
        if interactive:
            from rich.prompt import Confirm

            if Confirm.ask('Do you want to login now?'):
                login()

            return show_hint(interactive=interactive)
    except Exception as ex:
        print(f'Unknown error: {ex}')
