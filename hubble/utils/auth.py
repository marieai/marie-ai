import json
import os
import webbrowser
from typing import Optional
from urllib.parse import urlencode, urljoin

import aiohttp
import requests
from hubble.client.session import HubbleAPISession
from hubble.excepts import AuthenticationFailedError
from hubble.utils.api_utils import get_base_url, get_json_from_response
from hubble.utils.config import config
from rich import print as rich_print

JINA_LOGO = (
    'https://d2vchdhjlcm3i6.cloudfront.net/Company+Logo/Light/Company+logo_light.svg'
)

NOTEBOOK_LOGIN_HTML = f"""
<div class='custom-container'>
    <style>
        .button1 {{
            color: white;
            background-color: #009191;
            border: 1px solid #009191;
        }}
        .button2 {{
            color: #009191;
            background-color: white;
            border: 1px solid #009191;
        }}
        .link1 {{
            color:#009191 !important;
            position: relative;
            top: 32px;
            right: -120px;
            z-index: 99;
        }}
        .custom-container {{
            margin-top: 10px;
            margin-bottom: -10px;
        }}
        .spaced {{
            margin: 20px 0;
        }}
    </style>
    <center>
        <img src={JINA_LOGO} width=175 alt='Jina AI'>
        <div class='spaced'></div>
        <p>
            Copy a <b>Personal Access Token</b>, paste it below, and press the <b>Token login</b> button.
            <br>
            If you don't have a token, press the <b>Browser login</b> button to log in via the browser.
        </p>
        <a
            href='https://cloud.jina.ai/settings/tokens'
            target='__blank'
            class='link1'>
                Create
        </a>
    </center>
</div>
"""

NOTEBOOK_SUCCESS_HTML = f"""
<div class='custom-container'>
    <style>
        .custom-container {{
            margin-top: 10px;
            margin-bottom: 0;
        }}
        .spaced {{
            margin: 20px 0;
        }}
    </style>
    <center>
        <img src={JINA_LOGO} width=175 alt='Jina AI'>
        <div class='spaced'></div>
        <p>
            You are logged in to Jina AI!
        </p>
        <p>
            To log in again, run <code>login(force=True)</code>.
        </p>
    </center>
</div>
"""

NOTEBOOK_ERROR_HTML = """
<div class='custom-container'>
    <style>
        .custom-container {{
            margin-top: 10px;
            margin-bottom: 0;
        }}
        .spaced {{
            margin: 20px 0;
        }}
        .error {{
            text-align: left !important;
            background-color: WhiteSmoke;
            margin: 10px 0 !important;
            padding: 10px 50px 10px 20px;
            line-height: 16px;
        }}
        .red {{
            color: #d03c38;
        }}
    </style>
    <center>
        <img src={LOGO} width=175 alt='Jina AI'>
        <div class='spaced'></div>
        <p class='red'>
            An error occured, see the details below.
        </p>
        <div class='error'>
            <pre><code>{ERR}</code></pre>
        </div>
    </center>
</div>
"""

NOTEBOOK_REDIRECT_HTML = """
<div class='custom-container'>
    <style>
        .custom-container {{
            margin-top: 10px;
            margin-bottom: 0;
        }}
        .spaced {{
            margin: 20px 0;
        }}
        .link2 {{
            color:#009191 !important;
        }}
    </style>
    <center>
        <img src={LOGO} width=175 alt="Jina AI">
        <div class='spaced'></div>
        <p>
            Your browser is going to open the login page.
        </p>
        <p>
            If this fails, please open <a class='link2' href='{HREF}' target='_blank'>this link</a> to log in.
        </p>
    </center>
</div>
"""


class Auth:
    @staticmethod
    def get_auth_token_from_config():
        """Get user auth token from config file."""
        token_from_config: Optional[str] = None
        if isinstance(config.get('auth_token'), str):
            token_from_config = config.get('auth_token')

        return token_from_config

    @staticmethod
    def get_auth_token():
        """Get user auth token.

        .. note:: We first check `JINA_AUTH_TOKEN` environment variable.
          if token is not None, use env token. Otherwise, we get token from config.
        """
        token_from_env = os.environ.get('JINA_AUTH_TOKEN')

        token_from_config: Optional[str] = Auth.get_auth_token_from_config()

        return token_from_env if token_from_env else token_from_config

    @staticmethod
    def validate_token(token):
        try:
            session = HubbleAPISession()
            session.init_jwt_auth(token)
            resp = session.validate_token()
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            raise AuthenticationFailedError("Could not validate token")

    @staticmethod
    def login_notebook(force: bool = False, **kwargs):
        """Login user in notebook environments like colab"""

        # trying to import utilities (only available in notebook env)
        try:
            import ipywidgets.widgets as widgets
            from IPython.display import display
        except ImportError:
            raise ImportError(
                """
The `notebook_login` function can only be used in a notebook.
The function also requires `ipywidgets`.
                """
            )

        # login widget
        token_widget = widgets.Password(
            placeholder="Personal Access Token (PAT)",
            layer=widgets.Layout(width="300px"),
        )

        token_button_widget = widgets.Button(
            description="Token login",
            disabled=True,
            layout=widgets.Layout(width="300px"),
        )

        token_button_widget.add_class('button1')

        def _handle_token_change(change):
            if change.new is not None and change.new != '':
                token_button_widget.disabled = False
            else:
                token_button_widget.disabled = True

        token_widget.observe(_handle_token_change, names='value')

        browser_button_widget = widgets.Button(
            description="Browser login",
            layout=widgets.Layout(width="300px", margin="10px 0 0 0"),
        )

        browser_button_widget.add_class('button2')

        login_widget = widgets.VBox(
            [
                widgets.HTML(NOTEBOOK_LOGIN_HTML),
                token_widget,
                token_button_widget,
                browser_button_widget,
            ],
            layout=widgets.Layout(
                display="flex", flex_flow="column", align_items="center"
            ),
        )

        # sucess widget
        success_widget = widgets.VBox(
            [widgets.HTML(NOTEBOOK_SUCCESS_HTML)],
            layout=widgets.Layout(
                display="flex", flex_flow="column", align_items="center"
            ),
        )

        # redirect url widget
        redirect_url_widget = widgets.HTML(value="")
        redirect_widget = widgets.VBox(
            [
                redirect_url_widget,
            ],
            layout=widgets.Layout(
                display="flex", flex_flow="column", align_items="center"
            ),
        )

        # error widget
        error_description_widget = widgets.HTML(value="")
        error_widget = widgets.VBox(
            [
                error_description_widget,
            ],
            layout=widgets.Layout(
                display="flex", flex_flow="column", align_items="center"
            ),
        )

        # callback functions for login_async to communicate events
        def _success_callback(**kwargs):
            login_widget.layout.display = "none"
            redirect_widget.layout.display = "none"
            error_widget.layout.display = "none"
            success_widget.layout.display = "flex"

        def _redirect_callback(href=None, **kwargs):

            # format url
            redirect_url_widget.value = NOTEBOOK_REDIRECT_HTML.format(
                LOGO=JINA_LOGO, HREF=href
            )

            login_widget.layout.display = "none"
            redirect_widget.layout.display = "flex"
            error_widget.layout.display = "none"
            success_widget.layout.display = "none"

            # attempt to open the browser
            webbrowser.open(href)

        def _error_callback(err=None, **kwargs):
            # format error
            err = json.dumps(err, indent=4)
            error_description_widget.value = NOTEBOOK_ERROR_HTML.format(
                LOGO=JINA_LOGO, ERR=err
            )

            login_widget.layout.display = "none"
            redirect_widget.layout.display = "none"
            error_widget.layout.display = "flex"
            success_widget.layout.display = "none"

        # login function called when pressing the login button
        def _login(*args):
            # reading token, clearing form, disabling elements
            token = token_widget.value
            token_widget.value = ""
            token_widget.disabled = True
            token_button_widget.disabled = True
            browser_button_widget.disabled = True

            # verify token before login function
            if token != "":
                try:
                    Auth.validate_token(token)
                    config.set('auth_token', token)
                    _success_callback()

                    post_success = kwargs.get('post_success')
                    if post_success:
                        post_success()

                    return
                except AuthenticationFailedError:
                    pass

            Auth.login_sync(
                force=force,
                success_callback=_success_callback,
                redirect_callback=_redirect_callback,
                error_callback=_error_callback,
                **kwargs,
            )

        token_button_widget.on_click(_login)
        browser_button_widget.on_click(_login)

        # verifying existing token
        token = Auth.get_auth_token()
        if token and not force:
            try:
                Auth.validate_token(token)
                display(success_widget)

                post_success = kwargs.get('post_success')
                if post_success:
                    post_success()

                return
            except AuthenticationFailedError:
                pass

        all_widget = widgets.VBox(
            [login_widget, redirect_widget, error_widget, success_widget],
            layout=widgets.Layout(display="block"),
        )

        login_widget.layout.display = "flex"
        redirect_widget.layout.display = "none"
        error_widget.layout.display = "none"
        success_widget.layout.display = "none"

        display(all_widget)

    @staticmethod
    def login_sync(
        force=False,
        success_callback=None,
        redirect_callback=None,
        error_callback=None,
        post_success=None,
        **kwargs,
    ):
        # verify if token already exists, authenticate token if exists
        if not force:
            token = Auth.get_auth_token()
            if token:
                try:
                    Auth.validate_token(token)
                    if success_callback:
                        success_callback()
                    if post_success:
                        post_success()
                    return
                except AuthenticationFailedError:
                    pass

        api_host = get_base_url()
        auth_info = None

        # authorize user
        url = urljoin(
            api_host,
            'user.identity.proxiedAuthorize?{}'.format(
                urlencode({'provider': 'jina-login'})
            ),
        )

        response = requests.get(url, stream=True)

        # iterate through response
        for line in response.iter_lines():
            item = json.loads(line.decode('utf-8'))
            event = item['event']

            if event == 'redirect':
                href = item['data']["redirectTo"]
                if redirect_callback:
                    redirect_callback(href=href)
                else:
                    print(
                        f'Your browser is going to open the login page.\n'
                        f'If this fails please open the following link: {href}'
                    )
                    webbrowser.open(href)

            elif event == 'authorize':
                if item['data']['code'] and item['data']['state']:
                    auth_info = item['data']
                else:
                    err = item['data']["error_description"]
                    if error_callback:
                        error_callback(err=err)
                    else:
                        rich_print(
                            ':rotating_light: Authentication failed: {}'.format(err)
                        )

            elif event == 'error':
                err = item['data']
                if error_callback:
                    error_callback(err=err)
                else:
                    rich_print(':rotating_light: Authentication failed: {}'.format(err))
            else:
                err = f'Unknown event: {event}'
                if error_callback:
                    error_callback(err=err)
                else:
                    rich_print(':rotating_light: {}'.format(err))

        if auth_info is None:
            return

        # retrieving and saving token
        url = urljoin(api_host, 'user.identity.grant.auto')
        response = requests.post(url, json=auth_info)
        response.raise_for_status()
        json_response = get_json_from_response(response)
        token = json_response['data']['token']
        config.set('auth_token', token)

        user = json_response['data'].get('user', {})
        username = user.get('name')
        name = user.get('nickname') or username

        # dockerauth
        from hubble.dockerauth import auto_deploy_hubble_docker_credential_helper

        auto_deploy_hubble_docker_credential_helper()

        if success_callback:
            success_callback()
        else:
            rich_print(
                f':closed_lock_with_key: [green]Successfully logged in to Jina AI[/] '
                f'as [b]{name} (username: {username})[/b]!'
            )

        if post_success:
            post_success()

    @staticmethod
    async def login(force=False, **kwargs):
        # verify if token already exists, authenticate token if exists
        if not force:
            token = Auth.get_auth_token()
            if token:
                try:
                    Auth.validate_token(token)
                    return
                except AuthenticationFailedError:
                    pass

        api_host = get_base_url()
        auth_info = None
        async with aiohttp.ClientSession(trust_env=True) as session:
            kwargs['provider'] = kwargs.get('provider', 'jina-login')

            async with session.get(
                url=urljoin(
                    api_host,
                    'user.identity.proxiedAuthorize?{}'.format(urlencode(kwargs)),
                ),
            ) as response:
                async for line in response.content:
                    item = json.loads(line.decode('utf-8'))
                    event = item['event']
                    if event == 'redirect':
                        print(
                            f'Your browser is going to open the login page.\n'
                            f'If this fails please open the following link: {item["data"]["redirectTo"]}'
                        )
                        webbrowser.open(item['data']['redirectTo'])
                    elif event == 'authorize':
                        if item['data']['code'] and item['data']['state']:
                            auth_info = item['data']
                        else:
                            rich_print(
                                ':rotating_light: Authentication failed: {}'.format(
                                    item['data']['error_description']
                                )
                            )
                    elif event == 'error':
                        rich_print(
                            ':rotating_light: Authentication failed: {}'.format(
                                item['data']
                            )
                        )
                    else:
                        rich_print(':rotating_light: Unknown event: {}'.format(event))

        if auth_info is None:
            return

        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.post(
                url=urljoin(api_host, 'user.identity.grant.auto'),
                data=auth_info,
            ) as response:
                response.raise_for_status()
                json_response = await response.json()
                token = json_response['data']['token']

                user = json_response['data'].get('user', {})
                username = user.get('nickname') or user.get('name')

                config.set('auth_token', token)

                from hubble.dockerauth import (
                    auto_deploy_hubble_docker_credential_helper,
                )

                auto_deploy_hubble_docker_credential_helper()

                rich_print(
                    f':closed_lock_with_key: [green]Successfully logged in to Jina AI[/] as [b]{username}[/b]!'
                )

    @staticmethod
    async def logout():
        api_host = get_base_url()

        token = Auth.get_auth_token()
        token_from_config = Auth.get_auth_token_from_config()
        if token != token_from_config:
            rich_print(':warning: The token from environment variable is ignored.')

        async with aiohttp.ClientSession(trust_env=True) as session:
            session.headers.update({'Authorization': f'token {token_from_config}'})

            async with session.post(
                url=urljoin(api_host, 'user.session.dismiss')
            ) as response:
                json_response = await response.json()
                if json_response['code'] == 401:
                    from hubble.dockerauth import (
                        remove_all_hubble_docker_credential_helper,
                    )

                    remove_all_hubble_docker_credential_helper()
                    rich_print(
                        ':unlock: You are not logged in locally. There is no need to log out.'
                    )
                elif json_response['code'] == 200:
                    from hubble.dockerauth import (
                        remove_all_hubble_docker_credential_helper,
                    )

                    remove_all_hubble_docker_credential_helper()
                    config.delete('auth_token')
                    rich_print(':unlock: You have successfully logged out.')
                else:
                    rich_print(
                        f':rotating_light: Failed to log out. {json_response["message"]}'
                    )
