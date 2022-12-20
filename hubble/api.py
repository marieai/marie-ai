import os

from . import Client
from . import login as _login
from . import logout as _logout
from .excepts import AuthenticationFailedError, AuthenticationRequiredError
from .utils.api_utils import get_json_from_response


def logout(*args):
    _logout()


def login(args):
    client = Client(jsonify=True)
    username = ''

    if args.force:
        _login(prompt='login', force=True)
        return

    try:
        username = client.username
    except Exception as ex:
        if isinstance(ex, (AuthenticationRequiredError, AuthenticationFailedError)):
            _login(prompt='login')
        else:
            raise ex
    else:
        from .dockerauth import auto_deploy_hubble_docker_credential_helper

        auto_deploy_hubble_docker_credential_helper()

        if username:
            from rich.console import Console

            console = Console()
            console.print(
                f':closed_lock_with_key: You are already logged in as [b green]{username}[/b green].',
                '',
                'If you want to log in to another account, please run either:',
                '- [b]jina auth logout[/]',
                '- or, [b]jina auth login -f[/]',
                sep=os.linesep,
            )


def token(args):
    client = Client(max_retries=None)

    if args.operation == 'create':
        response = client.create_personal_access_token(
            name=args.name, expiration_days=args.expire
        )

        response.raise_for_status()
        json_response = get_json_from_response(response)
        token = json_response['data']['token']

        import rich
        from rich.panel import Panel

        rich.print(
            Panel(
                f'''[b]{token}[/b]

You can set it as an env var [b]JINA_AUTH_TOKEN[/b]''',
                title=':party_popper: [green]New token created[/]',
                subtitle=':point_up:️ [yellow] This token is only shown once![/]',
                width=50,
            )
        )

    if args.operation == 'delete':
        response = client.delete_personal_access_token(name=args.name)

        response.raise_for_status()

        import rich
        from rich.panel import Panel

        rich.print(
            Panel(
                f'[b]{args.name}[/b]',
                title=':party_popper: [green]Successfully deleted',
                width=50,
            )
        )

    if args.operation == 'list':
        response = client.list_personal_access_tokens()
        response.raise_for_status()
        json_response = get_json_from_response(response)
        tokens = json_response['data']['personal_access_tokens']

        import rich
        from rich.table import Table

        table = Table(title='Your Personal Access Tokens', highlight=True)
        table.add_column('Name')
        table.add_column('Type')
        table.add_column('Create at', justify='center')
        table.add_column('Expire at', justify='center')
        table.add_column('Last use at', justify='center')

        for token in tokens:
            table.add_row(
                token['name'],
                token['type'],
                token['createdAt'],
                token['expireAt'],
                token['updatedAt'],
            )

        rich.print(table)
