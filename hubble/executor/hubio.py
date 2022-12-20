"""Module for wrapping Jina Hub API calls."""

import argparse
import copy
import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urljoin

import hubble
from hubble.executor import HubExecutor
from hubble.executor.helper import (
    __resources_path__,
    __unset_msg__,
    archive_package,
    check_requirements_env_variable,
    disk_cache_offline,
    download_with_resume,
    get_async_tasks,
    get_cache_db,
    get_download_cache_dir,
    get_hub_packages_dir,
    get_hubble_error_message,
    get_request_header,
    get_requirements_env_variables,
    get_rich_console,
    get_tag_from_dist_info_path,
    parse_hub_uri,
    retry,
    status_task,
    upload_file,
)
from hubble.executor.hubapi import (
    extract_executor_name,
    get_dist_path_of_executor,
    get_lockfile,
    install_local,
    install_package_dependencies,
    list_local,
    load_config,
)
from hubble.utils.api_utils import get_json_from_response


class HubIO:
    """:class:`HubIO` lets you interact with the Jina Hub registry.
    You can use it from the CLI to package a directory into a Jina Hub Executor and publish it to the world.
    Examples:
        - :command:`jina hub push my_executor/` to push the Executor package to Jina Hub
        - :command:`jina hub pull <UUID8>` to download the Executor identified by its UUID

    To create a :class:`HubIO` object, simply:

        .. highlight:: python
        .. code-block:: python
            hubio = HubIO(args)

    :param args: arguments
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.logger = logging.getLogger(self.__class__.__name__)

    def new(self) -> None:
        """Create a new Executor folder interactively."""

        from rich import box, print
        from rich.panel import Panel
        from rich.progress import track
        from rich.prompt import Confirm, Prompt
        from rich.syntax import Syntax
        from rich.table import Table

        console = get_rich_console()

        print(
            Panel.fit(
                '''
An [bold green]Executor[/bold green] is how Jina processes [bold]Documents[/bold].

This guide helps you create an Executor in 30 seconds.''',
                title='Create New Executor',
            )
        )

        exec_name = (
            self.args.name
            if self.args.name
            else Prompt.ask(
                ':grey_question: What is the [bold]name[/bold] of your Executor?\n'
                '[dim]CamelCase is required[/dim]',
                default=f'MyExecutor{random.randint(0, 100)}',
            )
        )

        exec_path = (
            self.args.path
            if self.args.path
            else Prompt.ask(
                ':grey_question: Which [bold]folder[/bold] do you want to store your Executor in?',
                default=os.path.join(os.getcwd(), exec_name),
            )
        )
        exec_description = '{{}}'
        exec_keywords = '{{}}'
        exec_url = '{{}}'

        is_dockerfile = 'none'

        if self.args.advance_configuration or Confirm.ask(
            '[green]That\'s all we need to create an Executor![/green]\n'
            ':grey_question: Or do you want to proceed to advanced configuration? '
            '[dim](GPU support, meta information on Hub, etc.)[/]',
            default=False,
        ):
            print(
                Panel.fit(
                    '''
[bold]Dockerfile[/bold] describes how this Executor will be built. It is useful when
your Executor has non-trivial dependencies or must be run in a certain environment.

- If [bold]Dockerfile[/bold] is not given, Jina Cloud automatically generates one.
- If you provide a [bold]Dockerfile[/bold], then Jina Cloud will respect it when building the Executor.

Here are some Dockerfile templates to choose from:
- [b]cpu[/b]: CPU-only Executor with Jina as base image;
- [b]torch-gpu[/b]: GPU-enabled Executor with PyTorch as the base image;
- [b]tf-gpu[/b]: GPU-enabled Executor with TensorFlow as the base image;
- [b]jax-gpu[/b]: GPU-enabled Executor with JAX installed.
''',
                    title=':package: [bold]Dockerfile[/bold]',
                    width=80,
                )
            )

            is_dockerfile = self.args.dockerfile or Prompt.ask(
                ':grey_question: How do you want to generate the [bold]Dockerfile[/bold] for this Executor?',
                choices=['cpu', 'torch-gpu', 'tf-gpu', 'jax-gpu', 'none'],
                default='cpu',
            )

            print(
                Panel.fit(
                    '''
Metadata helps other users identify, search and reuse your Executor on Jina Cloud.
''',
                    title=':name_badge: [bold]Meta Info[/bold]',
                    width=80,
                )
            )

            exec_description = (
                self.args.description
                if self.args.description
                else (
                    Prompt.ask(
                        ':grey_question: Please give a [bold]short description[/bold] of your Executor:\n'
                        f'[dim]Example: {exec_name} embeds images into 128-dim vectors using ResNet.[/dim]'
                    )
                )
            )

            exec_keywords = (
                self.args.keywords
                if self.args.keywords
                else (
                    Prompt.ask(
                        ':grey_question: Please give some [bold]keywords[/bold] '
                        'to help people search your Executor [dim](separated by commas)[/dim]\n'
                        '[dim]Example: image, cv, embedding, encoding, resnet[/dim]'
                    )
                )
            )

            exec_url = (
                self.args.url
                if self.args.url
                else (
                    Prompt.ask(
                        ':grey_question: What is the [bold]URL[/bold] of the Executor\'s GitHub repo?\n'
                        '[dim]Example: https://github.com/yourname/my-executor[/dim]'
                    )
                )
            )

            print('[green]That\'s all we need to create an Executor![/green]')

        def mustache_repl(srcs):
            for src in track(
                srcs, description=f'Creating {exec_name}...', total=len(srcs)
            ):
                dest = src
                if dest.endswith('.Dockerfile'):
                    dest = 'Dockerfile'
                with open(
                    os.path.join(__resources_path__, 'executor-template', src)
                ) as fp, open(os.path.join(exec_path, dest), 'w') as fpw:
                    f = (
                        fp.read()
                        .replace('{{exec_name}}', exec_name)
                        .replace(
                            '{{exec_description}}',
                            exec_description if exec_description != '{{}}' else '',
                        )
                        .replace(
                            '{{exec_keywords}}',
                            str(exec_keywords.split(','))
                            if exec_keywords != '{{}}'
                            else '[]',
                        )
                        .replace('{{exec_url}}', exec_url if exec_url != '{{}}' else '')
                    )
                    fpw.writelines(f)

        Path(exec_path).mkdir(parents=True, exist_ok=True)
        pkg_files = [
            'executor.py',
            'README.md',
            'requirements.txt',
            'config.yml',
        ]

        if is_dockerfile == 'cpu':
            pkg_files.append('Dockerfile')
        elif is_dockerfile == 'torch-gpu':
            pkg_files.append('torch.Dockerfile')
        elif is_dockerfile == 'jax-gpu':
            pkg_files.append('torch.Dockerfile')
        elif is_dockerfile == 'tf-gpu':
            pkg_files.append('tf.Dockerfile')
        elif is_dockerfile != 'none':
            raise ValueError(f'Unknown Dockerfile type: {is_dockerfile}')

        mustache_repl(pkg_files)

        table = Table(box=box.SIMPLE)
        table.add_column('Filename', style='cyan', no_wrap=True)
        table.add_column('Description', no_wrap=True)

        # adding the columns in order of `ls` output
        table.add_row(
            'config.yml',
            'The YAML configuration file of the Executor. You can define [bold]__init__[/bold] '
            'arguments using the [bold]with[/bold] keyword.'
            + '\nYou can also define metadata for the Executor, for easier discovery on Jina Hub.',
        )

        table.add_row(
            '',
            Panel(
                Syntax(
                    f'''
jtype: {exec_name}
with:
  foo: 1
  bar: hello
py_modules:
  - executor.py
metas:
  name: {exec_name}
  description: {exec_description if exec_description != '{{}}' else ''}
  url: {exec_url if exec_url != '{{}}' else ''}
  keywords: {exec_keywords if exec_keywords != '{{}}' else '[]'}
''',
                    'yaml',
                    theme='monokai',
                    line_numbers=True,
                    word_wrap=True,
                ),
                title='config.yml',
                width=50,
                expand=False,
            ),
        )

        if is_dockerfile != 'none':
            table.add_row(
                'Dockerfile',
                'The Dockerfile describes how this Executor will be built.',
            )

        table.add_row('executor.py', 'The Executor\'s main logic file.')
        table.add_row('README.md', 'The Executor\'s usage guide.')
        table.add_row('requirements.txt', 'The Executor\'s Python dependencies.')

        final_table = Table(box=None)

        final_table.add_row(
            'Congratulations! You have successfully created an Executor! Here are the next steps:'
        )

        p0 = Panel(
            Syntax(
                f'ls {exec_path}',
                'console',
                theme='monokai',
                line_numbers=True,
                word_wrap=True,
            ),
            title='1. Check out your Executor',
            width=120,
            expand=False,
        )

        p1 = Panel(
            table,
            title='2. Understand Executor folder structure',
            width=120,
            expand=False,
        )

        p12 = Panel(
            Syntax(
                f'jina executor --uses {exec_path}/config.yml',
                'console',
                theme='monokai',
                line_numbers=True,
                word_wrap=True,
            ),
            title='3. Test your Executor locally',
            width=120,
            expand=False,
        )

        p2 = Panel(
            Syntax(
                f'jina hub push {exec_path}',
                'console',
                theme='monokai',
                line_numbers=True,
                word_wrap=True,
            ),
            title='4. Share your Executor on Jina Hub',
            width=120,
            expand=False,
        )

        for _p in [p0, p1, p12, p2]:
            final_table.add_row(_p)

        p = Panel(
            final_table,
            title=':tada: Next steps',
            width=130,
            expand=False,
        )
        console.print(p)

    def _send_push_request(self, console, st, url, req_header, content, form_data):
        st.update('Uploading...')
        resp = upload_file(
            url,
            'filename',
            content,
            dict_data=form_data,
            headers=req_header,
            stream=True,
            method='post',
        )

        warnings = []
        verbose = form_data.get('verbose', False)
        image = None
        session_id = req_header.get('jinameta-session-id')

        if resp.status_code >= 400:
            json_resp = get_json_from_response(resp)
            msg = json_resp.get('readableMessage')
            raise Exception(f'{ msg or "Unknown Error"} session_id: {session_id}')

        for stream_line in resp.iter_lines():

            stream_msg = json.loads(stream_line)
            t = stream_msg.get('type')
            subject = stream_msg.get('subject')
            payload = stream_msg.get('payload', '')

            if t == 'error':
                msg = stream_msg.get('message')
                hubble_err = payload
                overridden_msg = ''
                detail_msg = ''
                if isinstance(hubble_err, dict):
                    (overridden_msg, detail_msg) = get_hubble_error_message(hubble_err)
                    if not msg:
                        msg = detail_msg

                if overridden_msg and overridden_msg != detail_msg:
                    self.logger.warning(overridden_msg)

                raise Exception(
                    f'{overridden_msg or msg or "Unknown Error"} session_id: {session_id}'
                )
            elif t == 'warning':
                warnings.append(stream_msg.get('message'))

            if t == 'progress' and subject == 'buildWorkspace':
                legacy_message = stream_msg.get('legacyMessage', {})
                status = legacy_message.get('status', '')
                st.update(f'Cloud building ... [dim]{subject}: {t} ({status})[/dim]')

            elif t == 'complete':
                image = stream_msg['payload']
                if stream_msg.get('warning'):
                    warnings.append(stream_msg.get('warning'))
                st.update(
                    f'Cloud building ... [dim]{subject}: {t} ({stream_msg["message"]})[/dim]'
                )
                break

            elif t and subject:
                if verbose and t == 'console':
                    console.log(f'Cloud building ... [dim]{subject}: {payload}[/dim]')
                else:
                    st.update(f'Cloud building ... [dim]{subject}: {t} {payload}[/dim]')

        if image:
            self._prettyprint_result(console, image, warnings=warnings)
        else:
            raise Exception(f'Unknown Error, session_id: {session_id}')

        return image

    @hubble.login_required
    def push(self) -> None:
        """Push the Executor package to Jina Hub."""

        work_path = Path(self.args.path)

        exec_tags = None
        exec_immutable_tags = None
        image = None

        if self.args.tag:
            exec_tags = ','.join(self.args.tag)
        if self.args.protected_tag:
            exec_immutable_tags = ','.join(self.args.protected_tag)

        dockerfile = None
        if self.args.dockerfile:
            dockerfile = Path(self.args.dockerfile)
            if not dockerfile.exists():
                raise Exception(f'The given Dockerfile `{dockerfile}` does not exist!')
            if dockerfile.parent != work_path:
                raise Exception(
                    f'The Dockerfile must be placed at the given folder `{work_path}`'
                )

            dockerfile = dockerfile.relative_to(work_path)

        build_env = None
        if type(self.args.build_env) is list:
            build_env_dict = {}
            for index, env in enumerate(self.args.build_env):
                env_list = env.strip().split('=')
                if len(env_list) != 2:
                    raise Exception(
                        f'The `--build-env` parameter: `{env}` is in the wrong format. '
                        f'you can use: `--build-env {env}=YOUR_VALUE`.'
                    )
                if check_requirements_env_variable(env_list[0]) is False:
                    raise Exception(
                        f'The `--build-env` parameter key:`{env_list[0]}` can only '
                        'consist of numbers, upper-case letters and underscore.'
                    )
                build_env_dict[env_list[0]] = env_list[1]
            build_env = build_env_dict if build_env_dict else None

        requirements_file = work_path / 'requirements.txt'

        requirements_env_variables = []
        if requirements_file.exists():
            requirements_env_variables = get_requirements_env_variables(
                requirements_file
            )
            for index, env in enumerate(requirements_env_variables):
                if check_requirements_env_variable(env) is False:
                    raise Exception(
                        f'The requirements.txt environment variables:`${env}` '
                        'can only consist of numbers, upper-case letter and underscore.'
                    )

        if len(requirements_env_variables) and not build_env:
            env_variables_str = ','.join(requirements_env_variables)
            error_str = (
                'requirements.txt sets environment variables as follows:'
                f'`{env_variables_str}` should use `--build-env'
            )
            for item in requirements_env_variables:
                error_str += f' {item}=YOUR_VALUE'
            raise Exception(f'{error_str}` to add it.')
        elif len(requirements_env_variables) and build_env:
            build_env_keys = list(build_env.keys())
            diff_env_variables = list(
                set(requirements_env_variables).difference(set(build_env_keys))
            )
            if len(diff_env_variables):
                diff_env_variables_str = ",".join(diff_env_variables)
                error_str = (
                    'requirements.txt sets environment variables as follows:'
                    f'`{diff_env_variables_str}` should use `--build-env'
                )
                for item in diff_env_variables:
                    error_str += f' {item}=YOUR_VALUE'
                raise Exception(f'{error_str}` to add it.')

        executor_name = extract_executor_name(work_path)
        if not executor_name and (not self.args.force_update or not self.args.secret):
            raise Exception(
                f'Can not extract executor from {work_path}. Please create config.yml or add "metas.name" to it.'
            )

        console = get_rich_console()
        with console.status(f'Pushing `{self.args.path}` ...') as st:
            req_header = get_request_header()
            try:
                st.update(f'Packaging {self.args.path} ...')
                md5_hash = hashlib.md5()
                bytesio = archive_package(work_path)
                content = bytesio.getvalue()
                md5_hash.update(content)
                md5_digest = md5_hash.hexdigest()

                # upload the archived package
                form_data = {
                    'public': 'True' if getattr(self.args, 'public', None) else 'False',
                    'private': 'True'
                    if getattr(self.args, 'private', None)
                    else 'False',
                    'md5sum': md5_digest,
                }

                if self.args.verbose:
                    form_data['verbose'] = 'True'

                if self.args.no_cache:
                    form_data['buildWithNoCache'] = 'True'

                if exec_tags:
                    form_data['tags'] = exec_tags

                if exec_immutable_tags:
                    form_data['immutableTags'] = exec_immutable_tags

                if dockerfile:
                    form_data['dockerfile'] = str(dockerfile)

                if build_env:
                    form_data['buildEnv'] = json.dumps(build_env)

                st.update('Connecting to Jina Hub ...')

                if self.args.force_update and self.args.secret:
                    # for backward compatibility updating/pushing with secret
                    form_data['id'] = self.args.force_update
                    form_data['secret'] = self.args.secret
                    hubble_url = urljoin(hubble.utils.get_base_url(), 'executor.update')
                else:
                    form_data['id'] = executor_name
                    hubble_url = urljoin(hubble.utils.get_base_url(), 'executor.push')

                image = self._send_push_request(
                    console,
                    st,
                    hubble_url,
                    req_header,
                    content,
                    form_data,
                )

            except KeyboardInterrupt:
                pass

            except Exception as e:  # IO related errors
                self.logger.error(
                    f'Please report this session_id: [yellow bold]{req_header["jinameta-session-id"]}[/] '
                    'to https://github.com/jina-ai/jina-hubble-sdk/issues'
                )
                raise e

        return image

    def _prettyprint_result(
        self, console, image, *, warnings: Optional[List[str]] = None
    ):
        from rich import box
        from rich.panel import Panel
        from rich.table import Table

        uuid8 = image['id']
        secret = image.get('secret')
        visibility = image['visibility']
        commit = image.get('commit', {})
        tags = commit.get('tags', None)
        tag = tags[0] if tags and isinstance(tags, list) else None

        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column(no_wrap=True)
        table.add_column(no_wrap=True)
        if 'name' in image:
            table.add_row(':name_badge: Name', image['name'])

        table.add_row(
            ':link: Jina Hub URL',
            f'[link=https://cloud.jina.ai/executor/{uuid8}/]https://cloud.jina.ai/executor/{uuid8}/[/link]',
        )

        if secret:
            table.add_row(':lock: Secret', secret)
            table.add_row(
                '',
                '👆 [bold red]Please keep this token in a safe place!',
            )

        table.add_row(':eyes: Visibility', visibility)

        if warnings:
            table.add_row(
                ':exclamation: Warnings',
                '👇 [bold yellow]Process finished with warnings!',
            )
            for warning in warnings:
                table.add_row('', f'[yellow]• {warning}')

        p1 = Panel(
            table,
            title='Published',
            width=100,
            expand=False,
        )
        console.print(p1)

        name = image.get('name', uuid8)
        owner_name = image.get('owner', {}).get('name', None)

        # TODO: remove legacy "jinahub" support after the namespace release
        if not owner_name or bool(secret):
            scheme_prefix = 'jinahub'
            is_legacy_uri = True
        else:
            scheme_prefix = 'jinaai'
            is_legacy_uri = False

        if visibility == 'public' or not secret:
            executor_name = name
        else:
            executor_name = f'{name}:{secret}'

        if owner_name:
            executor_name = f'{owner_name}/{executor_name}'

        if tag:
            executor_name += f'/{tag}' if is_legacy_uri else f':{tag}'

        if not self.args.no_usage:
            self._prettyprint_usage(
                console, scheme_prefix=scheme_prefix, executor_name=executor_name
            )

        return uuid8, secret

    def _prettyprint_usage(self, console, *, scheme_prefix, executor_name):
        from rich import box
        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.table import Table

        param_str = Table(
            box=box.SIMPLE,
            show_header=False,
        )
        param_str.add_column('')
        param_str.add_column('')
        param_str.add_column('')

        param_str.add_row(
            'Container',
            'YAML',
            Syntax(f"uses: {scheme_prefix}+docker://{executor_name}", 'yaml'),
        )
        param_str.add_row(
            None,
            'Python',
            Syntax(f".add(uses='{scheme_prefix}+docker://{executor_name}')", 'python'),
        )

        param_str.add_row()
        param_str.add_row(
            'Sandbox',
            'YAML',
            Syntax(f"uses: {scheme_prefix}+sandbox://{executor_name}", 'yaml'),
        )
        param_str.add_row(
            '',
            'Python',
            Syntax(f".add(uses='{scheme_prefix}+sandbox://{executor_name}')", 'python'),
        )

        param_str.add_row()
        param_str.add_row(
            'Source',
            'YAML',
            Syntax(f"uses: {scheme_prefix}://{executor_name}", 'yaml'),
        )
        param_str.add_row(
            '',
            'Python',
            Syntax(f".add(uses='{scheme_prefix}://{executor_name}')", 'python'),
        )

        console.print(
            Panel(param_str, title='Usage', expand=False, width=len(executor_name) + 65)
        )

    def _prettyprint_build_env_usage(self, console, build_env, usage_kind=None):
        from rich import box
        from rich.panel import Panel
        from rich.table import Table

        param_str = Table(
            box=box.SIMPLE,
        )
        param_str.add_column('Environment variable')
        param_str.add_column('Your value')

        for index, item in enumerate(build_env):
            param_str.add_row(f'{item}', 'your value')

        console.print(
            Panel(
                param_str,
                title='build_env',
                subtitle='You have to set the above environment variables',
                expand=False,
                width=100,
            )
        )

    def _prettyprint_status_usage(
        self, console, work_path, task_id=None, usage_kind=None
    ):
        from rich import box
        from rich.panel import Panel
        from rich.table import Table

        param_str = Table(
            box=box.SIMPLE,
        )

        param_str.add_column(
            'If you don\'t want to wait for the build result, you can interrupt it with \'Ctrl + C\'. '
            'You can resume the build progress at any time.'
        )
        param_str.add_row(
            f'Run `jina hub status {work_path}` to check the latest build status of the '
            'Executor in the current working directory',
        )

        if task_id:
            param_str.add_row('')
            param_str.add_row(
                f'You can also run `jina hub status --id {task_id} --replay --verbose` '
                'to get the full log of this particular build.',
            )

        console.print(
            Panel(
                param_str,
                title='Building',
                expand=False,
                width=100,
            )
        )

    def _status_with_progress(self, console, st, task_id, replay=False, verbose=False):

        req_header = get_request_header()
        dict_data = {}
        dict_data['replay'] = replay
        dict_data['verbose'] = verbose

        session_id = req_header.get('jinameta-session-id')
        task_progress = status_task(
            url=urljoin(hubble.utils.get_base_url(), 'executor.queryProgress'),
            id=task_id,
            dict_data=dict_data,
            headers=req_header,
            stream=True,
            method='post',
        )

        image = None
        for stream_line in task_progress.iter_lines():
            stream_msg = json.loads(stream_line)
            t = stream_msg.get('type')

            code = stream_msg.get('code')
            if code and code >= 400:
                error = stream_msg.get('message')
                raise Exception(f'{ error or "Unknown Error"} session_id: {session_id}')

            if t == 'error':
                error = stream_msg.get('message')
                raise Exception(f'{ error or "Unknown Error"} session_id: {session_id}')

            elif t == 'report':
                status = stream_msg.get('status')
                if status == 'pending':
                    if replay:
                        console.log(
                            f'Cloud pending ... [dim]: {t} {task_id} ({status})[/dim]'
                        )
                    else:
                        st.update(f'Cloud pending ... [dim]: {t} ({status})[/dim]')

                elif status == 'failed':
                    error = stream_msg.get('error', {})
                    msg = error.get('message')
                    message = stream_msg.get('message')
                    raise Exception(
                        f'{ msg or message or "Unknown Error"} session_id: {session_id}'
                    )

                elif status == 'waiting':
                    error = stream_msg.get('error')
                    if replay:
                        task = stream_msg.get('task', {})
                        task_id = task.get('_id')
                        console.log(
                            f'Cloud waiting ... [dim]: {t} task: {task_id} {error or ""} ({status})[/dim]'
                        )
                    else:
                        st.update(
                            f'Cloud waiting ... [dim]: {t} {error or ""} ({status})[/dim]'
                        )

                elif status == 'succeeded':
                    if stream_msg.get('result', None):
                        image = stream_msg['result']
                        executor_id = image.get('id')
                        if replay:
                            console.log(
                                f'Cloud succeeded ... [dim]: {t} Executor: {executor_id} ({status})[/dim]'
                            )
                        else:
                            st.update(
                                f'Cloud succeeded ... [dim]: {t} Executor: {executor_id} ({status})[/dim]'
                            )
                    else:
                        task = stream_msg.get('task', {})
                        task_id = task.get('_id')
                        if replay:
                            console.log(
                                f'Cloud succeeded ... [dim]: {t} task: {task_id} ({status})[/dim]'
                            )
                        else:
                            st.update(
                                f'Cloud succeeded ... [dim]: {t} task: {task_id} ({status})[/dim]'
                            )

            elif t == 'progress':
                data = stream_msg.get('data', {})
                legacy_message = data.get('data', {})

                type = legacy_message.get('type')
                subject = legacy_message.get('subject')
                payload = legacy_message.get('payload', '')

                if verbose:
                    if type == 'console':
                        console.log(
                            f'Cloud building ... [dim]{subject}: {t} {payload}[/dim]'
                        )
                    else:
                        console.log(f'Cloud building ... [dim]{subject}: {type} [/dim]')
                else:
                    if type == 'console':
                        st.update(
                            f'Cloud building ... [dim]{subject}: {t} {payload}[/dim]'
                        )
                    else:
                        st.update(f'Cloud building ... [dim]{subject}: {type} [/dim]')

            else:
                console.log(f'Cloud succeeded ... [dim]: {t} [/dim]')

        return image

    def status(self) -> None:
        """Query the build status of the Executor."""

        task_id = None
        if self.args.id:
            task_id = self.args.id
        else:
            work_path = Path(self.args.path)
            name = extract_executor_name(work_path)
            if not name:
                raise Exception(
                    f'Can not extract executor from {work_path}. Please create config.yml or add "metas.name" to it.'
                )

            tasks = get_async_tasks(name=name)
            if not tasks:
                raise Exception(
                    f'No build task found for {name}. Please make sure you have built it before.'
                )

            task_id = tasks[0].get('_id')
            print(f'Found task_id: {task_id} for {name}')

        if not task_id:
            raise Exception(
                'Error: Can\'t get task_id! You can set `--id your task_id` to get build progress info.'
            )

        verbose = True if self.args.verbose else False
        replay = True if self.args.replay else False

        console = get_rich_console()
        with console.status(f'Querying `{task_id}` ...') as st:

            image = self._status_with_progress(console, st, task_id, replay, verbose)

            if image:
                self.args.no_usage = False
                self._prettyprint_result(console, image)
            else:
                console.log(f'Waiting `{task_id}` ...')

    def _prettyprint_list_usage(self, console, executors, base_path):
        from rich import box
        from rich.panel import Panel
        from rich.table import Column, Table

        param_str = Table(
            Column(header="Executor", no_wrap=True),
            Column(header="Tag", no_wrap=True),
            Column(header="Relative path", no_wrap=True),
            box=box.SIMPLE,
            title=f'Base path: {base_path}',
            title_style='blue',
            show_lines=True,
            min_width=30 + len(str(base_path)),
        )

        for item in executors:
            name = item['name']
            relative_path = item['relative_path']
            tag = item.get('tag', None)
            param_str.add_row(
                f'{name}',
                f'{tag}' if tag else '',
                f'{relative_path}',
            )

        console.print(
            Panel(
                param_str,
                title='List',
                expand=False,
                width=120,
            )
        )

    def list(self) -> None:
        executors = []
        base_path = get_hub_packages_dir()
        for executor_dist_info_path in list_local():
            executor_path = executor_dist_info_path.parent
            config = load_config(executor_path)
            tag = get_tag_from_dist_info_path(executor_dist_info_path)
            relative_executor_path = executor_path.stem
            executors.append(
                {
                    'name': config['jtype'],
                    'tag': tag,
                    'relative_path': f'./{relative_executor_path}',
                }
            )

        console = get_rich_console()
        self._prettyprint_list_usage(console, executors, base_path)

    @staticmethod
    @disk_cache_offline(cache_file=str(get_cache_db()))
    def fetch_meta(
        name: str,
        tag: str,
        image_required: bool = True,
        rebuild_image: bool = True,
        *,
        secret: Optional[str] = None,
        force: bool = False,
    ) -> HubExecutor:
        """Fetch Executor metadata from Jina Hub.
        :param name: the UUID/name of the Executor
        :param tag: the tag of the Executor if available, otherwise, use `None` as the value
        :param secret: the access secret of the Executor
        :param image_required: indicates whether a Docker image is required or not
        :param rebuild_image: indicates whether Jina Hub needs to rebuild image or not
        :param force: if set to True, access to fetch_meta will always pull latest Executor metas, otherwise, default
            to local cache
        :return: meta of Executor

        .. note::
            Significant parameters like `name` and `tag` should be passed via ``args``
            and `force` and `secret` as ``kwargs``, otherwise, caching will not work.
        """
        import requests

        @retry(num_retry=3)
        def _send_request_with_retry(url, **kwargs):
            resp = requests.post(url, **kwargs)
            if resp.status_code != 200:
                if resp.text:
                    raise Exception(resp.text)
                resp.raise_for_status()

            return resp

        pull_url = urljoin(hubble.utils.get_base_url(), 'executor.getPackage')

        payload = {'id': name, 'include': ['code'], 'rebuildImage': rebuild_image}
        if image_required:
            payload['include'].append('docker')
        if secret:
            payload['secret'] = secret
        if tag:
            payload['tag'] = tag

        req_header = get_request_header()

        resp = _send_request_with_retry(pull_url, json=payload, headers=req_header)
        resp = get_json_from_response(resp)['data']

        images = resp['package'].get('containers', [])
        image_name = images[0] if images else None
        if image_required and not image_name:
            raise RuntimeError(
                f'No image found for Executor "{name}", '
                f'tag: {tag}, commit: {resp.get("commit", {}).get("id")}, '
                f'session_id: {req_header.get("jinameta-session-id")}'
            )
        buildEnv = resp['commit'].get('commitParams', {}).get('buildEnv', None)
        return HubExecutor(
            uuid=resp['id'],
            name=resp.get('name', None),
            commit_id=resp['commit'].get('id'),
            tag=tag or resp['commit'].get('tags', [None])[0],
            visibility=resp['visibility'],
            image_name=image_name,
            archive_url=resp['package']['download'],
            md5sum=resp['package']['md5'],
            build_env=list(buildEnv.keys()) if buildEnv else [],
        )

    @staticmethod
    def deploy_public_sandbox(args: Union[argparse.Namespace, Dict]) -> str:
        """
        Deploy a public sandbox to Jina Hub.
        :param args: arguments parsed from the CLI

        :return: the host and port of the sandbox
        """

        try:
            from jina import __version__ as jina_version
        except ImportError:
            jina_version = __unset_msg__

        args_copy = copy.deepcopy(args)
        if not isinstance(args_copy, Dict):
            args_copy = vars(args_copy)

        scheme, name, tag, secret = parse_hub_uri(args_copy.pop('uses', ''))
        payload = {
            'name': name,
            'tag': tag if tag else 'latest',
            'jina': jina_version,
            'args': args_copy,
            'secret': secret,
        }

        import requests

        console = get_rich_console()

        host = None
        port = None

        headers = get_request_header()
        response = requests.post(
            url=urljoin(hubble.utils.get_base_url(), 'sandbox.get'),
            json=payload,
            headers=headers,
        )
        json_response = get_json_from_response(response)
        if json_response.get('code') == 200:
            host = json_response.get('data', {}).get('host', None)
            port = json_response.get('data', {}).get('port', None)

        if host and port:
            console.log('🎉 A sandbox already exists, reusing it.')
            return host, port

        with console.status(
            f'[bold green]🚧 Deploying sandbox for [bold white]{name}[/bold white] since none exists...'
        ):
            try:
                response = requests.post(
                    url=urljoin(hubble.utils.get_base_url(), 'sandbox.create'),
                    json=payload,
                    headers=headers,
                )
                json_response = get_json_from_response(response)

                data = json_response.get('data') or {}
                host = data.get('host', None)
                port = data.get('port', None)
                if not host or not port:
                    raise Exception(f'Failed to deploy sandbox: {json_response}')

                console.log('🎉 Deployment completed, using it.')
            except BaseException:
                console.log(
                    '🚨 Deployment failed. Please raise an issue: https://github.com/jina-ai/jina-hubble-sdk/issues/new'
                )
                raise

        return host, port

    def _pull_with_progress(self, log_streams, console):
        from rich.progress import BarColumn, DownloadColumn, Progress

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            DownloadColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            console=console,
            transient=True,
        ) as progress:
            tasks = {}
            for log in log_streams:
                if 'status' not in log:
                    continue
                status = log['status']
                status_id = log.get('id', None)
                pg_detail = log.get('progressDetail', None)

                if (pg_detail is None) or (status_id is None):
                    self.logger.debug(status)
                    continue

                if status_id not in tasks:
                    tasks[status_id] = progress.add_task(status, total=0)

                task_id = tasks[status_id]

                if ('current' in pg_detail) and ('total' in pg_detail):
                    progress.update(
                        task_id,
                        completed=pg_detail['current'],
                        total=pg_detail['total'],
                        description=status,
                    )
                elif not pg_detail:
                    progress.update(task_id, advance=0, description=status)

    def _load_docker_client(self):
        # with ImportExtensions(required=True):
        import docker.errors
        from docker import APIClient
        from hubble import __windows__

        try:
            self._client = docker.from_env()
            # low-level client
            self._raw_client = APIClient(
                base_url=docker.constants.DEFAULT_NPIPE
                if __windows__
                else docker.constants.DEFAULT_UNIX_SOCKET
            )
        except docker.errors.DockerException:
            self.logger.critical(
                'Docker daemon doesn\'t seem to be running. Please run the Docker daemon and try again.'
            )
            exit(1)

    def pull(self) -> str:
        """Pull the Executor package from Jina Hub.

        :return: the `uses` string
        """

        console = get_rich_console()
        cached_zip_file = None
        executor_name = None
        build_env = None
        scheme = None

        try:
            need_pull = self.args.force_update
            with console.status(f'Pulling {self.args.uri}...') as st:
                scheme, name, tag, secret = parse_hub_uri(self.args.uri)
                image_required = scheme.endswith('+docker')

                st.update(f'Fetching [bold]{name}[/bold] from Jina Hub ...')
                executor, from_cache = HubIO.fetch_meta(
                    name,
                    tag,
                    image_required,
                    secret=secret,
                    force=need_pull,
                )

                build_env = executor.build_env

                if executor.visibility == 'public' or not secret:
                    executor_name = name
                else:
                    executor_name = f'{name}:{secret}'

                is_legacy_uri = scheme.startswith('jinahub')
                if tag:
                    executor_name += f'/{tag}' if is_legacy_uri else f':{tag}'

                if scheme.endswith('+docker'):
                    self._load_docker_client()
                    import docker

                    try:
                        self._client.images.get(executor.image_name)
                    except docker.errors.ImageNotFound:
                        need_pull = True

                    if need_pull:
                        st.update('Pulling image ...')
                        log_stream = self._raw_client.pull(
                            executor.image_name, stream=True, decode=True
                        )
                        st.stop()
                        self._pull_with_progress(
                            log_stream,
                            console,
                        )
                    return f'docker://{executor.image_name}'
                elif scheme == 'jinahub' or scheme == 'jinaai':
                    import filelock

                    if build_env:
                        self._prettyprint_build_env_usage(console, build_env)

                    with filelock.FileLock(get_lockfile(), timeout=-1):
                        try:
                            pkg_path, pkg_dist_path = get_dist_path_of_executor(
                                executor
                            )
                            # check commit id to upgrade
                            commit_file_path = (
                                pkg_dist_path / f'PKG-COMMIT-{executor.commit_id or 0}'
                            )
                            if (not commit_file_path.exists()) and any(
                                pkg_dist_path.glob('PKG-COMMIT-*')
                            ):
                                raise FileNotFoundError(
                                    f'{pkg_path} need to be upgraded'
                                )

                            st.update(
                                'Installing dependencies from [bold]requirements.txt[/bold]...'
                            )
                            install_package_dependencies(
                                install_deps=self.args.install_requirements,
                                pkg_dist_path=pkg_dist_path,
                                pkg_path=pkg_dist_path,
                            )

                        except FileNotFoundError:
                            need_pull = True

                        if need_pull:
                            # pull the latest Executor meta, as the cached data would expire
                            if from_cache:
                                executor, _ = HubIO.fetch_meta(
                                    name,
                                    tag,
                                    image_required,
                                    secret=secret,
                                    force=True,
                                )

                            st.update(f'Downloading {name} ...')
                            cached_zip_file = download_with_resume(
                                executor.archive_url,
                                get_download_cache_dir(),
                                f'{executor.uuid}-{executor.md5sum}.zip',
                                md5sum=executor.md5sum,
                            )

                            st.update(f'Unpacking {name} ...')
                            install_local(
                                cached_zip_file,
                                executor,
                                install_deps=self.args.install_requirements,
                            )

                            pkg_path, _ = get_dist_path_of_executor(executor)

                        return f'{pkg_path / "config.yml"}'
                else:
                    raise ValueError(f'{self.args.uri} is not a valid scheme')
        except KeyboardInterrupt:
            executor_name = None
        except Exception:
            executor_name = None
            raise
        finally:
            # delete downloaded zip package if existed
            if cached_zip_file is not None:
                cached_zip_file.unlink()

            if not self.args.no_usage and executor_name:
                scheme_prefix = 'jinaai' if scheme.startswith('jinaai') else 'jinahub'
                self._prettyprint_usage(
                    console, scheme_prefix=scheme_prefix, executor_name=executor_name
                )
