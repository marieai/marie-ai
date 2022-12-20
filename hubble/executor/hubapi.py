"""Module wrapping interactions with the local executor packages."""

import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml
from hubble.executor import HubExecutor
from hubble.executor.helper import (
    get_hub_packages_dir,
    install_requirements,
    is_requirements_installed,
    unpack_package,
)

SECRET_PATH = 'secrets'


def get_dist_path(uuid: str, tag: str) -> Tuple[Path, Path]:
    """Get the package path according ID and TAG
    :param uuid: the UUID of the executor
    :param tag: the TAG of the executor
    :return: package and its dist-info path
    """
    pkg_path = get_hub_packages_dir() / uuid
    pkg_dist_path = pkg_path / f'{tag}.dist-info'
    return pkg_path, pkg_dist_path


def get_dist_path_of_executor(executor: 'HubExecutor') -> Tuple[Path, Path]:
    """Return the path of the executor if available.

    :param executor: the executor to check
    :return: the path of the executor package
    """

    pkg_path, pkg_dist_path = get_dist_path(executor.uuid, executor.tag)

    if not pkg_path.exists():
        raise FileNotFoundError(f'{pkg_path} does not exist')
    elif not pkg_dist_path.exists():
        raise FileNotFoundError(f'{pkg_dist_path} does not exist')
    else:
        return pkg_path, pkg_dist_path


def get_lockfile() -> str:
    """Get the path of file locker
    :return: the path of file locker
    """
    return str(get_hub_packages_dir() / 'LOCK')


def install_local(
    zip_package: 'Path',
    executor: 'HubExecutor',
    install_deps: bool = False,
):
    """Install the package in zip format to the Jina Hub root.

    :param zip_package: the path of the zip file
    :param executor: the executor to install
    :param install_deps: if set, install dependencies
    """

    pkg_path, pkg_dist_path = get_dist_path(executor.uuid, executor.tag)

    # clean the existed dist_path
    for dist in pkg_path.glob('*.dist-info'):
        shutil.rmtree(dist)

    # unpack the zip package to the root pkg_path
    unpack_package(zip_package, pkg_path)

    # create dist-info folder
    pkg_dist_path.mkdir(parents=False, exist_ok=True)

    install_package_dependencies(install_deps, pkg_dist_path, pkg_path)

    manifest_path = pkg_path / 'manifest.yml'
    if manifest_path.exists():
        shutil.copyfile(manifest_path, pkg_dist_path / 'manifest.yml')

    # store the commit id in local
    if executor.commit_id is not None:
        commit_file = pkg_dist_path / f'PKG-COMMIT-{executor.commit_id}'
        commit_file.touch()


def install_package_dependencies(
    install_deps: bool, pkg_dist_path: 'Path', pkg_path: 'Path'
) -> None:
    """

    :param install_deps: if set, then install dependencies
    :param pkg_dist_path: package distribution path
    :param pkg_path: package path
    """
    # install the dependencies included in requirements.txt
    requirements_file = pkg_path / 'requirements.txt'

    if requirements_file.exists():
        if pkg_path != pkg_dist_path:
            shutil.copyfile(requirements_file, pkg_dist_path / 'requirements.txt')

        if install_deps:
            install_requirements(requirements_file)
        elif not is_requirements_installed(requirements_file, show_warning=True):
            raise ModuleNotFoundError(
                'Dependencies listed in requirements.txt are not all installed locally, '
                'this Executor may not run as expect. To install dependencies, '
                'add `--install-requirements` or set `install_requirements = True`'
            )


def uninstall_local(uuid: str):
    """Uninstall the executor package.

    :param uuid: the UUID of the executor
    """
    pkg_path, _ = get_dist_path(uuid, None)
    for dist in get_hub_packages_dir().glob(f'{uuid}/*.dist-info'):
        shutil.rmtree(dist)
    if pkg_path.exists():
        shutil.rmtree(pkg_path)


def list_local():
    """List the locally-available executor packages.

    :return: the list of local executors (if found)
    """
    result = []
    for dist_name in get_hub_packages_dir().glob(r'*/*.dist-info'):
        result.append(dist_name)

    return result


def exist_local(uuid: str, tag: str = None) -> bool:
    """Check whether the executor exists in local

    :param uuid: the UUID of the executor
    :param tag: the TAG of the executor
    :return: True if existed, else False
    """
    try:
        get_dist_path_of_executor(HubExecutor(uuid=uuid, tag=tag))
        return True
    except FileNotFoundError:
        return False


def load_config(path: Path) -> Dict:
    """Load config of executor from YAML file.

    :param path: the path of the local executor
    :return: dict
    """
    with open(path / 'config.yml') as fp:
        tmp = yaml.safe_load(fp)

    return tmp


def extract_executor_name(path: Path) -> Optional[str]:
    """Extract the executor name from the config.yaml (or manifest.yml).

    :param path: the path of the local executor
    :return: the name of the executor
    """

    name = None

    if (path / 'config.yml').exists():
        with open(path / 'config.yml') as fp:
            tmp = yaml.safe_load(fp)
            name = tmp.get('metas', {}).get('name')

    if not name and (path / 'manifest.yml').exists():
        with open(path / 'manifest.yml') as fp:
            tmp = yaml.safe_load(fp)
            name = tmp.get('name')

    return name
