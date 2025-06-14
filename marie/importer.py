import importlib
import logging
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from marie.constants import __resources_path__

IMPORTED = SimpleNamespace()
IMPORTED.executors = False
IMPORTED.schema_executors = {}


# GB:MOD
class ImportExtensions:
    """
    A context manager for wrapping extension import and fallback. It guides the user to pip install correct package by looking up extra-requirements.txt.

    :param required: set to True if you want to raise the ModuleNotFound error
    :param logger: when not given, built-in warnings.warn will be used
    :param help_text: the help text followed after
    :param pkg_name: the package name to find in extra_requirements.txt, when not given the ModuleNotFound exec_val will be used as the best guess
    """

    def __init__(
        self,
        required: bool,
        logger=None,
        help_text: Optional[str] = None,
        pkg_name: Optional[str] = None,
        verbose: bool = True,
    ):
        self._required = required
        self._tags = []
        self._help_text = help_text
        self._logger = logger
        self._pkg_name = pkg_name
        self._verbose = verbose

    def __enter__(self):
        return self

    def _check_v(self, v, missing_module):
        if (
            v.strip()
            and not v.startswith('#')
            and v.startswith(missing_module)
            and ':' in v
        ):
            return True

    def _find_missing_module_in_extra_req(self, missing_module):
        with open(
            os.path.join(__resources_path__, 'extra-requirements.txt'), encoding='utf-8'
        ) as fp:
            for v in fp:
                if self._check_v(v, missing_module):
                    missing_module, install_tags = v.split(':')
                    self._tags.append(missing_module)
                    self._tags.extend(vv.strip() for vv in install_tags.split(','))
                    break

    def _find_missing_module(self, exc_val):
        missing_module = self._pkg_name or exc_val.name
        missing_module = self._find_missing_module_in_extra_req(missing_module)
        return missing_module

    def _err_msg(self, exc_val, missing_module):
        if self._tags:
            from marie.helper import colored

            req_msg = colored('fallback to default behavior', color='yellow')
            if self._required:
                req_msg = colored('and it is required', color='red')
            err_msg = f'''Python package "{colored(missing_module, attrs='bold')}" is not installed, {req_msg}.
            You are trying to use a feature not enabled by your current Jina installation.'''

            avail_tags = ' '.join(
                colored(f'[{tag}]', attrs='bold') for tag in self._tags
            )
            err_msg += (
                f'\n\nTo enable this feature, use {colored("pip install jina[TAG]", attrs="bold")}, '
                f'where {colored("[TAG]", attrs="bold")} is one of {avail_tags}.\n'
            )
        else:
            err_msg = f'{exc_val.msg}'
        return err_msg

    def _log_critical(self, err_msg):
        if self._verbose and self._logger:
            self._logger.critical(err_msg)
            if self._help_text:
                self._logger.error(self._help_text)

    def _log_warning(self, err_msg):
        if self._verbose and self._logger:
            self._logger.warning(err_msg)
            if self._help_text:
                self._logger.info(self._help_text)

    def _raise_or_supress(self, err_msg, exc_val):
        if self._verbose and not self._logger:
            warnings.warn(err_msg, RuntimeWarning, stacklevel=2)
        if self._required:
            self._log_critical(err_msg)
            raise exc_val
        else:
            self._log_warning(err_msg)
            return True  # suppress the error

    def __exit__(self, exc_type, exc_val, traceback):
        if exc_type != ModuleNotFoundError:
            return
        missing_module = self._find_missing_module(exc_val)
        err_msg = self._err_msg(exc_val, missing_module)
        return self._raise_or_supress(err_msg, exc_val)


def _path_import(absolute_path: str):
    import importlib.util

    try:
        # I dont want to trust user path based on directory structure, "user_module", period
        default_spec_name = 'user_module'
        user_module_name = os.path.splitext(os.path.basename(absolute_path))[0]
        if user_module_name == '__init__':
            # __init__ can not be used as a module name
            spec_name = default_spec_name
        elif user_module_name not in sys.modules:
            spec_name = user_module_name
        else:
            warnings.warn(
                f'''
            {user_module_name} shadows one of built-in Python module name.
            It is imported as `{default_spec_name}.{user_module_name}`

            Affects:
            - Either, change your code from using `from {user_module_name} import ...`
              to `from {default_spec_name}.{user_module_name} import ...`
            - Or, rename {user_module_name} to another name
            '''
            )
            spec_name = f'{default_spec_name}.{user_module_name}'

        spec = importlib.util.spec_from_file_location(spec_name, absolute_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec_name] = module
        spec.loader.exec_module(module)
    except Exception as ex:
        raise ImportError(f'can not import module from {absolute_path}') from ex


class PathImporter:
    """The class to import modules from paths."""

    @staticmethod
    def add_modules(*paths):
        """
        Import modules from paths.

        :param paths: Paths of the modules.
        """

        # assume paths are Python module names
        not_python_module_paths = []
        for path in paths:
            if not os.path.isfile(path):
                try:
                    importlib.import_module(path)
                except ModuleNotFoundError as e:
                    if e.msg != f"No module named '{path}'":
                        raise e
                    not_python_module_paths.append(path)
                except:
                    raise
            else:
                not_python_module_paths.append(path)

        # try again, but assume they are file paths instead of module names
        from marie.jaml.helper import complete_path

        for m in not_python_module_paths:
            _path_import(complete_path(m))


class PipWheelLoader:
    """Handles loading and reloading of Python wheels using pip"""

    def __init__(self):
        self.installed_wheels: Dict[str, Dict[str, Any]] = {}
        self.wheel_to_package_map: Dict[str, str] = {}
        self.package_to_modules_map: Dict[str, List[str]] = {}

    def _run_pip_command(
        self, args: List[str], capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """Run a pip command safely"""
        cmd = [sys.executable, '-m', 'pip'] + args
        try:
            result = subprocess.run(
                cmd, capture_output=capture_output, text=True, check=False, timeout=60
            )
            return result
        except subprocess.TimeoutExpired:
            logging.error(f"Pip command timed out: {' '.join(cmd)}")
            raise
        except Exception as e:
            logging.error(f"Failed to run pip command {' '.join(cmd)}: {e}")
            raise

    def _extract_package_name_from_wheel(self, wheel_path: str) -> str:
        """Extract package name from wheel filename"""
        wheel_name = Path(wheel_path).name
        # Wheel filename format: {name}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl
        parts = wheel_name.split('-')
        if len(parts) >= 2:
            return parts[0].replace('_', '-')
        return wheel_name.replace('.whl', '')

    def _get_installed_package_info(
        self, package_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get information about an installed package"""
        try:
            result = self._run_pip_command(['show', '--format', 'json', package_name])
            if result.returncode == 0 and result.stdout.strip():
                # pip show with --format json is not universally supported
                # Fall back to regular pip show
                result = self._run_pip_command(['show', package_name])
                if result.returncode == 0:
                    info = {}
                    for line in result.stdout.splitlines():
                        if ':' in line:
                            key, value = line.split(':', 1)
                            info[key.strip().lower()] = value.strip()
                    return info
        except Exception as e:
            logging.debug(f"Could not get package info for {package_name}: {e}")
        return None

    def _discover_package_modules(self, package_name: str) -> List[str]:
        """Discover modules provided by a package"""
        modules = []
        try:
            # Try to import the package and discover its modules
            package = importlib.import_module(package_name)
            if hasattr(package, '__path__'):
                # It's a package, look for submodules
                package_path = package.__path__[0]
                for root, dirs, files in os.walk(package_path):
                    # Skip __pycache__ directories
                    dirs[:] = [d for d in dirs if d != '__pycache__']

                    for file in files:
                        if file.endswith('.py') and not file.startswith('__'):
                            rel_path = os.path.relpath(
                                os.path.join(root, file), package_path
                            )
                            module_name = rel_path.replace(os.sep, '.').replace(
                                '.py', ''
                            )
                            full_module_name = f"{package_name}.{module_name}"
                            modules.append(full_module_name)
            else:
                # It's a single module
                modules.append(package_name)

        except ImportError as e:
            logging.debug(f"Could not discover modules for {package_name}: {e}")
        except Exception as e:
            logging.warning(f"Error discovering modules for {package_name}: {e}")

        return modules

    def install_wheel(
        self, wheel_path: str, force_reinstall: bool = False
    ) -> Dict[str, Any]:
        """Install a wheel using pip"""
        if not os.path.exists(wheel_path):
            raise FileNotFoundError(f"Wheel file not found: {wheel_path}")

        wheel_name = Path(wheel_path).name
        package_name = self._extract_package_name_from_wheel(wheel_path)
        wheel_stat = os.stat(wheel_path)

        # Check if already installed and up-to-date
        if not force_reinstall and wheel_name in self.installed_wheels:
            installed_info = self.installed_wheels[wheel_name]
            if installed_info['mtime'] >= wheel_stat.st_mtime:
                logging.debug(f"Wheel {wheel_name} already installed and up-to-date")
                return installed_info

        try:
            # Prepare pip install command
            pip_args = ['install', '--force-reinstall', '--no-deps', wheel_path]

            logging.info(f"Installing wheel: {wheel_name}")
            result = self._run_pip_command(pip_args)

            if result.returncode != 0:
                error_msg = f"Failed to install wheel {wheel_name}: {result.stderr}"
                logging.error(error_msg)
                raise RuntimeError(error_msg)

            # Get package information
            package_info = self._get_installed_package_info(package_name)

            # Discover modules
            modules = self._discover_package_modules(package_name)

            # Force reload of modules
            for module_name in modules:
                if module_name in sys.modules:
                    importlib.reload(sys.modules[module_name])
                    logging.debug(f"Reloaded module: {module_name}")
                else:
                    try:
                        importlib.import_module(module_name)
                        logging.debug(f"Imported module: {module_name}")
                    except ImportError as e:
                        logging.debug(f"Could not import module {module_name}: {e}")

            wheel_info = {
                'wheel_path': wheel_path,
                'package_name': package_name,
                'mtime': wheel_stat.st_mtime,
                'install_time': time.time(),
                'package_info': package_info,
                'modules': modules,
                'pip_output': result.stdout,
            }

            self.installed_wheels[wheel_name] = wheel_info
            self.wheel_to_package_map[wheel_name] = package_name
            self.package_to_modules_map[package_name] = modules

            logging.info(
                f"Successfully installed wheel {wheel_name} -> package {package_name}"
            )
            logging.info(
                f"Discovered {len(modules)} modules: {modules[:5]}{'...' if len(modules) > 5 else ''}"
            )

            return wheel_info

        except Exception as e:
            logging.error(f"Failed to install wheel {wheel_path}: {e}")
            raise

    def uninstall_package(self, package_name: str) -> bool:
        """Uninstall a package using pip"""
        try:
            logging.info(f"Uninstalling package: {package_name}")
            result = self._run_pip_command(['uninstall', '-y', package_name])

            if result.returncode == 0:
                # Remove from our tracking
                wheels_to_remove = [
                    wheel
                    for wheel, pkg in self.wheel_to_package_map.items()
                    if pkg == package_name
                ]

                for wheel_name in wheels_to_remove:
                    if wheel_name in self.installed_wheels:
                        del self.installed_wheels[wheel_name]
                    if wheel_name in self.wheel_to_package_map:
                        del self.wheel_to_package_map[wheel_name]

                if package_name in self.package_to_modules_map:
                    # Remove modules from sys.modules
                    modules = self.package_to_modules_map[package_name]
                    for module_name in modules:
                        if module_name in sys.modules:
                            del sys.modules[module_name]
                            logging.debug(
                                f"Removed module from sys.modules: {module_name}"
                            )
                    del self.package_to_modules_map[package_name]

                logging.info(f"Successfully uninstalled package: {package_name}")
                return True
            else:
                logging.error(
                    f"Failed to uninstall package {package_name}: {result.stderr}"
                )
                return False

        except Exception as e:
            logging.error(f"Error uninstalling package {package_name}: {e}")
            return False

    def get_installed_wheels(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all installed wheels"""
        return self.installed_wheels.copy()

    def cleanup(self):
        """Clean up by uninstalling all tracked packages"""
        packages_to_uninstall = list(self.package_to_modules_map.keys())
        for package_name in packages_to_uninstall:
            try:
                self.uninstall_package(package_name)
            except Exception as e:
                logging.error(f"Error during cleanup of package {package_name}: {e}")
