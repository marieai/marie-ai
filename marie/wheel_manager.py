import importlib
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from marie.logging_core.predefined import default_logger as logger


class WheelInstallationCallback(Protocol):
    """Protocol for wheel installation callbacks"""

    def on_wheel_installed(self, wheel_info: Dict[str, Any]) -> None:
        """Called when a wheel is successfully installed"""
        ...

    def on_wheel_uninstalled(self, package_name: str) -> None:
        """Called when a wheel is successfully uninstalled"""
        ...

    def on_wheel_error(self, wheel_path: str, error: Exception) -> None:
        """Called when a wheel installation/uninstallation fails"""
        ...


class PipWheelManager:
    """Manages Python wheel installation and uninstallation using pip"""

    def __init__(self):
        self.installed_wheels: Dict[str, Dict[str, Any]] = {}
        self.wheel_to_package_map: Dict[str, str] = {}
        self.package_to_modules_map: Dict[str, List[str]] = {}
        self.callbacks: List[WheelInstallationCallback] = []

    def add_callback(self, callback: WheelInstallationCallback) -> None:
        """Add a callback for wheel installation events"""
        self.callbacks.append(callback)

    def remove_callback(self, callback: WheelInstallationCallback) -> None:
        """Remove a callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def _notify_installed(self, wheel_info: Dict[str, Any]) -> None:
        """Notify callbacks of successful installation"""
        for callback in self.callbacks:
            try:
                callback.on_wheel_installed(wheel_info)
            except Exception as e:
                logger.error(f"Error in wheel installation callback: {e}")

    def _notify_uninstalled(self, package_name: str) -> None:
        """Notify callbacks of successful uninstallation"""
        for callback in self.callbacks:
            try:
                callback.on_wheel_uninstalled(package_name)
            except Exception as e:
                logger.error(f"Error in wheel uninstallation callback: {e}")

    def _notify_error(self, wheel_path: str, error: Exception) -> None:
        """Notify callbacks of installation/uninstallation errors"""
        for callback in self.callbacks:
            try:
                callback.on_wheel_error(wheel_path, error)
            except Exception as e:
                logger.error(f"Error in wheel error callback: {e}")

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
            logger.error(f"Pip command timed out: {' '.join(cmd)}")
            raise
        except Exception as e:
            logger.error(f"Failed to run pip command {' '.join(cmd)}: {e}")
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
            result = self._run_pip_command(['show', package_name])
            if result.returncode == 0:
                info = {}
                for line in result.stdout.splitlines():
                    if ':' in line:
                        key, value = line.split(':', 1)
                        info[key.strip().lower()] = value.strip()
                return info
        except Exception as e:
            logger.debug(f"Could not get package info for {package_name}: {e}")
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
            logger.debug(f"Could not discover modules for {package_name}: {e}")
        except Exception as e:
            logger.warning(f"Error discovering modules for {package_name}: {e}")

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
                logger.debug(f"Wheel {wheel_name} already installed and up-to-date")
                return installed_info

        try:
            # Prepare pip install command
            pip_args = ['install', '--force-reinstall', '--no-deps', wheel_path]

            logger.info(f"Installing wheel: {wheel_name}")
            result = self._run_pip_command(pip_args)

            if result.returncode != 0:
                error_msg = f"Failed to install wheel {wheel_name}: {result.stderr}"
                logger.error(error_msg)
                error = RuntimeError(error_msg)
                self._notify_error(wheel_path, error)
                raise error

            # Get package information
            package_info = self._get_installed_package_info(package_name)

            # Discover modules
            modules = self._discover_package_modules(package_name)

            # Force reload of modules
            for module_name in modules:
                if module_name in sys.modules:
                    importlib.reload(sys.modules[module_name])
                    logger.debug(f"Reloaded module: {module_name}")
                else:
                    try:
                        importlib.import_module(module_name)
                        logger.debug(f"Imported module: {module_name}")
                    except ImportError as e:
                        logger.debug(f"Could not import module {module_name}: {e}")

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

            logger.info(
                f"Successfully installed wheel {wheel_name} -> package {package_name}"
            )
            logger.info(
                f"Discovered {len(modules)} modules: {modules[:5]}{'...' if len(modules) > 5 else ''}"
            )

            # Notify callbacks
            self._notify_installed(wheel_info)

            return wheel_info

        except Exception as e:
            logger.error(f"Failed to install wheel {wheel_path}: {e}")
            self._notify_error(wheel_path, e)
            raise

    def uninstall_package(self, package_name: str) -> bool:
        """Uninstall a package using pip"""
        try:
            logger.info(f"Uninstalling package: {package_name}")
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
                            logger.debug(
                                f"Removed module from sys.modules: {module_name}"
                            )
                    del self.package_to_modules_map[package_name]

                logger.info(f"Successfully uninstalled package: {package_name}")

                # Notify callbacks
                self._notify_uninstalled(package_name)

                return True
            else:
                logger.error(
                    f"Failed to uninstall package {package_name}: {result.stderr}"
                )
                return False

        except Exception as e:
            logger.error(f"Error uninstalling package {package_name}: {e}")
            return False

    def get_installed_wheels(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all installed wheels"""
        return self.installed_wheels.copy()

    def get_installed_packages(self) -> List[str]:
        """Get list of installed package names"""
        return list(self.package_to_modules_map.keys())

    def cleanup(self):
        """Clean up by uninstalling all tracked packages"""
        packages_to_uninstall = list(self.package_to_modules_map.keys())
        for package_name in packages_to_uninstall:
            try:
                self.uninstall_package(package_name)
            except Exception as e:
                logger.error(f"Error during cleanup of package {package_name}: {e}")


class WheelDirectoryWatcher:
    """Watches directories for wheel file changes and manages installations"""

    def __init__(self, wheel_manager: PipWheelManager):
        self.wheel_manager = wheel_manager
        self.observer = None
        self.watched_directories: Set[str] = set()
        self.event_handler = WheelFileHandler(wheel_manager)

    def watch_directory(self, directory: str) -> None:
        """Start watching a directory for wheel changes"""
        if not os.path.exists(directory):
            raise ValueError(f"Directory does not exist: {directory}")

        if not self.observer:
            self.observer = Observer()
            self.observer.start()

        if directory not in self.watched_directories:
            self.observer.schedule(self.event_handler, directory, recursive=False)
            self.watched_directories.add(directory)
            logger.info(f"Started watching wheel directory: {directory}")

    def stop_watching(self, directory: Optional[str] = None) -> None:
        """Stop watching a specific directory or all directories"""
        if self.observer:
            if directory:
                # Remove specific directory (would need more complex implementation)
                self.watched_directories.discard(directory)
            else:
                # Stop all watching
                self.observer.stop()
                self.observer.join()
                self.observer = None
                self.watched_directories.clear()
                logger.info("Stopped all wheel directory watching")

    def install_existing_wheels(self, directory: str) -> Dict[str, Any]:
        """Install all existing wheels in a directory"""
        if not os.path.exists(directory):
            raise ValueError(f"Directory does not exist: {directory}")

        results = {'installed_wheels': [], 'failed_wheels': []}

        for file in os.listdir(directory):
            if file.endswith('.whl'):
                wheel_path = os.path.join(directory, file)
                try:
                    wheel_info = self.wheel_manager.install_wheel(wheel_path)
                    results['installed_wheels'].append(
                        {
                            'name': file,
                            'path': wheel_path,
                            'package_name': wheel_info['package_name'],
                            'modules': wheel_info['modules'],
                        }
                    )
                except Exception as e:
                    results['failed_wheels'].append(
                        {'name': file, 'path': wheel_path, 'error': str(e)}
                    )
        logger.info(
            f"Installed {len(results['installed_wheels'])} wheels from {directory}"
        )
        return results


class WheelFileHandler(FileSystemEventHandler):
    """File system event handler for wheel files"""

    def __init__(self, wheel_manager: PipWheelManager):
        self.wheel_manager = wheel_manager
        self.debounce_time = 2.0
        self.pending_events = {}
        self.lock = threading.Lock()

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.whl'):
            self._schedule_install(event.src_path, 'created')

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.whl'):
            self._schedule_install(event.src_path, 'modified')

    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith('.whl'):
            self._schedule_uninstall(event.src_path)

    def _schedule_install(self, wheel_path: str, event_type: str):
        """Schedule a wheel installation with debouncing"""
        with self.lock:
            # Cancel any pending install for this file
            if wheel_path in self.pending_events:
                self.pending_events[wheel_path].cancel()

            # Schedule new install
            timer = threading.Timer(
                self.debounce_time, self._install_wheel, [wheel_path, event_type]
            )
            self.pending_events[wheel_path] = timer
            timer.start()

    def _schedule_uninstall(self, wheel_path: str):
        """Schedule uninstalling a wheel"""
        wheel_name = Path(wheel_path).name

        with self.lock:
            # Cancel any pending install
            if wheel_path in self.pending_events:
                self.pending_events[wheel_path].cancel()
                del self.pending_events[wheel_path]

        # Find package name and uninstall
        package_name = self.wheel_manager.wheel_to_package_map.get(wheel_name)
        if package_name:
            self.wheel_manager.uninstall_package(package_name)
            logger.info(
                f"Wheel deleted and package uninstalled: {wheel_name} -> {package_name}"
            )

    def _install_wheel(self, wheel_path: str, event_type: str):
        """Install a wheel file using pip"""
        with self.lock:
            if wheel_path in self.pending_events:
                del self.pending_events[wheel_path]

        if os.path.exists(wheel_path):
            try:
                # Install the wheel
                wheel_info = self.wheel_manager.install_wheel(
                    wheel_path, force_reinstall=True
                )

                logger.info(
                    f"Hot reloaded wheel via pip: {Path(wheel_path).name} ({event_type})"
                )
                logger.info(
                    f"Package: {wheel_info['package_name']}, Modules: {len(wheel_info['modules'])}"
                )

            except Exception as e:
                logger.error(f"Failed to install wheel {wheel_path}: {e}")


if __name__ == "__main__":
    # Example usage
    wheel_manager = PipWheelManager()

    # Install a specific wheel
    # wheel_info = wheel_manager.install_wheel(
    #     '/home/gbugaj/dev/hello_world_wheel/dist/hello_world_wheel-0.0.post1.dev1+g2164318-py3-none-any.whl')
    #
    # print(f"Installed: {wheel_info['package_name']}")
    # print(wheel_info)

    if True:
        # using a wheel watcher
        watcher = WheelDirectoryWatcher(wheel_manager)
        # Watch a directory for wheel changes
        try:
            watcher.watch_directory('/home/gbugaj/dev/hello_world_wheel/dist')
            print("Watching for wheel changes...")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            watcher.stop_watching()
            wheel_manager.cleanup()
            print("Stopped watching and cleaned up.")
