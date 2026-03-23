import importlib
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from marie.connectors.model import ConnectorManifest, ConnectorsConf
from marie.logging_core.predefined import default_logger as logger


class ConnectorRegistry:
    """Registry for connector manifests with auto-discovery."""

    _connectors: Dict[str, ConnectorManifest] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, manifest: ConnectorManifest, import_modules: bool = True) -> None:
        """Store manifest and optionally import backend modules to trigger existing decorators."""
        if manifest.id in cls._connectors:
            logger.warning(f"Connector '{manifest.id}' already registered, skipping")
            return

        if import_modules and manifest.package_name:
            cls._import_backend_modules(manifest)

        cls._connectors[manifest.id] = manifest
        logger.info(
            f"Registered connector: {manifest.id} (source: {manifest.source_type})"
        )

    @classmethod
    def _import_backend_modules(cls, manifest: ConnectorManifest) -> None:
        """Import executor + query_definition modules.

        This triggers @requests and @QueryTypeRegistry.register via side effects.
        """
        pkg = manifest.package_name
        for rel_module in [
            manifest.backend.query_definition_module,
            manifest.backend.executor_module,
        ]:
            abs_module = cls._resolve_module(pkg, rel_module)
            try:
                importlib.import_module(abs_module)
                logger.info(f"Imported {abs_module}")
            except Exception as e:
                logger.error(f"Failed to import {abs_module}: {e}")
                warnings.warn(f"Failed to import connector module {abs_module}: {e}")

        for cred in manifest.backend.credentials:
            abs_module = cls._resolve_module(pkg, cred.module)
            try:
                importlib.import_module(abs_module)
            except Exception as e:
                logger.warning(f"Failed to import credential module {abs_module}: {e}")

    @classmethod
    def _resolve_module(cls, package_name: str, relative_path: str) -> str:
        """Resolve '.query_definition' against 'marie.connectors.bitly'
        -> 'marie.connectors.bitly.query_definition'.
        """
        if relative_path.startswith("."):
            return package_name + relative_path
        return relative_path

    @classmethod
    def register_from_yaml(
        cls,
        yaml_path: str,
        package_name: str,
        source_type: str = "builtin",
    ) -> None:
        """Parse connector.yaml and register."""
        manifest = ConnectorManifest.from_yaml_file(
            yaml_path, package_name=package_name
        )
        manifest.source_type = source_type
        cls.register(manifest)

    @classmethod
    def discover_builtin(cls, package_name: str, pattern: str = "*") -> Dict[str, Any]:
        """Tier 1: Scan subpackages for connector.yaml files."""
        from fnmatch import fnmatch

        result: Dict[str, Any] = {"loaded": [], "failed": [], "skipped": []}

        logger.info(
            f"Discovering connectors from '{package_name}' (pattern: {pattern})"
        )
        try:
            pkg = importlib.import_module(package_name)
            if not hasattr(pkg, "__file__") or pkg.__file__ is None:
                result["error"] = f"Package '{package_name}' has no __file__"
                return result

            pkg_path = Path(pkg.__file__).parent
            for item in sorted(pkg_path.iterdir()):
                if not item.is_dir() or item.name.startswith("_"):
                    continue
                if not fnmatch(item.name, pattern):
                    result["skipped"].append(item.name)
                    continue

                manifest_path = item / "connector.yaml"
                if not manifest_path.exists():
                    result["skipped"].append(f"{item.name} (no connector.yaml)")
                    continue

                sub_package = f"{package_name}.{item.name}"
                try:
                    cls.register_from_yaml(
                        str(manifest_path), sub_package, source_type="builtin"
                    )
                    result["loaded"].append(sub_package)
                except Exception as e:
                    logger.error(
                        f"Failed to register connector from {manifest_path}: {e}"
                    )
                    result["failed"].append((sub_package, str(e)))

        except ImportError as e:
            logger.error(f"Failed to import package '{package_name}': {e}")
            result["error"] = str(e)

        logger.info(
            f"Discovered {len(result['loaded'])} connectors from '{package_name}'"
        )
        return result

    @classmethod
    def discover_thirdparty(cls) -> Dict[str, Any]:
        """Tier 2: Scan installed marie-connector-* pip packages."""
        from importlib.metadata import distributions

        result: Dict[str, Any] = {"loaded": [], "failed": []}

        for dist in distributions():
            name = dist.metadata["Name"] or ""
            if not name.startswith("marie-connector-"):
                continue

            top_level = name.replace("-", "_")
            try:
                pkg = importlib.import_module(top_level)
                if pkg.__file__:
                    manifest_path = Path(pkg.__file__).parent / "connector.yaml"
                    if manifest_path.exists():
                        cls.register_from_yaml(
                            str(manifest_path),
                            top_level,
                            source_type="thirdparty",
                        )
                        result["loaded"].append(top_level)
                    else:
                        result["failed"].append((top_level, "No connector.yaml found"))
            except Exception as e:
                result["failed"].append((top_level, str(e)))

        logger.info(f"Discovered {len(result['loaded'])} third-party connectors")
        return result

    @classmethod
    def initialize_from_config(cls, conf: ConnectorsConf) -> Dict[str, Any]:
        """Called at startup: runs all discovery tiers."""
        logger.info("Initializing connectors from configuration")
        result: Dict[str, Any] = {
            "loaded": [],
            "failed": [],
            "discovered": {},
        }

        # Discover from configured packages
        for dp in conf.discover_packages:
            dr = cls.discover_builtin(dp.package, dp.pattern)
            result["discovered"][dp.package] = dr
            result["loaded"].extend(dr.get("loaded", []))
            result["failed"].extend(dr.get("failed", []))

        # Discover installed third-party packages
        if conf.discover_installed:
            tr = cls.discover_thirdparty()
            result["loaded"].extend(tr.get("loaded", []))
            result["failed"].extend(tr.get("failed", []))

        # Explicit module registrations — require connector.yaml in the module's package
        for conn in conf.connectors:
            try:
                pkg = importlib.import_module(conn.py_module)
                if not pkg.__file__:
                    result["failed"].append((conn.py_module, "Package has no __file__"))
                    continue

                manifest_path = Path(pkg.__file__).parent / "connector.yaml"
                if manifest_path.exists():
                    count_before = len(cls._connectors)
                    cls.register_from_yaml(
                        str(manifest_path),
                        conn.py_module,
                        source_type="builtin",
                    )
                    if len(cls._connectors) > count_before:
                        result["loaded"].append(conn.py_module)
                    else:
                        result["failed"].append(
                            (
                                conn.py_module,
                                "connector.yaml parsed but no new connector registered (duplicate?)",
                            )
                        )
                else:
                    result["failed"].append(
                        (
                            conn.py_module,
                            f"No connector.yaml found in {Path(pkg.__file__).parent}",
                        )
                    )
            except Exception as e:
                result["failed"].append((conn.py_module, str(e)))

        result["total"] = len(cls._connectors)
        cls._initialized = True
        logger.info(
            f"Connector initialization complete: {result['total']} connectors registered"
        )
        return result

    @classmethod
    def get(cls, connector_id: str) -> Optional[ConnectorManifest]:
        return cls._connectors.get(connector_id)

    @classmethod
    def list_connectors(cls) -> List[str]:
        return list(cls._connectors.keys())

    @classmethod
    def list_connectors_with_metadata(cls) -> List[Dict[str, Any]]:
        """Full connector metadata for API responses.

        Includes everything needed to build a NodeSpec.
        """
        result = []
        for manifest in cls._connectors.values():
            result.append(
                {
                    "id": manifest.id,
                    "version": manifest.version,
                    "display_name": manifest.display_name,
                    "description": manifest.description,
                    "icon": manifest.icon,
                    "category": manifest.category,
                    "tags": manifest.tags,
                    "source_type": manifest.source_type,
                    "resources": [
                        {
                            "name": r.name,
                            "display_name": r.display_name,
                            "operations": [
                                {
                                    "name": op.name,
                                    "display_name": op.display_name,
                                }
                                for op in r.operations
                            ],
                        }
                        for r in manifest.resources
                    ],
                    "credentials": [
                        {"name": c.name, "auth_type": c.auth_type}
                        for c in manifest.backend.credentials
                    ],
                    "runner": {
                        "op": manifest.runner.op,
                        "timeout_sec": manifest.runner.timeout_sec,
                        "max_retries": manifest.runner.max_retries,
                    },
                }
            )
        return result

    @classmethod
    def get_registry_info(cls) -> Dict[str, Any]:
        return {
            "total_connectors": len(cls._connectors),
            "connector_names": cls.list_connectors(),
            "initialized": cls._initialized,
        }

    @classmethod
    def unregister(cls, connector_id: str) -> bool:
        """Remove a connector from the registry.

        Only deployed and thirdparty connectors may be unregistered.
        Returns True if the connector was found and removed.
        """
        manifest = cls._connectors.get(connector_id)
        if manifest is None:
            return False
        if manifest.source_type == "builtin":
            raise ValueError(f"Cannot unregister builtin connector '{connector_id}'")
        del cls._connectors[connector_id]
        logger.info(f"Unregistered connector: {connector_id}")
        return True

    @classmethod
    def register_from_bundle(
        cls,
        target_dir: str,
        files: Dict[str, str],
        source_type: str = "deployed",
        overwrite: bool = False,
    ) -> ConnectorManifest:
        """Write a connector file bundle to *target_dir* and register it.

        The bundle is written atomically via a temp directory. The
        ``connector.yaml`` entry in *files* is mandatory and is parsed to
        validate the manifest before any files are written.

        Raises ``ValueError`` if the bundle is invalid, ``FileExistsError``
        if *target_dir* already exists and *overwrite* is False.
        """
        import yaml

        if "connector.yaml" not in files:
            raise ValueError("Bundle must include connector.yaml")

        # Validate path safety — reject traversal attempts
        for rel_path in files:
            normalized = os.path.normpath(rel_path)
            if normalized.startswith("..") or os.path.isabs(normalized):
                raise ValueError(f"Invalid path in bundle: {rel_path}")

        # Parse manifest to validate before writing anything
        manifest_data = yaml.safe_load(files["connector.yaml"])
        manifest = ConnectorManifest(**manifest_data)

        if not overwrite and os.path.exists(target_dir):
            raise FileExistsError(f"Connector directory already exists: {target_dir}")

        # Write to temp dir, then atomic move
        tmp = tempfile.mkdtemp(prefix="marie-connector-deploy-")
        try:
            for rel_path, content in files.items():
                abs_path = os.path.join(tmp, rel_path)
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                with open(abs_path, "w") as f:
                    f.write(content)

            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            shutil.move(tmp, target_dir)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

        package_name = cls._infer_package_name(target_dir)
        manifest.source_type = source_type
        manifest.package_name = package_name

        # If already registered (overwrite case), remove old entry first
        cls._connectors.pop(manifest.id, None)
        cls.register(manifest, import_modules=True)

        return manifest

    @classmethod
    def _infer_package_name(cls, target_dir: str) -> str:
        """Infer dotted package name from a filesystem path.

        Walks up from *target_dir* looking for ``__init__.py`` markers to
        build the full dotted package name.  For example::

            /home/user/marie-ai/marie/connectors/bitly
            →  marie.connectors.bitly
        """
        parts: List[str] = []
        current = Path(target_dir).resolve()
        while True:
            parts.append(current.name)
            parent = current.parent
            if parent == current:
                break
            if not (parent / "__init__.py").exists():
                break
            current = parent
        parts.reverse()
        return ".".join(parts)

    @classmethod
    def cleanup(cls) -> None:
        cls._connectors.clear()
        cls._initialized = False
