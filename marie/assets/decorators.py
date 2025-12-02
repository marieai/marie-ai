"""
Asset decorators for Marie-AI.

Provides @asset, @multi_asset, and other decorators for defining assets
with metadata, dependencies, and configuration.
"""

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from marie.assets.models import AssetKind, AssetSpec


def asset(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    key: Optional[str] = None,
    key_prefix: Optional[Union[str, List[str]]] = None,
    deps: Optional[Sequence[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    kind: Optional[Union[str, AssetKind]] = None,
    kinds: Optional[List[str]] = None,
    group_name: Optional[str] = None,
    code_version: Optional[str] = None,
    owners: Optional[List[str]] = None,
    tags: Optional[Dict[str, str]] = None,
    is_required: bool = True,
    compute_kind: Optional[str] = None,
    **kwargs,
) -> Callable:
    """
    Decorator to define a single asset.

    This decorator marks a function as producing a single asset and automatically
    registers its metadata, dependencies, and configuration.

    Args:
        func: The function to decorate (provided automatically)
        name: Override the asset name (default: function name)
        key: Full asset key (e.g., "ocr/text"). Overrides key_prefix.
        key_prefix: Prefix for asset key (e.g., "ocr" -> "ocr/function_name")
        deps: List of upstream asset keys this asset depends on
        metadata: Additional metadata dictionary
        description: Human-readable description of the asset
        kind: Primary asset kind (TEXT, JSON, BBOX, etc.)
        kinds: Additional technology/tool kinds (python, llm, torch, etc.)
        group_name: Logical grouping for UI/organization
        code_version: Version string for tracking code changes
        owners: List of owner identifiers (emails, teams, etc.)
        tags: Key-value tags for filtering/organization
        is_required: Whether this asset is critical for data quality
        compute_kind: Override compute kind (e.g., "python", "sql")
        **kwargs: Additional decorator arguments

    Returns:
        Decorated function with asset metadata attached

    Example:
        ```python
        @asset(
            key="ocr/text",
            kind=AssetKind.TEXT,
            kinds=["craft_ocr", "torch"],
            metadata={"model": "craft-v2"},
            owners=["ocr-team@company.com"],
        )
        def extract_text(context, docs):
            # Extract text from documents
            return extracted_text
        ```
    """

    def decorator(fn: Callable) -> Callable:
        # Determine asset key
        asset_name = name or fn.__name__
        if key:
            asset_key = key
        elif key_prefix:
            prefix = (
                "/".join(key_prefix) if isinstance(key_prefix, list) else key_prefix
            )
            asset_key = f"{prefix}/{asset_name}"
        else:
            asset_key = asset_name

        # Infer kind if not specified
        inferred_kind = kind or AssetKind.JSON

        # Build metadata
        asset_metadata = metadata or {}
        if group_name:
            asset_metadata["group_name"] = group_name
        if code_version:
            asset_metadata["code_version"] = code_version
        if owners:
            asset_metadata["owners"] = owners
        if tags:
            asset_metadata["tags"] = tags
        if compute_kind:
            asset_metadata["compute_kind"] = compute_kind

        # Create AssetSpec
        spec = AssetSpec(
            key=asset_key,
            kind=inferred_kind,
            is_primary=True,
            is_required=is_required,
            description=description or fn.__doc__,
            kinds=kinds or [],
            metadata=asset_metadata,
        )

        # Attach metadata to function
        fn.__asset_spec__ = spec
        fn.__asset_deps__ = deps or []
        fn.__is_marie_asset__ = True

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        # Copy asset metadata to wrapper
        wrapper.__asset_spec__ = spec
        wrapper.__asset_deps__ = deps or []
        wrapper.__is_marie_asset__ = True

        return wrapper

    # Support both @asset and @asset(...)
    if func is not None:
        return decorator(func)
    return decorator


def multi_asset(
    func: Optional[Callable] = None,
    *,
    specs: Optional[List[Dict[str, Any]]] = None,
    assets: Optional[List[AssetSpec]] = None,
    name: Optional[str] = None,
    key_prefix: Optional[Union[str, List[str]]] = None,
    deps: Optional[Sequence[str]] = None,
    group_name: Optional[str] = None,
    code_version: Optional[str] = None,
    owners: Optional[List[str]] = None,
    tags: Optional[Dict[str, str]] = None,
    compute_kind: Optional[str] = None,
    **kwargs,
) -> Callable:
    """
    Decorator to define multiple assets from a single operation.

    Use this when a single computation produces multiple related outputs.

    Args:
        func: The function to decorate (provided automatically)
        specs: List of asset specification dicts (simplified format)
        assets: List of AssetSpec objects (full control)
        name: Override the operation name (default: function name)
        key_prefix: Prefix for all asset keys
        deps: Shared upstream dependencies for all assets
        group_name: Logical grouping for all assets
        code_version: Version string for tracking code changes
        owners: List of owner identifiers
        tags: Key-value tags for all assets
        compute_kind: Override compute kind
        **kwargs: Additional decorator arguments

    Returns:
        Decorated function with multi-asset metadata attached

    Example:
        ```python
        @multi_asset(
            specs=[
                {"key": "ocr/text", "kind": "text", "is_primary": True},
                {"key": "ocr/bboxes", "kind": "bbox"},
                {"key": "ocr/confidence", "kind": "metadata", "is_required": False},
            ],
            group_name="ocr",
            owners=["ocr-team@company.com"],
        )
        def extract_ocr_data(context, docs):
            # Extract multiple outputs
            return {"ocr/text": text, "ocr/bboxes": bboxes, "ocr/confidence": confidence}
        ```
    """

    def decorator(fn: Callable) -> Callable:
        # Build asset specs
        asset_specs = []

        if assets:
            # Use provided AssetSpec objects
            asset_specs = assets
        elif specs:
            # Build from simplified spec dicts
            for spec_dict in specs:
                spec = _spec_dict_to_asset_spec(
                    spec_dict,
                    key_prefix,
                    group_name,
                    code_version,
                    owners,
                    tags,
                    compute_kind,
                )
                asset_specs.append(spec)
        else:
            raise ValueError(
                "Either 'specs' or 'assets' must be provided to @multi_asset"
            )

        # Attach metadata to function
        fn.__asset_specs__ = asset_specs
        fn.__asset_deps__ = deps or []
        fn.__is_marie_multi_asset__ = True
        fn.__operation_name__ = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        # Copy asset metadata to wrapper
        wrapper.__asset_specs__ = asset_specs
        wrapper.__asset_deps__ = deps or []
        wrapper.__is_marie_multi_asset__ = True
        wrapper.__operation_name__ = name or fn.__name__

        return wrapper

    # Support both @multi_asset and @multi_asset(...)
    if func is not None:
        return decorator(func)
    return decorator


def graph_asset(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    key: Optional[str] = None,
    key_prefix: Optional[Union[str, List[str]]] = None,
    description: Optional[str] = None,
    group_name: Optional[str] = None,
    **kwargs,
) -> Callable:
    """
    Decorator to define an asset from a graph of operations.

    Use this for complex assets built from multiple internal steps without exposing
    each step as a separate asset.

    Args:
        func: The function defining the graph
        name: Override the asset name
        key: Full asset key
        key_prefix: Prefix for asset key
        description: Human-readable description
        group_name: Logical grouping
        **kwargs: Additional decorator arguments

    Returns:
        Decorated function with graph asset metadata

    Example:
        ```python
        @graph_asset(key="reports/weekly_summary", group_name="reporting")
        def weekly_summary_asset():
            # Compose multiple ops into a single asset
            data = fetch_data()
            processed = process_data(data)
            return generate_report(processed)
        ```
    """

    def decorator(fn: Callable) -> Callable:
        asset_name = name or fn.__name__
        if key:
            asset_key = key
        elif key_prefix:
            prefix = (
                "/".join(key_prefix) if isinstance(key_prefix, list) else key_prefix
            )
            asset_key = f"{prefix}/{asset_name}"
        else:
            asset_key = asset_name

        fn.__is_marie_graph_asset__ = True
        fn.__asset_key__ = asset_key
        fn.__asset_description__ = description or fn.__doc__
        fn.__group_name__ = group_name

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        wrapper.__is_marie_graph_asset__ = True
        wrapper.__asset_key__ = asset_key
        wrapper.__asset_description__ = description or fn.__doc__
        wrapper.__group_name__ = group_name

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def observable_source_asset(
    func: Optional[Callable] = None,
    *,
    key: str,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Callable:
    """
    Decorator to define an external/observable asset.

    Use this for assets that exist outside Marie-AI's control but can be observed.

    Args:
        func: The observation function
        key: Asset key to observe
        description: Description of the external asset
        metadata: Metadata about the external asset
        **kwargs: Additional decorator arguments

    Returns:
        Decorated observation function

    Example:
        ```python
        @observable_source_asset(
            key="s3/raw_documents", description="Raw documents from S3 bucket"
        )
        def observe_s3_bucket(context):
            # Check S3 for new documents
            return observation_metadata
        ```
    """

    def decorator(fn: Callable) -> Callable:
        fn.__is_marie_observable_asset__ = True
        fn.__asset_key__ = key
        fn.__asset_description__ = description or fn.__doc__
        fn.__asset_metadata__ = metadata or {}

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        wrapper.__is_marie_observable_asset__ = True
        wrapper.__asset_key__ = key
        wrapper.__asset_description__ = description or fn.__doc__
        wrapper.__asset_metadata__ = metadata or {}

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


# Helper functions


def _spec_dict_to_asset_spec(
    spec_dict: Dict[str, Any],
    key_prefix: Optional[Union[str, List[str]]],
    group_name: Optional[str],
    code_version: Optional[str],
    owners: Optional[List[str]],
    tags: Optional[Dict[str, str]],
    compute_kind: Optional[str],
) -> AssetSpec:
    """Convert a simplified spec dict to AssetSpec."""
    # Apply key prefix if specified
    asset_key = spec_dict["key"]
    if key_prefix and not asset_key.startswith("/"):
        prefix = "/".join(key_prefix) if isinstance(key_prefix, list) else key_prefix
        asset_key = f"{prefix}/{asset_key}"

    # Get or infer kind
    kind = spec_dict.get("kind", "json")
    if isinstance(kind, str):
        kind = AssetKind(kind)

    # Build metadata
    metadata = spec_dict.get("metadata", {})
    if group_name:
        metadata["group_name"] = group_name
    if code_version:
        metadata["code_version"] = code_version
    if owners:
        metadata["owners"] = owners
    if tags:
        metadata["tags"] = tags
    if compute_kind:
        metadata["compute_kind"] = compute_kind

    return AssetSpec(
        key=asset_key,
        kind=kind,
        is_primary=spec_dict.get("is_primary", False),
        is_required=spec_dict.get("is_required", True),
        description=spec_dict.get("description"),
        kinds=spec_dict.get("kinds", []),
        metadata=metadata,
    )


def get_asset_spec(func: Callable) -> Optional[AssetSpec]:
    """Extract AssetSpec from a decorated function."""
    return getattr(func, "__asset_spec__", None)


def get_asset_specs(func: Callable) -> Optional[List[AssetSpec]]:
    """Extract list of AssetSpecs from a multi_asset decorated function."""
    return getattr(func, "__asset_specs__", None)


def is_asset(func: Callable) -> bool:
    """Check if a function is decorated with @asset."""
    return getattr(func, "__is_marie_asset__", False)


def is_multi_asset(func: Callable) -> bool:
    """Check if a function is decorated with @multi_asset."""
    return getattr(func, "__is_marie_multi_asset__", False)


def is_graph_asset(func: Callable) -> bool:
    """Check if a function is decorated with @graph_asset."""
    return getattr(func, "__is_marie_graph_asset__", False)


def is_observable_asset(func: Callable) -> bool:
    """Check if a function is decorated with @observable_source_asset."""
    return getattr(func, "__is_marie_observable_asset__", False)
