"""Asset scaffolding for generating asset tracking boilerplate."""

import os
from typing import Dict

from marie.logging_core.logger import MarieLogger

logger = MarieLogger("scaffold").logger

# Template mappings
TEMPLATES = {
    "single": {
        "asset_key": "{executor_name}/output",
        "description": "Single asset executor",
        "assets": [
            {"key": "{executor_name}/output", "kind": "json", "is_primary": True}
        ],
    },
    "multi-asset": {
        "asset_key": "{executor_name}/primary",
        "description": "Multi-asset executor",
        "assets": [
            {
                "key": "{executor_name}/primary",
                "kind": "json",
                "is_primary": True,
                "is_required": True,
            },
            {
                "key": "{executor_name}/secondary",
                "kind": "json",
                "is_primary": False,
                "is_required": True,
            },
            {
                "key": "{executor_name}/metadata",
                "kind": "metadata",
                "is_primary": False,
                "is_required": False,
            },
        ],
    },
    "ocr": {
        "asset_key": "ocr/text",
        "description": "OCR executor with text, bboxes, and confidence",
        "assets": [
            {
                "key": "ocr/text",
                "kind": "text",
                "is_primary": True,
                "is_required": True,
            },
            {
                "key": "ocr/bboxes",
                "kind": "bbox",
                "is_primary": False,
                "is_required": True,
            },
            {
                "key": "ocr/confidence",
                "kind": "metadata",
                "is_primary": False,
                "is_required": False,
            },
        ],
    },
    "classification": {
        "asset_key": "classify/document_type",
        "description": "Classification executor",
        "assets": [
            {
                "key": "classify/document_type",
                "kind": "classification",
                "is_primary": True,
                "is_required": True,
            },
            {
                "key": "classify/confidence_scores",
                "kind": "metadata",
                "is_primary": False,
                "is_required": False,
            },
        ],
    },
    "extraction": {
        "asset_key": "extract/claims",
        "description": "Extraction executor",
        "assets": [
            {
                "key": "extract/claims",
                "kind": "json",
                "is_primary": True,
                "is_required": True,
            },
            {
                "key": "extract/headers",
                "kind": "json",
                "is_primary": False,
                "is_required": True,
            },
            {
                "key": "extract/service_lines",
                "kind": "table",
                "is_primary": False,
                "is_required": True,
            },
            {
                "key": "extract/metadata",
                "kind": "metadata",
                "is_primary": False,
                "is_required": False,
            },
        ],
    },
}


def _generate_yaml_config(executor_name: str, template: str) -> str:
    """Generate YAML configuration for asset tracking."""
    template_data = TEMPLATES[template]
    assets = template_data["assets"]

    # Format asset keys
    for asset in assets:
        asset["key"] = asset["key"].format(executor_name=executor_name)

    # Generate simple format if single asset
    if len(assets) == 1:
        return f"""# Auto-generated asset configuration
asset_config:
  {executor_name}: "{assets[0]['key']}"
"""

    # Generate list format if multiple assets with default properties
    all_defaults = all(
        a.get("is_primary") == (i == 0)
        and a.get("is_required", True)
        and a.get("description") is None
        for i, a in enumerate(assets)
    )

    if all_defaults:
        keys = "\n    - ".join(f'"{a["key"]}"' for a in assets)
        return f"""# Auto-generated asset configuration
asset_config:
  {executor_name}:
    - {keys}
"""

    # Generate full format
    yaml = f"""# Auto-generated asset configuration
asset_config:
  {executor_name}:
    assets:
"""
    for asset in assets:
        yaml += f"""      - key: "{asset['key']}"
        kind: "{asset['kind']}"
        is_primary: {str(asset.get('is_primary', False)).lower()}
        is_required: {str(asset.get('is_required', True)).lower()}
"""
        if asset.get("description"):
            yaml += f'        description: "{asset["description"]}"\n'

    return yaml


def _generate_python_code(executor_name: str, class_name: str, template: str) -> str:
    """Generate Python code for asset tracking."""
    template_data = TEMPLATES[template]
    assets = template_data["assets"]

    # Format asset keys
    for asset in assets:
        asset["key"] = asset["key"].format(executor_name=executor_name)

    # Build asset list code
    asset_code_blocks = []

    for i, asset in enumerate(assets):
        block = f'''
            # {"Primary" if asset.get("is_primary") else "Secondary"} asset: {asset["key"]}
            {f"asset_{i}_data" if len(assets) > 1 else "result_data"} = doc.tags.get('{asset["key"].split("/")[-1]}')
            {f"asset_{i}_bytes" if len(assets) > 1 else "result_bytes"} = json.dumps({f"asset_{i}_data" if len(assets) > 1 else "result_data"}).encode('utf-8')
            {f"asset_{i}_version" if len(assets) > 1 else "version"} = AssetTracker.compute_asset_version(
                payload_bytes={f"asset_{i}_bytes" if len(assets) > 1 else "result_bytes"},
                code_fingerprint=self.code_version,
                prompt_fingerprint=getattr(self, 'model_version', None),
                upstream_versions=self._get_upstream_versions(dag_id, node_task_id)
            )

            assets.append({{
                "asset_key": "{asset["key"]}",
                "version": {f"asset_{i}_version" if len(assets) > 1 else "version"},
                "kind": "{asset["kind"]}",
                "size_bytes": len({f"asset_{i}_bytes" if len(assets) > 1 else "result_bytes"}),
                "checksum": hashlib.sha256({f"asset_{i}_bytes" if len(assets) > 1 else "result_bytes"}).hexdigest(),
                "uri": f"s3://bucket/{{job_id}}/{asset["key"]}",
                "metadata": {{}}  # Add useful metadata here
            }})'''
        asset_code_blocks.append(block)

    assets_code = "\n".join(asset_code_blocks)

    code = f'''"""
Auto-generated asset tracking for {class_name}
Generated by: marie scaffold asset
"""

import hashlib
import json
from marie import Executor, requests, DocumentArray
from marie.executor.mixin import StorageMixin
from marie.assets import AssetTracker, DAGAssetMapper


class {class_name}(StorageMixin, Executor):
    """
    {template_data["description"]}

    Produces assets:
{chr(10).join(f"    - {a['key']} ({a['kind']})" for a in assets)}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set code/model versions for reproducibility
        self.code_version = "git:unknown"  # TODO: Set from git commit
        self.model_version = None  # TODO: Set if using ML models

        # Initialize asset tracker
        if self.asset_tracking_enabled:
            self.asset_tracker = AssetTracker(
                storage_handler=self.storage_handler,
                storage_conf=self.storage_conf
            )

    @requests(on='/process')
    async def process(self, docs: DocumentArray, **kwargs):
        """
        Process documents and track assets.

        Args:
            docs: Input documents
            **kwargs: Additional arguments (job_id, dag_id, node_task_id, etc.)

        Returns:
            Processed documents
        """
        # TODO: Implement your processing logic here
        for doc in docs:
            result = self._do_processing(doc)
            doc.tags['result'] = result

        # Persist with asset tracking
        if self.asset_tracking_enabled:
            await self.persist(
                docs=docs,
                job_id=kwargs.get('job_id'),
                dag_id=kwargs.get('dag_id'),
                node_task_id=kwargs.get('node_task_id'),
                partition_key=kwargs.get('partition_key'),
            )

        return docs

    def _do_processing(self, doc):
        """
        TODO: Implement your actual processing logic here.

        Args:
            doc: Input document

        Returns:
            Processing result
        """
        raise NotImplementedError("Implement your processing logic")

    async def persist(self, docs, job_id, dag_id=None, node_task_id=None, partition_key=None):
        """
        Persist results and record asset materializations.

        Args:
            docs: Documents to persist
            job_id: Job ID
            dag_id: DAG ID (optional)
            node_task_id: Node task ID (optional)
            partition_key: Partition key (optional)
        """
        # Prepare asset list
        assets = []

        for doc in docs:
{assets_code}

        # Get upstream assets for lineage
        upstream = self._get_upstream_asset_tuples(dag_id, node_task_id)

        # Record materializations
        try:
            await self.asset_tracker.record_materializations(
                storage_event_id=None,
                assets=assets,
                job_id=job_id,
                dag_id=dag_id,
                node_task_id=node_task_id,
                partition_key=partition_key,
                upstream_assets=upstream,
            )
        except Exception as e:
            self.logger.error(f"Failed to record assets: {{e}}", exc_info=True)
            # Continue execution - asset tracking should not block

    def _get_upstream_versions(self, dag_id, node_task_id):
        """Get versions of upstream assets for version computation."""
        if not dag_id or not node_task_id:
            return []

        upstream = DAGAssetMapper.get_upstream_assets_for_node(
            dag_id=dag_id,
            node_task_id=node_task_id,
            get_connection_fn=self.storage_handler._get_connection,
            close_connection_fn=self.storage_handler._close_connection
        )

        return [u['latest_version'] for u in upstream if u['latest_version']]

    def _get_upstream_asset_tuples(self, dag_id, node_task_id):
        """Get upstream asset tuples for lineage recording."""
        if not dag_id or not node_task_id:
            return []

        upstream = DAGAssetMapper.get_upstream_assets_for_node(
            dag_id=dag_id,
            node_task_id=node_task_id,
            get_connection_fn=self.storage_handler._get_connection,
            close_connection_fn=self.storage_handler._close_connection
        )

        return [
            (u['asset_key'], u['latest_version'], u['partition_key'])
            for u in upstream
        ]
'''

    return code


def scaffold_asset(args):
    """
    Scaffold asset tracking for an executor.

    Args:
        args: Command-line arguments with:
            - path: Path to executor file
            - template: Template to use
            - output: Output file path
            - config: Generate YAML config instead
            - dry_run: Preview without writing
            - force: Overwrite existing file
    """
    from marie.helper import get_rich_console

    console = get_rich_console()

    # Extract executor name from path
    file_path = args.path
    base_name = os.path.basename(file_path)
    executor_name = base_name.replace("_executor.py", "").replace(".py", "")

    # Generate class name
    class_name = "".join(word.capitalize() for word in executor_name.split("_"))
    if not class_name.endswith("Executor"):
        class_name += "Executor"

    # Determine output path
    output_path = args.output if args.output else file_path

    # Check if output exists and not force
    if os.path.exists(output_path) and not args.force and not args.dry_run:
        console.print(f"[red]Error:[/red] Output file already exists: {output_path}")
        console.print("[yellow]Use --force to overwrite[/yellow]")
        return

    # Generate content
    if args.config:
        content = _generate_yaml_config(executor_name, args.template)
        content_type = "YAML"
    else:
        content = _generate_python_code(executor_name, class_name, args.template)
        content_type = "Python"

    # Preview or write
    if args.dry_run:
        console.print(f"\n[bold]Generated {content_type} Code:[/bold]\n")
        console.print(content)
        console.print(f"\n[yellow]Dry run - nothing written[/yellow]")
    else:
        with open(output_path, "w") as f:
            f.write(content)

        console.print(f"[green]✓[/green] Scaffolded asset tracking to: {output_path}")
        console.print(f"[blue]Template:[/blue] {args.template}")
        console.print(f"[blue]Executor:[/blue] {class_name}")

        if args.config:
            console.print(
                "\n[yellow]Next steps:[/yellow] Add this configuration to your service YAML"
            )
        else:
            console.print(
                "\n[yellow]Next steps:[/yellow] Review and customize the generated code:"
            )
            console.print("  1. Implement _do_processing() method")
            console.print("  2. Customize asset keys and metadata")
            console.print("  3. Set code_version and model_version")
            console.print("  4. Add your business logic")
