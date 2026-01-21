"""Document Processing Agent Example.

This example demonstrates how to create an agent that integrates with
Marie's document processing API for comprehensive document understanding.

This is the core use case for Marie-AI: Visual Document Understanding.

Shows:
- OCR via Marie's /api/document/extract endpoint
- Document classification via /api/document/classify endpoint
- NER extraction via /api/ner/extract endpoint
- Table extraction using Marie's pipeline features
- Both direct API calls and S3-based processing

Usage:
    python document_agent.py --document path/to/document.pdf --task "Extract all text"
    python document_agent.py --tui
    python document_agent.py --demo
"""

import argparse
import base64
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from marie.agent import (
    AgentTool,
    AssistantAgent,
    MarieEngineLLMWrapper,
    PlanningAgent,
    ToolMetadata,
    ToolOutput,
    register_tool,
)

# =============================================================================
# Configuration
# =============================================================================


class MarieAPIConfig:
    """Configuration for Marie API connection."""

    def __init__(
        self,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        queue_id: Optional[str] = None,
    ):
        self.api_base_url = api_base_url or os.getenv(
            "MARIE_API_URL", "http://127.0.0.1:51000/api"
        )
        self.api_key = api_key or os.getenv("MARIE_API_KEY", "")
        self.queue_id = queue_id or os.getenv("MARIE_QUEUE_ID", "0000-0000-0000-0000")

    @property
    def headers(self) -> Dict[str, str]:
        """Get authorization headers for API requests."""
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


# Global config instance - can be overridden
_config = MarieAPIConfig()


def set_api_config(
    api_base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    queue_id: Optional[str] = None,
):
    """Set the global Marie API configuration."""
    global _config
    _config = MarieAPIConfig(api_base_url, api_key, queue_id)


def get_api_config() -> MarieAPIConfig:
    """Get the current Marie API configuration."""
    return _config


# =============================================================================
# OCR Tool - Uses Marie's /api/document/extract endpoint
# =============================================================================


class OCRTool(AgentTool):
    """Tool for extracting text from documents using Marie's OCR API.

    Uses Marie's document extraction endpoint which provides:
    - High-quality OCR with multiple model options
    - Page classification and splitting
    - Bounding box detection with confidence scores
    - Support for multiple document formats
    """

    def __init__(self, config: Optional[MarieAPIConfig] = None):
        """Initialize OCR tool with optional config override."""
        self._config = config

    @property
    def config(self) -> MarieAPIConfig:
        return self._config or get_api_config()

    @property
    def name(self) -> str:
        return "document_ocr"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description="Extract text from a document image using Marie's OCR API. "
            "Returns structured text with bounding boxes and confidence scores.",
            parameters={
                "type": "object",
                "properties": {
                    "document_path": {
                        "type": "string",
                        "description": "Path to document image or PDF",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["word", "sparse", "line", "raw_line", "multiline"],
                        "description": "OCR mode: word (single word), sparse (find all text), "
                        "line (single line), raw_line (no bbox detection), "
                        "multiline (multiple lines, default)",
                    },
                    "pipeline": {
                        "type": "string",
                        "description": "Processing pipeline name (default: 'default')",
                    },
                },
                "required": ["document_path"],
            },
        )

    def _encode_file(self, file_path: str) -> str:
        """Encode file to base64."""
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def call(self, **kwargs) -> ToolOutput:
        """Execute OCR on a document via Marie API."""
        document_path = kwargs.get("document_path", kwargs.get("path", ""))
        mode = kwargs.get("mode", "multiline")
        pipeline = kwargs.get("pipeline", "default")

        if not document_path:
            return ToolOutput(
                content=json.dumps({"error": "document_path is required"}),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"error": "document_path is required"},
                is_error=True,
            )

        path = Path(document_path)
        if not path.exists():
            return ToolOutput(
                content=json.dumps({"error": f"File not found: {document_path}"}),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"error": "File not found"},
                is_error=True,
            )

        try:
            # Encode file as base64
            base64_data = self._encode_file(document_path)
            uid = str(uuid.uuid4())

            # Build request payload
            endpoint_url = f"{self.config.api_base_url}/document/extract"
            payload = {
                "queue_id": self.config.queue_id,
                "data": base64_data,
                "doc_id": f"ocr-{uid}",
                "doc_type": "agent_ocr",
                "mode": mode,
                "output": "json",
                "features": [
                    {
                        "type": "pipeline",
                        "name": pipeline,
                        "page_classifier": {"enabled": True},
                        "page_splitter": {"enabled": True},
                        "ocr": {
                            "document": {"model": "default"},
                            "region": {"model": "best"},
                        },
                    }
                ],
            }

            # Make API request
            response = requests.post(
                endpoint_url,
                headers=self.config.headers,
                json=payload,
                timeout=120,
            )

            if response.status_code != 200:
                return ToolOutput(
                    content=json.dumps(
                        {
                            "error": f"API request failed with status {response.status_code}",
                            "details": response.text[:500],
                        }
                    ),
                    tool_name=self.name,
                    raw_input=kwargs,
                    raw_output={"error": "API request failed"},
                    is_error=True,
                )

            result = response.json()
            result["document_path"] = document_path
            result["mode"] = mode
            result["method"] = "marie_api"

            return ToolOutput(
                content=json.dumps(result, indent=2),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output=result,
                is_error=False,
            )

        except requests.exceptions.ConnectionError:
            return ToolOutput(
                content=json.dumps(
                    {
                        "error": "Cannot connect to Marie API",
                        "api_url": self.config.api_base_url,
                        "hint": "Ensure Marie server is running: marie server start",
                    }
                ),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"error": "Connection failed"},
                is_error=True,
            )
        except Exception as e:
            return ToolOutput(
                content=json.dumps({"error": str(e), "document_path": document_path}),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"error": str(e)},
                is_error=True,
            )


# =============================================================================
# Document Classifier Tool - Uses Marie's /api/document/classify endpoint
# =============================================================================


class DocumentClassifierTool(AgentTool):
    """Tool for classifying document types using Marie's classification API.

    Uses Marie's document classification endpoint which provides:
    - ML-based document type classification
    - Confidence scores for each category
    - Support for custom classification pipelines
    """

    def __init__(self, config: Optional[MarieAPIConfig] = None):
        self._config = config

    @property
    def config(self) -> MarieAPIConfig:
        return self._config or get_api_config()

    @property
    def name(self) -> str:
        return "document_classifier"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description="Classify a document into categories using Marie's ML classification API. "
            "Returns document type with confidence scores.",
            parameters={
                "type": "object",
                "properties": {
                    "document_path": {
                        "type": "string",
                        "description": "Path to document image",
                    },
                    "pipeline": {
                        "type": "string",
                        "description": "Classification pipeline name (default: 'default')",
                    },
                },
                "required": ["document_path"],
            },
        )

    def _encode_file(self, file_path: str) -> str:
        """Encode file to base64."""
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def call(self, **kwargs) -> ToolOutput:
        """Classify a document via Marie API."""
        document_path = kwargs.get("document_path", kwargs.get("path", ""))
        pipeline = kwargs.get("pipeline", "default")

        if not document_path:
            return ToolOutput(
                content=json.dumps({"error": "document_path is required"}),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"error": "document_path is required"},
                is_error=True,
            )

        if not Path(document_path).exists():
            return ToolOutput(
                content=json.dumps({"error": f"File not found: {document_path}"}),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"error": "File not found"},
                is_error=True,
            )

        try:
            base64_data = self._encode_file(document_path)
            uid = str(uuid.uuid4())

            endpoint_url = f"{self.config.api_base_url}/document/classify"
            payload = {
                "queue_id": self.config.queue_id,
                "data": base64_data,
                "doc_id": f"classify-{uid}",
                "doc_type": "agent_classify",
                "mode": "multiline",
                "output": "json",
                "pipeline": pipeline,
                "features": [],
            }

            response = requests.post(
                endpoint_url,
                headers=self.config.headers,
                json=payload,
                timeout=60,
            )

            if response.status_code != 200:
                return ToolOutput(
                    content=json.dumps(
                        {
                            "error": f"API request failed with status {response.status_code}",
                            "details": response.text[:500],
                        }
                    ),
                    tool_name=self.name,
                    raw_input=kwargs,
                    raw_output={"error": "API request failed"},
                    is_error=True,
                )

            result = response.json()
            result["document_path"] = document_path
            result["method"] = "marie_api"

            return ToolOutput(
                content=json.dumps(result, indent=2),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output=result,
                is_error=False,
            )

        except requests.exceptions.ConnectionError:
            return ToolOutput(
                content=json.dumps(
                    {
                        "error": "Cannot connect to Marie API",
                        "api_url": self.config.api_base_url,
                        "hint": "Ensure Marie server is running",
                    }
                ),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"error": "Connection failed"},
                is_error=True,
            )
        except Exception as e:
            return ToolOutput(
                content=json.dumps({"error": str(e), "document_path": document_path}),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"error": str(e)},
                is_error=True,
            )


# =============================================================================
# NER Extractor Tool - Uses Marie's /api/ner/extract endpoint
# =============================================================================


class NERExtractorTool(AgentTool):
    """Tool for Named Entity Recognition using Marie's NER API.

    Uses Marie's NER extraction endpoint which provides:
    - ML-based entity extraction
    - Support for dates, amounts, names, organizations, etc.
    - High accuracy on document images
    """

    def __init__(self, config: Optional[MarieAPIConfig] = None):
        self._config = config

    @property
    def config(self) -> MarieAPIConfig:
        return self._config or get_api_config()

    @property
    def name(self) -> str:
        return "ner_extractor"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description="Extract named entities from a document using Marie's NER API. "
            "Returns entities like dates, amounts, names, organizations, etc.",
            parameters={
                "type": "object",
                "properties": {
                    "document_path": {
                        "type": "string",
                        "description": "Path to document image",
                    },
                    "mode": {
                        "type": "string",
                        "description": "Extraction mode (default: 'multiline')",
                    },
                },
                "required": ["document_path"],
            },
        )

    def _encode_file(self, file_path: str) -> str:
        """Encode file to base64."""
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def call(self, **kwargs) -> ToolOutput:
        """Extract named entities via Marie API."""
        document_path = kwargs.get("document_path", kwargs.get("path", ""))
        mode = kwargs.get("mode", "multiline")

        if not document_path:
            return ToolOutput(
                content=json.dumps({"error": "document_path is required"}),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"error": "document_path is required"},
                is_error=True,
            )

        if not Path(document_path).exists():
            return ToolOutput(
                content=json.dumps({"error": f"File not found: {document_path}"}),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"error": "File not found"},
                is_error=True,
            )

        try:
            base64_data = self._encode_file(document_path)
            uid = str(uuid.uuid4())

            endpoint_url = f"{self.config.api_base_url}/ner/extract"
            payload = {
                "queue_id": self.config.queue_id,
                "data": base64_data,
                "doc_id": f"ner-{uid}",
                "doc_type": "agent_ner",
                "mode": mode,
                "output": "json",
            }

            response = requests.post(
                endpoint_url,
                headers=self.config.headers,
                json=payload,
                timeout=60,
            )

            if response.status_code != 200:
                return ToolOutput(
                    content=json.dumps(
                        {
                            "error": f"API request failed with status {response.status_code}",
                            "details": response.text[:500],
                        }
                    ),
                    tool_name=self.name,
                    raw_input=kwargs,
                    raw_output={"error": "API request failed"},
                    is_error=True,
                )

            result = response.json()
            result["document_path"] = document_path
            result["method"] = "marie_api"

            return ToolOutput(
                content=json.dumps(result, indent=2),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output=result,
                is_error=False,
            )

        except requests.exceptions.ConnectionError:
            return ToolOutput(
                content=json.dumps(
                    {
                        "error": "Cannot connect to Marie API",
                        "api_url": self.config.api_base_url,
                        "hint": "Ensure Marie server is running",
                    }
                ),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"error": "Connection failed"},
                is_error=True,
            )
        except Exception as e:
            return ToolOutput(
                content=json.dumps({"error": str(e), "document_path": document_path}),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"error": str(e)},
                is_error=True,
            )


# =============================================================================
# Table Extractor Tool - Uses Marie's extract API with table features
# =============================================================================


class TableExtractorTool(AgentTool):
    """Tool for extracting tables from documents using Marie's API.

    Uses Marie's document extraction endpoint with table detection features
    to identify and extract structured table data.
    """

    def __init__(self, config: Optional[MarieAPIConfig] = None):
        self._config = config

    @property
    def config(self) -> MarieAPIConfig:
        return self._config or get_api_config()

    @property
    def name(self) -> str:
        return "table_extractor"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description="Extract tables from a document using Marie's API. "
            "Returns structured table data with rows and columns.",
            parameters={
                "type": "object",
                "properties": {
                    "document_path": {
                        "type": "string",
                        "description": "Path to document image",
                    },
                },
                "required": ["document_path"],
            },
        )

    def _encode_file(self, file_path: str) -> str:
        """Encode file to base64."""
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def call(self, **kwargs) -> ToolOutput:
        """Extract tables from a document via Marie API."""
        document_path = kwargs.get("document_path", kwargs.get("path", ""))

        if not document_path:
            return ToolOutput(
                content=json.dumps({"error": "document_path is required"}),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"error": "document_path is required"},
                is_error=True,
            )

        if not Path(document_path).exists():
            return ToolOutput(
                content=json.dumps({"error": f"File not found: {document_path}"}),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"error": "File not found"},
                is_error=True,
            )

        try:
            base64_data = self._encode_file(document_path)
            uid = str(uuid.uuid4())

            # Use extract endpoint with table-focused pipeline
            endpoint_url = f"{self.config.api_base_url}/document/extract"
            payload = {
                "queue_id": self.config.queue_id,
                "data": base64_data,
                "doc_id": f"table-{uid}",
                "doc_type": "agent_table",
                "mode": "multiline",
                "output": "json",
                "features": [
                    {
                        "type": "pipeline",
                        "name": "default",
                        "page_classifier": {"enabled": True},
                        "table_detection": {"enabled": True},
                        "ocr": {
                            "document": {"model": "default"},
                            "region": {"model": "best"},
                        },
                    }
                ],
            }

            response = requests.post(
                endpoint_url,
                headers=self.config.headers,
                json=payload,
                timeout=120,
            )

            if response.status_code != 200:
                return ToolOutput(
                    content=json.dumps(
                        {
                            "error": f"API request failed with status {response.status_code}",
                            "details": response.text[:500],
                        }
                    ),
                    tool_name=self.name,
                    raw_input=kwargs,
                    raw_output={"error": "API request failed"},
                    is_error=True,
                )

            result = response.json()
            result["document_path"] = document_path
            result["method"] = "marie_api"

            return ToolOutput(
                content=json.dumps(result, indent=2),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output=result,
                is_error=False,
            )

        except requests.exceptions.ConnectionError:
            return ToolOutput(
                content=json.dumps(
                    {
                        "error": "Cannot connect to Marie API",
                        "api_url": self.config.api_base_url,
                        "hint": "Ensure Marie server is running",
                    }
                ),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"error": "Connection failed"},
                is_error=True,
            )
        except Exception as e:
            return ToolOutput(
                content=json.dumps({"error": str(e), "document_path": document_path}),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"error": str(e)},
                is_error=True,
            )


# =============================================================================
# Key-Value Extractor Tool - Uses Marie's extract API for form fields
# =============================================================================


class KeyValueExtractorTool(AgentTool):
    """Tool for extracting key-value pairs from form-like documents.

    Uses Marie's document extraction API to identify and extract
    key-value pairs from forms, invoices, and structured documents.
    """

    def __init__(self, config: Optional[MarieAPIConfig] = None):
        self._config = config

    @property
    def config(self) -> MarieAPIConfig:
        return self._config or get_api_config()

    @property
    def name(self) -> str:
        return "kv_extractor"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description="Extract key-value pairs from form-like documents using Marie's API. "
            "Works with invoices, forms, and structured documents.",
            parameters={
                "type": "object",
                "properties": {
                    "document_path": {
                        "type": "string",
                        "description": "Path to document image",
                    },
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific keys to look for (optional)",
                    },
                },
                "required": ["document_path"],
            },
        )

    def _encode_file(self, file_path: str) -> str:
        """Encode file to base64."""
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def call(self, **kwargs) -> ToolOutput:
        """Extract key-value pairs from a document via Marie API."""
        document_path = kwargs.get("document_path", kwargs.get("path", ""))
        keys = kwargs.get("keys", [])

        if not document_path:
            return ToolOutput(
                content=json.dumps({"error": "document_path is required"}),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"error": "document_path is required"},
                is_error=True,
            )

        if not Path(document_path).exists():
            return ToolOutput(
                content=json.dumps({"error": f"File not found: {document_path}"}),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"error": "File not found"},
                is_error=True,
            )

        try:
            base64_data = self._encode_file(document_path)
            uid = str(uuid.uuid4())

            # Use extract endpoint - the OCR results can be post-processed for KV pairs
            endpoint_url = f"{self.config.api_base_url}/document/extract"
            payload = {
                "queue_id": self.config.queue_id,
                "data": base64_data,
                "doc_id": f"kv-{uid}",
                "doc_type": "agent_kv",
                "mode": "multiline",
                "output": "json",
                "features": [
                    {
                        "type": "pipeline",
                        "name": "default",
                        "page_classifier": {"enabled": True},
                        "kv_extraction": (
                            {"enabled": True, "keys": keys}
                            if keys
                            else {"enabled": True}
                        ),
                        "ocr": {
                            "document": {"model": "default"},
                            "region": {"model": "best"},
                        },
                    }
                ],
            }

            response = requests.post(
                endpoint_url,
                headers=self.config.headers,
                json=payload,
                timeout=120,
            )

            if response.status_code != 200:
                return ToolOutput(
                    content=json.dumps(
                        {
                            "error": f"API request failed with status {response.status_code}",
                            "details": response.text[:500],
                        }
                    ),
                    tool_name=self.name,
                    raw_input=kwargs,
                    raw_output={"error": "API request failed"},
                    is_error=True,
                )

            result = response.json()
            result["document_path"] = document_path
            result["method"] = "marie_api"
            if keys:
                result["requested_keys"] = keys

            return ToolOutput(
                content=json.dumps(result, indent=2),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output=result,
                is_error=False,
            )

        except requests.exceptions.ConnectionError:
            return ToolOutput(
                content=json.dumps(
                    {
                        "error": "Cannot connect to Marie API",
                        "api_url": self.config.api_base_url,
                        "hint": "Ensure Marie server is running",
                    }
                ),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"error": "Connection failed"},
                is_error=True,
            )
        except Exception as e:
            return ToolOutput(
                content=json.dumps({"error": str(e), "document_path": document_path}),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"error": str(e)},
                is_error=True,
            )


# =============================================================================
# Function-Based Tools
# =============================================================================


@register_tool("document_info")
def document_info(document_path: str) -> str:
    """Get metadata and basic information about a document.

    Args:
        document_path: Path to the document file

    Returns:
        JSON string with document information.
    """
    path = Path(document_path)

    if not path.exists():
        return json.dumps({"error": f"File not found: {document_path}"})

    info = {
        "file_path": str(path.absolute()),
        "file_name": path.name,
        "extension": path.suffix.lower(),
        "file_size_bytes": path.stat().st_size,
        "file_size_kb": round(path.stat().st_size / 1024, 2),
        "modified_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
    }

    # Get image-specific info
    if path.suffix.lower() in [
        '.png',
        '.jpg',
        '.jpeg',
        '.tiff',
        '.tif',
        '.bmp',
        '.gif',
    ]:
        try:
            from PIL import Image

            with Image.open(path) as img:
                info["image_width"] = img.width
                info["image_height"] = img.height
                info["image_mode"] = img.mode
                info["image_format"] = img.format
        except Exception:
            pass

    # Get PDF-specific info
    if path.suffix.lower() == '.pdf':
        try:
            import PyPDF2

            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                info["page_count"] = len(reader.pages)
                if reader.metadata:
                    info["pdf_metadata"] = {
                        k: str(v)
                        for k, v in reader.metadata.items()
                        if v and not k.startswith('/')
                    }
        except Exception:
            pass

    return json.dumps(info, indent=2)


@register_tool("check_marie_status")
def check_marie_status() -> str:
    """Check if Marie API server is online and accessible.

    Returns:
        JSON string with connection status and API info.
    """
    config = get_api_config()

    try:
        response = requests.head(config.api_base_url, timeout=5)
        online = response.status_code in [200, 308]

        return json.dumps(
            {
                "online": online,
                "api_base_url": config.api_base_url,
                "status_code": response.status_code,
                "has_api_key": bool(config.api_key),
            }
        )
    except requests.exceptions.ConnectionError:
        return json.dumps(
            {
                "online": False,
                "api_base_url": config.api_base_url,
                "error": "Connection refused",
                "hint": "Start Marie server with: marie server start",
            }
        )
    except Exception as e:
        return json.dumps(
            {
                "online": False,
                "api_base_url": config.api_base_url,
                "error": str(e),
            }
        )


# =============================================================================
# Agent Initialization
# =============================================================================


def init_document_agent(
    backend: str = "marie",
    model: Optional[str] = None,
    use_planning: bool = False,
    api_config: Optional[MarieAPIConfig] = None,
) -> AssistantAgent:
    """Initialize the document processing agent.

    Args:
        backend: LLM backend to use ("marie" or "openai")
        model: Model name
        use_planning: If True, use PlanningAgent for multi-step tasks
        api_config: Optional Marie API configuration override

    Returns:
        Configured agent instance.
    """
    if api_config:
        set_api_config(api_config.api_base_url, api_config.api_key, api_config.queue_id)

    if backend == "marie":
        llm = MarieEngineLLMWrapper(engine_name=model or "qwen2_5_vl_7b")
    else:
        from marie.agent import OpenAICompatibleWrapper

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable required for OpenAI backend"
            )
        llm = OpenAICompatibleWrapper(
            model=model or "gpt-4o",
            api_key=api_key,
            api_base="https://api.openai.com/v1",
        )

    # Initialize document processing tools (using Marie API)
    tools = [
        OCRTool(),
        DocumentClassifierTool(),
        TableExtractorTool(),
        NERExtractorTool(),
        KeyValueExtractorTool(),
        "document_info",
        "check_marie_status",
    ]

    system_message = """You are a document processing AI assistant specialized in Visual Document Understanding.

You have access to Marie's document analysis tools (all use Marie's API):

1. **document_ocr**: Extract text from documents with high-quality OCR
   - Modes: word, sparse, line, raw_line, multiline (default)
   - Returns structured text with bounding boxes and confidence scores

2. **document_classifier**: Classify document type using ML models
   - Returns document category with confidence scores

3. **table_extractor**: Detect and extract tables from documents
   - Returns structured table data with rows and columns

4. **ner_extractor**: Extract named entities (dates, amounts, names, etc.)
   - Uses Marie's ML-based NER models

5. **kv_extractor**: Extract key-value pairs from forms and invoices
   - Can target specific keys if provided

6. **document_info**: Get file metadata and basic document information

7. **check_marie_status**: Verify Marie API server is accessible

When processing a document:
1. First check Marie status to ensure the API is available
2. Get document info to understand what you're working with
3. Classify the document type to guide your analysis
4. Use appropriate tools based on the document type and user's request
5. Combine results from multiple tools for comprehensive analysis
6. Always provide clear, structured output

For complex requests, break down the task into steps and execute them systematically.

Note: All document processing is done via Marie's API endpoints. Ensure the Marie server
is running before processing documents."""

    AgentClass = PlanningAgent if use_planning else AssistantAgent

    return AgentClass(
        llm=llm,
        function_list=tools,
        name="Marie Document Agent",
        description="AI agent for comprehensive document understanding using Marie's API.",
        system_message=system_message,
        max_iterations=15 if use_planning else 10,
    )


# =============================================================================
# Running Modes
# =============================================================================


def process_document(
    document_path: str,
    task: str = "Extract all information from this document",
    use_planning: bool = False,
    backend: str = "marie",
):
    """Process a document with the agent.

    Args:
        document_path: Path to the document
        task: Task description
        use_planning: Whether to use the planning agent
        backend: LLM backend to use
    """
    print(f"Document: {document_path}")
    print(f"Task: {task}")
    print(f"Mode: {'Planning' if use_planning else 'Standard'}")
    print("=" * 60)

    agent = init_document_agent(use_planning=use_planning, backend=backend)

    # Create message with document
    messages = [
        {
            "role": "user",
            "content": [
                {"file": document_path},
                {"text": task},
            ],
        }
    ]

    print("\nAgent Processing...\n")

    for responses in agent.run(messages=messages):
        if responses:
            last = responses[-1]
            content = (
                last.get("content", "") if isinstance(last, dict) else last.content
            )
            if content:
                print(content)
                print("-" * 40)

    print("\n" + "=" * 60)


def run_interactive():
    """Run the document agent in interactive TUI mode."""
    print("=" * 60)
    print("Marie Document Agent - Interactive Mode")
    print("=" * 60)
    print("Commands:")
    print("  doc <path>    - Set document to process")
    print("  plan          - Toggle planning mode")
    print("  info          - Get info about current document")
    print("  status        - Check Marie API status")
    print("  quit/exit     - End session")
    print()

    use_planning = False
    current_doc = None
    agent = init_document_agent(use_planning=use_planning)
    messages = []

    while True:
        try:
            mode = "[PLAN] " if use_planning else ""
            user_input = input(f"\n{mode}You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower().startswith("doc "):
            current_doc = user_input[4:].strip()
            if Path(current_doc).exists():
                print(f"Document set to: {current_doc}")
                info = document_info(current_doc)
                print(f"Info: {info}")
            else:
                print(f"Warning: File not found: {current_doc}")
            continue

        if user_input.lower() == "plan":
            use_planning = not use_planning
            agent = init_document_agent(use_planning=use_planning)
            messages = []
            print(f"Planning mode: {'ON' if use_planning else 'OFF'}")
            continue

        if user_input.lower() == "info" and current_doc:
            info = document_info(current_doc)
            print(info)
            continue

        if user_input.lower() == "status":
            status = check_marie_status()
            print(status)
            continue

        # Build message
        if current_doc:
            content = [
                {"file": current_doc},
                {"text": user_input},
            ]
        else:
            content = user_input

        messages.append({"role": "user", "content": content})

        print("\nAgent: ", end="", flush=True)

        response_list = []
        for response_list in agent.run(messages=messages):
            if response_list:
                last = response_list[-1]
                content = (
                    last.get("content", "") if isinstance(last, dict) else last.content
                )
                if content:
                    print(content)

        if response_list:
            for r in response_list:
                messages.append(r if isinstance(r, dict) else r.model_dump())


def run_demo():
    """Run a demo showing Marie document processing capabilities."""
    print("\n" + "=" * 60)
    print("Marie Document Agent - Demo")
    print("=" * 60)

    print("\nThis agent uses Marie's document processing API for:")
    print()
    print("1. OCR (document_ocr)")
    print("   - Endpoint: /api/document/extract")
    print("   - Modes: word, sparse, line, raw_line, multiline")
    print("   - Returns: Text with bounding boxes and confidence scores")
    print()
    print("2. Document Classification (document_classifier)")
    print("   - Endpoint: /api/document/classify")
    print("   - ML-based document type classification")
    print("   - Returns: Document category with confidence")
    print()
    print("3. Table Extraction (table_extractor)")
    print("   - Endpoint: /api/document/extract (with table features)")
    print("   - Returns: Structured table data")
    print()
    print("4. NER Extraction (ner_extractor)")
    print("   - Endpoint: /api/ner/extract")
    print("   - Returns: Named entities (dates, amounts, names, etc.)")
    print()
    print("5. Key-Value Extraction (kv_extractor)")
    print("   - Endpoint: /api/document/extract (with KV features)")
    print("   - Returns: Key-value pairs from forms")
    print()

    print("Configuration:")
    config = get_api_config()
    print(f"  MARIE_API_URL: {config.api_base_url}")
    print(f"  MARIE_API_KEY: {'***' if config.api_key else '(not set)'}")
    print(f"  MARIE_QUEUE_ID: {config.queue_id}")
    print()

    print("Usage:")
    print(
        "  python document_agent.py --document path/to/doc.png --task 'Extract all text'"
    )
    print("  python document_agent.py --tui")
    print()

    # Check API status
    print("API Status:")
    status = check_marie_status()
    status_data = json.loads(status)
    if status_data.get("online"):
        print("  [ONLINE] Marie API is accessible")
    else:
        print("  [OFFLINE] Marie API is not accessible")
        if "hint" in status_data:
            print(f"  Hint: {status_data['hint']}")
    print()


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marie Document Agent Example")
    parser.add_argument("--document", type=str, help="Path to document file")
    parser.add_argument(
        "--task", type=str, default="Extract all information from this document"
    )
    parser.add_argument(
        "--tui", action="store_true", help="Run in interactive TUI mode"
    )
    parser.add_argument("--planning", action="store_true", help="Use planning agent")
    parser.add_argument(
        "--demo", action="store_true", help="Show demo and configuration"
    )
    parser.add_argument("--backend", default="marie", choices=["marie", "openai"])
    parser.add_argument("--api-url", type=str, help="Marie API base URL")
    parser.add_argument("--api-key", type=str, help="Marie API key")

    args = parser.parse_args()

    # Set API config from command line if provided
    if args.api_url or args.api_key:
        set_api_config(api_base_url=args.api_url, api_key=args.api_key)

    if args.demo:
        run_demo()
    elif args.tui:
        run_interactive()
    elif args.document:
        process_document(args.document, args.task, args.planning, args.backend)
    else:
        run_demo()
