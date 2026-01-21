"""Vision Document Agent Example.

This example demonstrates the VisionDocumentAgent which specializes in
Visual Document Understanding (VDU) tasks using design patterns.

The agent automatically:
1. Categorizes document tasks (OCR, table extraction, form extraction, etc.)
2. Suggests appropriate design patterns
3. Uses relevant tools for the task
4. Verifies results when configured

Shows:
- OCR via Marie's /api/document/extract endpoint
- Table detection via Marie's pipeline features
- Layout analysis via Marie's API
- Named entity extraction via /api/ner/extract
- Document classification via /api/document/classify
- Visual question answering (pattern-based + optionally via VQA executor)

Usage:
    python vision_document_agent.py --task "Extract all tables" --image doc.png
    python vision_document_agent.py --demo table
    python vision_document_agent.py --tui
"""

import argparse
import base64
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from marie.agent import (
    AgentTool,
    DocumentExtractionAgent,
    DocumentQAAgent,
    MarieEngineLLMWrapper,
    OpenAICompatibleWrapper,
    ToolMetadata,
    ToolOutput,
    VisionDocumentAgent,
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


# Global config instance
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


def _encode_file(file_path: str) -> str:
    """Encode file to base64."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _check_api_available() -> bool:
    """Check if Marie API is available."""
    config = get_api_config()
    try:
        response = requests.head(config.api_base_url, timeout=3)
        return response.status_code in [200, 308]
    except:
        return False


# =============================================================================
# Document Processing Tools (using Marie API)
# =============================================================================


@register_tool("ocr")
def ocr(image: str, language: str = "eng", mode: str = "multiline") -> str:
    """Extract text from a document image using Marie's OCR API.

    Args:
        image: Path to the image file
        language: Language code for OCR (e.g., "eng", "deu", "fra")
        mode: OCR mode (word, sparse, line, raw_line, multiline)

    Returns:
        JSON string with extracted text and confidence scores.
    """
    if not image or not Path(image).exists():
        return json.dumps({"error": f"Image not found: {image}"})

    config = get_api_config()

    try:
        base64_data = _encode_file(image)
        uid = str(uuid.uuid4())

        endpoint_url = f"{config.api_base_url}/document/extract"
        payload = {
            "queue_id": config.queue_id,
            "data": base64_data,
            "doc_id": f"ocr-{uid}",
            "doc_type": "agent_ocr",
            "mode": mode,
            "output": "json",
            "features": [
                {
                    "type": "pipeline",
                    "name": "default",
                    "page_classifier": {"enabled": True},
                    "page_splitter": {"enabled": True},
                    "ocr": {
                        "document": {"model": "default"},
                        "region": {"model": "best"},
                    },
                }
            ],
        }

        response = requests.post(
            endpoint_url,
            headers=config.headers,
            json=payload,
            timeout=120,
        )

        if response.status_code != 200:
            return json.dumps(
                {
                    "error": f"API request failed with status {response.status_code}",
                    "details": response.text[:500],
                }
            )

        result = response.json()
        result["image"] = image
        result["language"] = language
        result["method"] = "marie_api"

        return json.dumps(result, indent=2)

    except requests.exceptions.ConnectionError:
        return json.dumps(
            {
                "error": "Cannot connect to Marie API",
                "api_url": config.api_base_url,
                "hint": "Ensure Marie server is running: marie server start",
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e), "image": image})


@register_tool("detect_tables")
def detect_tables(image: str) -> str:
    """Detect tables in a document image using Marie's API.

    Args:
        image: Path to the image file

    Returns:
        JSON string with detected table regions and structure.
    """
    if not image or not Path(image).exists():
        return json.dumps({"error": f"Image not found: {image}"})

    config = get_api_config()

    try:
        base64_data = _encode_file(image)
        uid = str(uuid.uuid4())

        endpoint_url = f"{config.api_base_url}/document/extract"
        payload = {
            "queue_id": config.queue_id,
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
            headers=config.headers,
            json=payload,
            timeout=120,
        )

        if response.status_code != 200:
            return json.dumps(
                {
                    "error": f"API request failed with status {response.status_code}",
                    "details": response.text[:500],
                }
            )

        result = response.json()
        result["image"] = image
        result["method"] = "marie_api"

        return json.dumps(result, indent=2)

    except requests.exceptions.ConnectionError:
        return json.dumps(
            {
                "error": "Cannot connect to Marie API",
                "hint": "Ensure Marie server is running",
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e), "image": image})


@register_tool("extract_table_structure")
def extract_table_structure(image: str, table_bbox: str = "") -> str:
    """Extract structured data from a detected table using Marie's API.

    Args:
        image: Path to the image file
        table_bbox: Bounding box of the table region as JSON string "[x1,y1,x2,y2]"

    Returns:
        JSON string with table rows and cells.
    """
    if not image or not Path(image).exists():
        return json.dumps({"error": f"Image not found: {image}"})

    config = get_api_config()

    try:
        base64_data = _encode_file(image)
        uid = str(uuid.uuid4())

        # Build features with optional region of interest
        features = [
            {
                "type": "pipeline",
                "name": "default",
                "table_extraction": {"enabled": True},
                "ocr": {
                    "document": {"model": "default"},
                    "region": {"model": "best"},
                },
            }
        ]

        # Add bbox if provided
        if table_bbox:
            try:
                bbox = json.loads(table_bbox)
                features[0]["region_of_interest"] = bbox
            except json.JSONDecodeError:
                pass

        endpoint_url = f"{config.api_base_url}/document/extract"
        payload = {
            "queue_id": config.queue_id,
            "data": base64_data,
            "doc_id": f"table-struct-{uid}",
            "doc_type": "agent_table_struct",
            "mode": "multiline",
            "output": "json",
            "features": features,
        }

        response = requests.post(
            endpoint_url,
            headers=config.headers,
            json=payload,
            timeout=120,
        )

        if response.status_code != 200:
            return json.dumps(
                {
                    "error": f"API request failed with status {response.status_code}",
                    "details": response.text[:500],
                }
            )

        result = response.json()
        result["image"] = image
        result["method"] = "marie_api"

        return json.dumps(result, indent=2)

    except requests.exceptions.ConnectionError:
        return json.dumps(
            {
                "error": "Cannot connect to Marie API",
                "hint": "Ensure Marie server is running",
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


@register_tool("classify_document")
def classify_document(image: str) -> str:
    """Classify the type of document using Marie's classification API.

    Args:
        image: Path to the image file

    Returns:
        JSON string with document classification.
    """
    if not image or not Path(image).exists():
        return json.dumps({"error": f"Image not found: {image}"})

    config = get_api_config()

    try:
        base64_data = _encode_file(image)
        uid = str(uuid.uuid4())

        endpoint_url = f"{config.api_base_url}/document/classify"
        payload = {
            "queue_id": config.queue_id,
            "data": base64_data,
            "doc_id": f"classify-{uid}",
            "doc_type": "agent_classify",
            "mode": "multiline",
            "output": "json",
            "pipeline": "default",
            "features": [],
        }

        response = requests.post(
            endpoint_url,
            headers=config.headers,
            json=payload,
            timeout=60,
        )

        if response.status_code != 200:
            return json.dumps(
                {
                    "error": f"API request failed with status {response.status_code}",
                    "details": response.text[:500],
                }
            )

        result = response.json()
        result["image"] = image
        result["method"] = "marie_api"

        return json.dumps(result, indent=2)

    except requests.exceptions.ConnectionError:
        return json.dumps(
            {
                "error": "Cannot connect to Marie API",
                "hint": "Ensure Marie server is running",
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


@register_tool("extract_key_value")
def extract_key_value(image: str, fields: str = "") -> str:
    """Extract key-value pairs from a document using Marie's API.

    Args:
        image: Path to the image file
        fields: Optional JSON list of field names to look for

    Returns:
        JSON string with extracted key-value pairs.
    """
    if not image or not Path(image).exists():
        return json.dumps({"error": f"Image not found: {image}"})

    config = get_api_config()

    # Parse requested fields
    target_fields = []
    if fields:
        try:
            target_fields = json.loads(fields)
        except json.JSONDecodeError:
            target_fields = [f.strip() for f in fields.split(',')]

    try:
        base64_data = _encode_file(image)
        uid = str(uuid.uuid4())

        endpoint_url = f"{config.api_base_url}/document/extract"
        payload = {
            "queue_id": config.queue_id,
            "data": base64_data,
            "doc_id": f"kv-{uid}",
            "doc_type": "agent_kv",
            "mode": "multiline",
            "output": "json",
            "features": [
                {
                    "type": "pipeline",
                    "name": "default",
                    "kv_extraction": {
                        "enabled": True,
                        "keys": target_fields if target_fields else None,
                    },
                    "ocr": {
                        "document": {"model": "default"},
                        "region": {"model": "best"},
                    },
                }
            ],
        }

        response = requests.post(
            endpoint_url,
            headers=config.headers,
            json=payload,
            timeout=120,
        )

        if response.status_code != 200:
            return json.dumps(
                {
                    "error": f"API request failed with status {response.status_code}",
                    "details": response.text[:500],
                }
            )

        result = response.json()
        result["image"] = image
        result["method"] = "marie_api"
        if target_fields:
            result["requested_fields"] = target_fields

        return json.dumps(result, indent=2)

    except requests.exceptions.ConnectionError:
        return json.dumps(
            {
                "error": "Cannot connect to Marie API",
                "hint": "Ensure Marie server is running",
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


@register_tool("extract_entities")
def extract_entities(text: str = "", image: str = "") -> str:
    """Extract named entities from text or document image using Marie's NER API.

    Args:
        text: Text to extract entities from
        image: Path to image (if text not provided, will process the image)

    Returns:
        JSON string with extracted entities.
    """
    config = get_api_config()

    # If image provided, use Marie's NER endpoint
    if image and Path(image).exists():
        try:
            base64_data = _encode_file(image)
            uid = str(uuid.uuid4())

            endpoint_url = f"{config.api_base_url}/ner/extract"
            payload = {
                "queue_id": config.queue_id,
                "data": base64_data,
                "doc_id": f"ner-{uid}",
                "doc_type": "agent_ner",
                "mode": "multiline",
                "output": "json",
            }

            response = requests.post(
                endpoint_url,
                headers=config.headers,
                json=payload,
                timeout=60,
            )

            if response.status_code != 200:
                return json.dumps(
                    {
                        "error": f"API request failed with status {response.status_code}",
                        "details": response.text[:500],
                    }
                )

            result = response.json()
            result["image"] = image
            result["method"] = "marie_api"

            return json.dumps(result, indent=2)

        except requests.exceptions.ConnectionError:
            return json.dumps(
                {
                    "error": "Cannot connect to Marie API",
                    "hint": "Ensure Marie server is running",
                }
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    elif text:
        # Fallback to regex-based extraction for text-only input
        PATTERNS = {
            "DATE": [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b',
            ],
            "MONEY": [
                r'[$€£]\s*[\d,]+\.?\d*',
                r'\b\d+[,\d]*\.\d{2}\b(?:\s*(?:USD|EUR|GBP))?',
            ],
            "EMAIL": [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
            "PHONE": [r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'],
            "INVOICE_NUMBER": [r'\b(?:INV|Invoice|Inv)[#:\s-]*([A-Z0-9-]+)\b'],
        }

        entities = []
        for entity_type, patterns in PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    value = match.group(0)
                    if not any(
                        e["text"] == value and e["type"] == entity_type
                        for e in entities
                    ):
                        entities.append(
                            {
                                "text": value,
                                "type": entity_type,
                                "start": match.start(),
                                "end": match.end(),
                                "confidence": 0.9,
                            }
                        )

        entities.sort(key=lambda x: x["start"])

        return json.dumps(
            {
                "entities": entities,
                "count": len(entities),
                "types_found": list(set(e["type"] for e in entities)),
                "method": "regex_fallback",
            }
        )

    return json.dumps({"error": "No text or image provided"})


@register_tool("vqa")
def vqa(image: str, question: str) -> str:
    """Answer a question about a document image.

    Uses Marie's VQA executor if available, otherwise falls back to
    extracting entities and pattern matching.

    Args:
        image: Path to the image file
        question: Question to answer about the document

    Returns:
        JSON string with the answer.
    """
    if not image or not Path(image).exists():
        return json.dumps({"error": f"Image not found: {image}"})

    config = get_api_config()

    # Try Marie's VQA endpoint first (if available)
    vqa_executor_url = os.getenv("MARIE_VQA_EXECUTOR_URL")
    if vqa_executor_url:
        try:
            base64_data = _encode_file(image)

            response = requests.post(
                vqa_executor_url,
                headers=config.headers,
                json={
                    "data": base64_data,
                    "question": question,
                },
                timeout=60,
            )

            if response.status_code == 200:
                result = response.json()
                result["method"] = "marie_vqa_executor"
                return json.dumps(result, indent=2)
        except:
            pass

    # Fallback: Extract entities and use pattern matching
    try:
        # Get entities from image
        entities_result = extract_entities(image=image)
        entities_data = json.loads(entities_result)
        entities = entities_data.get("entities", [])

        question_lower = question.lower()
        answer = None
        confidence = 0.0

        # Total/amount questions
        if any(
            w in question_lower for w in ["total", "amount", "cost", "price", "due"]
        ):
            money_entities = [e for e in entities if e.get("type") == "MONEY"]
            if money_entities:
                amounts = []
                for e in money_entities:
                    try:
                        val = float(re.sub(r'[^\d.]', '', e["text"]))
                        amounts.append((val, e["text"]))
                    except:
                        pass
                if amounts:
                    amounts.sort(reverse=True)
                    answer = f"The total/amount is {amounts[0][1]}"
                    confidence = 0.85

        # Date questions
        elif any(w in question_lower for w in ["date", "when"]):
            date_entities = [e for e in entities if e.get("type") == "DATE"]
            if date_entities:
                answer = f"Date found: {date_entities[0]['text']}"
                confidence = 0.82

        # Invoice/document number
        elif any(w in question_lower for w in ["invoice", "number", "id", "reference"]):
            inv_entities = [e for e in entities if e.get("type") == "INVOICE_NUMBER"]
            if inv_entities:
                answer = f"The invoice/reference number is {inv_entities[0]['text']}"
                confidence = 0.88

        # Fallback
        if not answer:
            # Get document classification for context
            classify_result = classify_document(image)
            classify_data = json.loads(classify_result)
            doc_type = classify_data.get("type", "document")
            answer = (
                f"This appears to be a {doc_type}. Please ask a more specific question."
            )
            confidence = 0.50

        return json.dumps(
            {
                "question": question,
                "answer": answer,
                "confidence": round(confidence, 2),
                "method": "pattern_matching",
                "hint": "For better VQA, set MARIE_VQA_EXECUTOR_URL",
            }
        )

    except Exception as e:
        return json.dumps({"error": str(e)})


@register_tool("detect_layout")
def detect_layout(image: str) -> str:
    """Detect the layout structure of a document using Marie's API.

    Args:
        image: Path to the image file

    Returns:
        JSON string with layout regions.
    """
    if not image or not Path(image).exists():
        return json.dumps({"error": f"Image not found: {image}"})

    config = get_api_config()

    try:
        base64_data = _encode_file(image)
        uid = str(uuid.uuid4())

        endpoint_url = f"{config.api_base_url}/document/extract"
        payload = {
            "queue_id": config.queue_id,
            "data": base64_data,
            "doc_id": f"layout-{uid}",
            "doc_type": "agent_layout",
            "mode": "multiline",
            "output": "json",
            "features": [
                {
                    "type": "pipeline",
                    "name": "default",
                    "layout_analysis": {"enabled": True},
                    "page_classifier": {"enabled": True},
                    "ocr": {
                        "document": {"model": "default"},
                        "region": {"model": "best"},
                    },
                }
            ],
        }

        response = requests.post(
            endpoint_url,
            headers=config.headers,
            json=payload,
            timeout=120,
        )

        if response.status_code != 200:
            return json.dumps(
                {
                    "error": f"API request failed with status {response.status_code}",
                    "details": response.text[:500],
                }
            )

        result = response.json()
        result["image"] = image
        result["method"] = "marie_api"

        return json.dumps(result, indent=2)

    except requests.exceptions.ConnectionError:
        return json.dumps(
            {
                "error": "Cannot connect to Marie API",
                "hint": "Ensure Marie server is running",
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


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


def init_vision_document_agent(
    backend: str = "marie",
    model: Optional[str] = None,
    agent_type: str = "vision",
) -> VisionDocumentAgent:
    """Initialize a VisionDocumentAgent.

    Args:
        backend: LLM backend to use ("marie" or "openai")
        model: Model name to use
        agent_type: Type of agent ("vision", "extraction", "qa")

    Returns:
        Configured agent instance.
    """
    if backend == "marie":
        llm = MarieEngineLLMWrapper(engine_name=model or "qwen2_5_vl_7b")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI backend")
        llm = OpenAICompatibleWrapper(
            model=model or "gpt-4o",
            api_key=api_key,
            api_base="https://api.openai.com/v1",
        )

    # Document processing tools (all use Marie API)
    tools = [
        "ocr",
        "detect_tables",
        "extract_table_structure",
        "classify_document",
        "extract_key_value",
        "extract_entities",
        "vqa",
        "detect_layout",
        "check_marie_status",
    ]

    # Select agent class based on type
    if agent_type == "extraction":
        agent_class = DocumentExtractionAgent
        name = "Marie Document Extraction Agent"
        description = "Agent for extracting structured data from documents"
    elif agent_type == "qa":
        agent_class = DocumentQAAgent
        name = "Marie Document QA Agent"
        description = "Agent for answering questions about documents"
    else:
        agent_class = VisionDocumentAgent
        name = "Marie Vision Document Agent"
        description = "Agent for Visual Document Understanding tasks"

    return agent_class(
        llm=llm,
        function_list=tools,
        name=name,
        description=description,
    )


def run_task(task: str, image_path: Optional[str] = None, agent_type: str = "vision"):
    """Run a document processing task.

    Args:
        task: Task description
        image_path: Optional path to document image
        agent_type: Type of agent to use
    """
    print("=" * 60)
    print("VISION DOCUMENT AGENT")
    print("=" * 60)
    print(f"\nTask: {task}")
    if image_path:
        print(f"Image: {image_path}")
    print("-" * 60)

    agent = init_vision_document_agent(agent_type=agent_type)

    # Show task analysis
    task_info = agent.get_task_info(task)
    print(f"\nTask Analysis:")
    print(f"  Category: {task_info['category']}")
    print(f"  Suggested Tools: {', '.join(task_info['suggested_tools'])}")
    print(f"  Available Tools: {', '.join(task_info['available_tools'])}")
    print()

    # Build message
    if image_path:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task},
                    {"type": "image", "image": image_path},
                ],
            }
        ]
    else:
        messages = [{"role": "user", "content": task}]

    print("Agent Execution:\n")
    print("-" * 60)

    for responses in agent.run(messages=messages):
        if responses:
            last = responses[-1]
            content = (
                last.get("content", "") if isinstance(last, dict) else last.content
            )
            if content:
                print(content)
                print()

    print("-" * 60)


# =============================================================================
# Demo Tasks
# =============================================================================


def demo_table_extraction():
    """Demo: Extract tables from a document."""
    print("\n" + "=" * 60)
    print("Demo: Table Extraction")
    print("=" * 60)
    print("\nThis demo shows how the table extraction tools work:")
    print()
    print("1. detect_tables - Uses Marie's table detection API")
    print("   - Endpoint: /api/document/extract with table_detection feature")
    print("   - Returns bounding boxes and estimated dimensions")
    print()
    print("2. extract_table_structure - Extracts cell data")
    print("   - Endpoint: /api/document/extract with table_extraction feature")
    print("   - Parses text into rows and columns")
    print("   - Identifies headers")
    print()
    print("Usage with real document:")
    print(
        "  python vision_document_agent.py --task 'Extract tables' --image invoice.png"
    )
    print()


def demo_invoice_processing():
    """Demo: Full invoice processing workflow."""
    print("\n" + "=" * 60)
    print("Demo: Invoice Processing")
    print("=" * 60)
    print("\nThis demo shows the invoice processing workflow:")
    print()
    print("1. classify_document - Identifies document type via /api/document/classify")
    print("2. ocr - Extracts all text via /api/document/extract")
    print("3. extract_key_value - Gets field values with KV extraction feature")
    print("4. extract_entities - Finds entities via /api/ner/extract")
    print("5. detect_tables - Locates line items table")
    print()
    print("Usage:")
    print(
        "  python vision_document_agent.py --task 'Process invoice' --image invoice.png"
    )
    print()


def demo_document_qa():
    """Demo: Document question answering."""
    print("\n" + "=" * 60)
    print("Demo: Document QA")
    print("=" * 60)
    print("\nThe vqa tool answers questions about documents:")
    print()
    print("- Uses Marie's VQA executor if MARIE_VQA_EXECUTOR_URL is set")
    print("- Falls back to entity extraction + pattern matching")
    print("- Works well for:")
    print("  - 'What is the total amount?'")
    print("  - 'When is the due date?'")
    print("  - 'What is the invoice number?'")
    print()
    print("For advanced VQA, set MARIE_VQA_EXECUTOR_URL to a VQA endpoint")
    print()
    print("Usage:")
    print(
        "  python vision_document_agent.py --task 'What is the total?' --image invoice.png --type qa"
    )
    print()


def demo_form_extraction():
    """Demo: Form field extraction."""
    print("\n" + "=" * 60)
    print("Demo: Form Extraction")
    print("=" * 60)
    print("\nThe extract_key_value tool finds form fields:")
    print()
    print("- Uses Marie's /api/document/extract with kv_extraction feature")
    print("- Can target specific fields if requested")
    print("- Returns structured key-value pairs")
    print()
    print("Usage:")
    print(
        "  python vision_document_agent.py --task 'Extract all fields' --image form.png"
    )
    print()


def run_demo():
    """Show available tools and API status."""
    print("\n" + "=" * 60)
    print("Marie Vision Document Agent - Demo")
    print("=" * 60)
    print("\nThis agent uses Marie's document processing API for:")
    print()
    print("Tools available (all use Marie API):")
    print("  - ocr: Text extraction via /api/document/extract")
    print("  - detect_tables: Table detection via extract API")
    print("  - extract_table_structure: Table parsing")
    print("  - classify_document: Classification via /api/document/classify")
    print("  - extract_key_value: Form field extraction")
    print("  - extract_entities: NER via /api/ner/extract")
    print("  - vqa: Visual question answering")
    print("  - detect_layout: Layout analysis")
    print("  - check_marie_status: API connectivity check")
    print()

    # Show configuration
    config = get_api_config()
    print("Configuration:")
    print(f"  MARIE_API_URL: {config.api_base_url}")
    print(f"  MARIE_API_KEY: {'***' if config.api_key else '(not set)'}")
    print(f"  MARIE_QUEUE_ID: {config.queue_id}")
    print(
        f"  MARIE_VQA_EXECUTOR_URL: {os.getenv('MARIE_VQA_EXECUTOR_URL', '(not set)')}"
    )
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

    print("Usage examples:")
    print("  python vision_document_agent.py --tui")
    print("  python vision_document_agent.py --task 'Extract text' --image doc.png")
    print("  python vision_document_agent.py --demo table")
    print()


def app_tui():
    """Run the agent in interactive TUI mode."""
    print("=" * 60)
    print("Marie Vision Document Agent - Interactive Mode")
    print("=" * 60)
    print("Process documents with automatic task categorization.")
    print()
    print("Commands:")
    print("  image <path>  - Set document image")
    print("  info <task>   - Show task analysis without running")
    print("  status        - Check Marie API status")
    print("  quit/exit     - End session")
    print()

    agent = init_vision_document_agent()
    messages = []
    current_image = None

    while True:
        try:
            user_input = input("\nTask: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        # Set image command
        if user_input.lower().startswith("image "):
            current_image = user_input[6:].strip()
            if Path(current_image).exists():
                print(f"Image set: {current_image}")
            else:
                print(f"Warning: File not found: {current_image}")
            continue

        # Info command
        if user_input.lower().startswith("info "):
            task = user_input[5:].strip()
            info = agent.get_task_info(task)
            print(f"\nTask Analysis for: {task}")
            print(f"  Category: {info['category']}")
            print(f"  Suggested Tools: {', '.join(info['suggested_tools'])}")
            print(f"  Pattern Available: {'Yes' if info['pattern'] else 'No'}")
            continue

        # Status command
        if user_input.lower() == "status":
            status = check_marie_status()
            print(status)
            continue

        # Build message
        if current_image:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_input},
                        {"type": "image", "image": current_image},
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": user_input})

        print("\n" + "-" * 40)
        print("Processing...")
        print("-" * 40 + "\n")

        response_list = []
        for response_list in agent.run(messages=messages):
            if response_list:
                last = response_list[-1]
                content = (
                    last.get("content", "") if isinstance(last, dict) else last.content
                )
                if content:
                    print(content)

        print("\n" + "-" * 40)

        if response_list:
            for r in response_list:
                messages.append(r if isinstance(r, dict) else r.model_dump())


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marie Vision Document Agent Example")
    parser.add_argument("--task", type=str, help="Document processing task")
    parser.add_argument("--image", type=str, help="Path to document image")
    parser.add_argument(
        "--type", type=str, choices=["vision", "extraction", "qa"], default="vision"
    )
    parser.add_argument("--tui", action="store_true", help="Interactive mode")
    parser.add_argument(
        "--demo", type=str, choices=["table", "invoice", "qa", "form", "all"]
    )
    parser.add_argument("--backend", default="marie", choices=["marie", "openai"])
    parser.add_argument("--api-url", type=str, help="Marie API base URL")
    parser.add_argument("--api-key", type=str, help="Marie API key")

    args = parser.parse_args()

    # Set API config from command line if provided
    if args.api_url or args.api_key:
        set_api_config(api_base_url=args.api_url, api_key=args.api_key)

    if args.tui:
        app_tui()
    elif args.demo == "table":
        demo_table_extraction()
    elif args.demo == "invoice":
        demo_invoice_processing()
    elif args.demo == "qa":
        demo_document_qa()
    elif args.demo == "form":
        demo_form_extraction()
    elif args.demo == "all" or args.demo is None and not args.task:
        run_demo()
    elif args.task:
        run_task(args.task, image_path=args.image, agent_type=args.type)
    else:
        run_demo()
