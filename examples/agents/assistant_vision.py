"""Vision-Language Assistant Agent Example.

This example demonstrates how to create a vision-language assistant that can
analyze images and documents using Marie's executor infrastructure.

Shows:
- Integration with Marie executors via ExecutorTool wrapper
- Multimodal message handling (images + text)
- Real image processing operations
- Vision-language model integration

Usage:
    python assistant_vision.py --image path/to/image.jpg --query "Describe this document"
    python assistant_vision.py --tui
"""

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from marie.agent import (
    AgentTool,
    ContentItem,
    Message,
    ReactAgent,
    ToolMetadata,
    ToolOutput,
    register_tool,
)

# Load environment variables from .env file
load_dotenv()


@register_tool("image_info")
def image_info(image_path: str) -> str:
    """Get metadata and basic information about an image.

    Args:
        image_path: Path to the image file

    Returns:
        JSON string with image information.
    """
    try:
        from PIL import Image

        path = Path(image_path)
        if not path.exists():
            return json.dumps({"error": f"File not found: {image_path}"})

        with Image.open(path) as img:
            info = {
                "file_path": str(path.absolute()),
                "file_name": path.name,
                "format": img.format,
                "mode": img.mode,
                "width": img.width,
                "height": img.height,
                "aspect_ratio": round(img.width / img.height, 2),
                "file_size_bytes": path.stat().st_size,
                "file_size_kb": round(path.stat().st_size / 1024, 2),
            }

            # Get EXIF data if available
            if hasattr(img, '_getexif') and img._getexif():
                info["has_exif"] = True
            else:
                info["has_exif"] = False

        return json.dumps(info)

    except ImportError:
        return json.dumps(
            {"error": "PIL/Pillow not installed. Run: pip install Pillow"}
        )
    except Exception as e:
        return json.dumps({"error": str(e), "image_path": image_path})


@register_tool("crop_image")
def crop_image(image_path: str, region: str, output_path: Optional[str] = None) -> str:
    """Crop a region from an image.

    Args:
        image_path: Path to the image file
        region: Region to crop - either "top", "bottom", "left", "right", "center",
                or coordinates as "x1,y1,x2,y2"
        output_path: Optional path to save cropped image (defaults to temp file)

    Returns:
        JSON string with cropped image information.
    """
    try:
        import tempfile

        from PIL import Image

        with Image.open(image_path) as img:
            w, h = img.width, img.height

            # Parse region
            if region == "top":
                box = (0, 0, w, h // 2)
            elif region == "bottom":
                box = (0, h // 2, w, h)
            elif region == "left":
                box = (0, 0, w // 2, h)
            elif region == "right":
                box = (w // 2, 0, w, h)
            elif region == "center":
                margin_x, margin_y = w // 4, h // 4
                box = (margin_x, margin_y, w - margin_x, h - margin_y)
            elif "," in region:
                coords = [int(x.strip()) for x in region.split(",")]
                if len(coords) == 4:
                    box = tuple(coords)
                else:
                    return json.dumps({"error": "Coordinates must be x1,y1,x2,y2"})
            else:
                return json.dumps({"error": f"Unknown region: {region}"})

            cropped = img.crop(box)

            # Save to output path or temp file
            if output_path:
                save_path = output_path
            else:
                suffix = Path(image_path).suffix or ".png"
                fd, save_path = tempfile.mkstemp(suffix=suffix)
                os.close(fd)

            cropped.save(save_path)

            return json.dumps(
                {
                    "success": True,
                    "original_size": [w, h],
                    "crop_box": list(box),
                    "cropped_size": [cropped.width, cropped.height],
                    "output_path": save_path,
                }
            )

    except ImportError:
        return json.dumps({"error": "PIL/Pillow not installed"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@register_tool("resize_image")
def resize_image(
    image_path: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: Optional[float] = None,
    output_path: Optional[str] = None,
) -> str:
    """Resize an image.

    Args:
        image_path: Path to the image file
        width: Target width (maintains aspect ratio if height not specified)
        height: Target height (maintains aspect ratio if width not specified)
        scale: Scale factor (e.g., 0.5 for half size, 2.0 for double)
        output_path: Optional path to save resized image

    Returns:
        JSON string with resized image information.
    """
    try:
        import tempfile

        from PIL import Image

        with Image.open(image_path) as img:
            orig_w, orig_h = img.width, img.height

            if scale:
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
            elif width and height:
                new_w, new_h = width, height
            elif width:
                new_w = width
                new_h = int(orig_h * (width / orig_w))
            elif height:
                new_h = height
                new_w = int(orig_w * (height / orig_h))
            else:
                return json.dumps({"error": "Specify width, height, or scale"})

            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            if output_path:
                save_path = output_path
            else:
                suffix = Path(image_path).suffix or ".png"
                fd, save_path = tempfile.mkstemp(suffix=suffix)
                os.close(fd)

            resized.save(save_path)

            return json.dumps(
                {
                    "success": True,
                    "original_size": [orig_w, orig_h],
                    "new_size": [new_w, new_h],
                    "output_path": save_path,
                }
            )

    except ImportError:
        return json.dumps({"error": "PIL/Pillow not installed"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@register_tool("convert_image")
def convert_image(
    image_path: str, output_format: str, output_path: Optional[str] = None
) -> str:
    """Convert an image to a different format.

    Args:
        image_path: Path to the image file
        output_format: Target format (png, jpg, webp, bmp, gif)
        output_path: Optional path to save converted image

    Returns:
        JSON string with conversion result.
    """
    try:
        import tempfile

        from PIL import Image

        format_map = {
            "jpg": "JPEG",
            "jpeg": "JPEG",
            "png": "PNG",
            "webp": "WEBP",
            "bmp": "BMP",
            "gif": "GIF",
        }

        out_fmt = format_map.get(output_format.lower())
        if not out_fmt:
            return json.dumps(
                {
                    "error": f"Unknown format: {output_format}",
                    "supported": list(format_map.keys()),
                }
            )

        with Image.open(image_path) as img:
            # Convert RGBA to RGB for JPEG
            if out_fmt == "JPEG" and img.mode == "RGBA":
                img = img.convert("RGB")

            if output_path:
                save_path = output_path
            else:
                suffix = f".{output_format.lower()}"
                if suffix == ".jpg":
                    suffix = ".jpeg"
                fd, save_path = tempfile.mkstemp(suffix=suffix)
                os.close(fd)

            img.save(save_path, format=out_fmt)

            return json.dumps(
                {
                    "success": True,
                    "input_path": image_path,
                    "output_path": save_path,
                    "format": out_fmt,
                }
            )

    except ImportError:
        return json.dumps({"error": "PIL/Pillow not installed"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Executor Integration Tool (demonstrates Marie executor wrapping)
# =============================================================================


class ExecutorVisionTool(AgentTool):
    """Tool that wraps a Marie executor for vision tasks.

    This demonstrates how to integrate with Marie's executor infrastructure.
    In production, this would connect to a running Marie server.
    """

    def __init__(
        self, executor_name: str = "document_analysis", server_url: Optional[str] = None
    ):
        """Initialize the executor tool.

        Args:
            executor_name: Name of the Marie executor to use
            server_url: URL of the Marie server (defaults to localhost)
        """
        self.executor_name = executor_name
        self.server_url = server_url or os.getenv(
            "MARIE_SERVER_URL", "http://localhost:5000"
        )
        self._executor = None

    @property
    def name(self) -> str:
        return f"executor_{self.executor_name}"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description=f"Run {self.executor_name} executor for document/image analysis. "
            "Processes images through Marie's ML pipeline.",
            parameters={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to image file",
                    },
                    "task": {
                        "type": "string",
                        "description": "Task to perform (ocr, classify, detect)",
                    },
                },
                "required": ["image_path"],
            },
        )

    def _get_executor(self):
        """Lazy-load the executor connection."""
        if self._executor is None:
            try:
                # In production, this would connect to Marie server
                # For this example, we simulate the executor response
                pass
            except Exception:
                pass
        return self._executor

    def call(self, **kwargs) -> ToolOutput:
        """Execute the vision task."""
        image_path = kwargs.get("image_path", "")
        task = kwargs.get("task", "analyze")

        if not image_path:
            return ToolOutput(
                content=json.dumps({"error": "image_path is required"}),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"error": "image_path is required"},
                is_error=True,
            )

        # Check if file exists
        if not Path(image_path).exists():
            return ToolOutput(
                content=json.dumps({"error": f"File not found: {image_path}"}),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"error": "File not found"},
                is_error=True,
            )

        # In production, this would call the actual Marie executor
        # For demonstration, we show the expected interface
        try:
            # Example of how to call Marie executor (pseudo-code):
            # from marie.executor import DocumentAnalysisExecutor
            # executor = DocumentAnalysisExecutor()
            # result = executor.run(image_path, task=task)

            # For this example, return info about what would be processed
            result = {
                "executor": self.executor_name,
                "server_url": self.server_url,
                "image_path": image_path,
                "task": task,
                "status": "executor_not_connected",
                "message": "To use real executor, start Marie server and set MARIE_SERVER_URL",
                "example_usage": {
                    "start_server": "marie server start",
                    "set_env": "export MARIE_SERVER_URL=http://localhost:5000",
                },
            }

            return ToolOutput(
                content=json.dumps(result, indent=2),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output=result,
                is_error=False,
            )

        except Exception as e:
            return ToolOutput(
                content=json.dumps({"error": str(e)}),
                tool_name=self.name,
                raw_input=kwargs,
                raw_output={"error": str(e)},
                is_error=True,
            )


# =============================================================================
# Agent Initialization
# =============================================================================


def create_vision_assistant(
    backend: str = "marie", model: Optional[str] = None
) -> ReactAgent:
    """Create a vision-language assistant agent.

    Args:
        backend: LLM backend ("marie", "openai")
        model: Model name (should support vision)

    Returns:
        Configured ReactAgent instance.
    """
    from utils import create_llm

    llm = create_llm(backend=backend, model=model)

    tools = [
        "image_info",
        "crop_image",
        "resize_image",
        "convert_image",
        ExecutorVisionTool(executor_name="document_analysis"),
    ]

    return ReactAgent(
        llm=llm,
        function_list=tools,
        name="Vision Assistant",
        description="A vision-language assistant for image and document analysis.",
        system_message="""You are a vision-language AI assistant specialized in image and document analysis.

You can see and analyze images directly. You also have tools for image manipulation:

1. **image_info**: Get image metadata (size, format, etc.)
2. **crop_image**: Crop regions from images (top, bottom, left, right, center, or x1,y1,x2,y2)
3. **resize_image**: Resize images by width, height, or scale factor
4. **convert_image**: Convert between formats (png, jpg, webp, bmp, gif)
5. **executor_document_analysis**: Run Marie's document analysis pipeline

When analyzing images:
1. First describe what you see in the image
2. Use tools when you need to manipulate or get details about the image
3. Provide clear, detailed analysis

You have direct vision capabilities - describe images before using tools.""",
        max_iterations=10,
    )


# =============================================================================
# Running Modes
# =============================================================================


def analyze_image(image_path: str, query: str, backend: str = "marie"):
    """Analyze an image with a query."""
    print(f"Image: {image_path}")
    print(f"Query: {query}")
    print("-" * 60)

    agent = create_vision_assistant(backend=backend)

    # Create multimodal message
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_path},
                {"text": query},
            ],
        }
    ]

    for responses in agent.run(messages=messages):
        if responses:
            last = responses[-1]
            content = (
                last.get("content", "") if isinstance(last, dict) else last.content
            )
            if content:
                print(content)


def run_interactive():
    """Run in interactive mode."""
    print("=" * 60)
    print("Vision Assistant - Interactive Mode")
    print("=" * 60)
    print("Commands:")
    print("  image <path>  - Set image to analyze")
    print("  info          - Get current image info")
    print("  clear         - Clear conversation")
    print("  quit          - Exit")
    print()

    agent = create_vision_assistant()
    messages = []
    current_image = None

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break
        if user_input.lower() == "clear":
            messages = []
            print("Conversation cleared.")
            continue
        if user_input.lower().startswith("image "):
            current_image = user_input[6:].strip()
            if Path(current_image).exists():
                print(f"Image set: {current_image}")
            else:
                print(f"Warning: File not found: {current_image}")
            continue
        if user_input.lower() == "info" and current_image:
            result = image_info(current_image)
            print(result)
            continue

        # Build message
        if current_image:
            content = [{"image": current_image}, {"text": user_input}]
        else:
            content = user_input

        messages.append({"role": "user", "content": content})
        print("\nAssistant: ", end="", flush=True)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision Assistant Agent")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--query", type=str, default="Describe this image in detail.")
    parser.add_argument("--tui", action="store_true", help="Interactive mode")
    parser.add_argument("--backend", default="marie", choices=["marie", "openai"])

    args = parser.parse_args()

    if args.tui:
        run_interactive()
    elif args.image:
        analyze_image(args.image, args.query, backend=args.backend)
    else:
        print("Examples:")
        print(
            "  python assistant_vision.py --image photo.jpg --query 'What is in this image?'"
        )
        print("  python assistant_vision.py --tui")
