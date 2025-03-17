# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "gradio-client",
#     "pillow",
#     "xattr",
#     "jsonschema",
#     "requests",
# ]
# ///

import argparse
import json
import os
import re
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import requests
import xattr
from gradio_client import Client, handle_file
from jsonschema import ValidationError, validate
from PIL import Image


def get_version():
    try:
        script_dir = Path(__file__).resolve().parent
        pyproject_path = script_dir / "pyproject.toml"

        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)

        return pyproject_data["project"]["version"]
    except Exception as e:
        print(f"Warning: Could not read version from pyproject.toml: {e}")
        return "unknown"


# Get the script version from pyproject.toml
SCRIPT_VERSION = get_version()


def get_image_format(file_path):
    format_map = {
        '.jpg': 'JPEG',
        '.jpeg': 'JPEG',
        '.png': 'PNG',
        '.gif': 'GIF',
        '.bmp': 'BMP',
        '.tiff': 'TIFF',
    }
    ext = file_path.suffix.lower()
    return format_map.get(ext, 'JPEG')


def get_rotation_from_xattr(file_path):
    try:
        rotation_str = xattr.getxattr(
            str(file_path), "org.gunzel.spin.rotation#CS"
        ).decode()
        return int(float(rotation_str))  # Convert string to float, then to int
    except (OSError, ValueError):
        return 0


def resize_rotate_and_save_image(image_path, max_size, debug):
    with Image.open(image_path) as img:
        original_size = img.size
        ratio = max_size / max(original_size)
        resized = False
        rotated = False

        if ratio < 1:  # Only resize if the image is larger than max_size
            new_size = tuple(int(dim * ratio) for dim in original_size)
            img = img.resize(new_size, Image.LANCZOS)
            resized = True
            print(f"Resized {image_path} from {original_size} to {new_size}")

        # Check for rotation attribute and rotate if necessary
        rotation_degrees = get_rotation_from_xattr(image_path)
        if rotation_degrees != 0:
            img = img.rotate(
                -rotation_degrees, expand=True
            )  # Negative because PIL rotates counter-clockwise
            rotated = True
            print(f"Rotated {image_path} by {rotation_degrees} degrees clockwise")

        if (resized or rotated) and debug:
            # Create the new filename
            new_filename = image_path.stem
            if resized:
                new_filename += "_resized"
            if rotated:
                new_filename += f"_rotated{rotation_degrees}"
            new_filename += image_path.suffix
            new_path = image_path.parent / new_filename

            # Save the modified image
            img.save(new_path)
            print(f"Saved modified image as {new_path}")

            return new_path
        elif resized or rotated:
            # If debug is False, return a temporary file
            temp_file = Path(f"/tmp/{image_path.stem}_temp{image_path.suffix}")
            img.save(temp_file)
            return temp_file
        else:
            print(f"{image_path} does not need resizing or rotation")
            return image_path


def save_response_to_file(image_path, response, debug):
    if debug:
        response_file = image_path.with_name(f"{image_path.stem}_qwen.txt")
        with open(response_file, 'w', encoding='utf-8') as f:
            f.write(str(response))
        print(f"Saved API response to {response_file}")
        return response_file
    return None


def load_json_schema(schema_path):
    if schema_path.startswith(('http://', 'https://')):
        response = requests.get(schema_path)
        response.raise_for_status()
        return response.json()
    else:
        with open(schema_path, 'r') as f:
            return json.load(f)


def validate_json_structure(json_data, schema=None):
    if schema:
        try:
            validate(instance=json_data, schema=schema)
            return True, "JSON structure is valid according to the provided schema"
        except ValidationError as e:
            return False, f"JSON validation error: {e.message}"
    else:
        # If no schema is provided, just check if it's valid JSON
        try:
            json.dumps(json_data)
            return True, "JSON is valid"
        except (TypeError, ValueError) as e:
            return False, f"Invalid JSON: {str(e)}"


def check_transcribed_attribute(image_path):
    # Check for extended attribute
    try:
        xattr.getxattr(str(image_path), "org.gunzel.clerk.transcribed#S")
        return True
    except OSError:
        pass

    # Check for JSON file
    json_file = image_path.with_suffix('.json')
    if json_file.exists():
        return True

    return False


def save_json_output(image_path, json_data, output_type):
    if output_type == "xattr":
        xattr.setxattr(
            str(image_path),
            "org.gunzel.clerk.transcribed#S",
            json.dumps(json_data).encode(),
        )
        print(f"Saved JSON data as extended attribute on {image_path}")
    elif output_type == "file":
        json_file = image_path.with_suffix('.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        print(f"Saved JSON data to file: {json_file}")


def add_clerk_metadata(json_data, elapsed_time, args):
    selected_args = {
        "max_size": args.max_size,
        "prompt": args.prompt,
        "space": args.space,
        "model": args.model,
        "schema": args.schema,
        "duplicate_space": args.duplicate_space,
        "repeat": args.repeat,
    }

    clerk_metadata = {
        "version": SCRIPT_VERSION,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_time_ms": round(elapsed_time),
        "args": selected_args,
    }

    json_data["_clerk"] = clerk_metadata
    return json_data


def extract_and_validate_json(
    response, image_path, schema, json_output, elapsed_time, args
):
    json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        try:
            json_data = json.loads(json_str)
            is_valid, message = validate_json_structure(json_data, schema)

            if is_valid:
                print("Extracted JSON data is valid.")

                # Print the validated JSON data before adding clerk metadata
                print("Validated JSON data:")
                print(json.dumps(json_data, indent=2))

                # Add clerk metadata
                json_data = add_clerk_metadata(json_data, elapsed_time, args)

                save_json_output(image_path, json_data, json_output)

                return json_data
            else:
                print(f"Error: {message}")
        except json.JSONDecodeError as e:
            print(f"Error: Extracted JSON data is not valid. {str(e)}")
    else:
        print("Error: No JSON code block found in the response.")
    return None


def process_image(
    image_path,
    client,
    max_size,
    prompt,
    model,
    debug,
    schema,
    repeat,
    json_output,
    args,
):
    if check_transcribed_attribute(image_path) and not repeat:
        print(f"Skipping {image_path}: Already processed (use --repeat to override)")
        return

    try:
        image_to_process = resize_rotate_and_save_image(image_path, max_size, debug)

        start_time = time.time()

        result = client.predict(
            image=handle_file(str(image_to_process)),
            text_input=prompt,
            model_id=model,
            api_name="/run_example",
        )

        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000

        print(f"Successfully processed {image_to_process}")
        print(f"API call took {elapsed_time:.3f} milliseconds")

        if debug:
            print("Raw API response:")
            print(result)

        save_response_to_file(image_path, result, debug)

        extract_and_validate_json(
            result, image_path, schema, json_output, elapsed_time, args
        )

        if not debug and image_to_process != image_path:
            os.remove(image_to_process)

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Process images using a specified Hugging Face space and model with optional resizing, rotation, custom prompt, and JSON schema validation.",
        epilog="Environment Variables:\n"
        "  HF_TOKEN: If set, this Hugging Face API token will be used for authentication.\n"
        "            You can set it by running 'export HF_TOKEN=your_token' before running this script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("images", nargs="+", help="Paths to image files to process")
    parser.add_argument(
        "--max-size",
        type=int,
        default=1280,
        help="Maximum size for image dimension (default: 1280)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Extract text",
        help="Prompt for the model (default: 'Extract text')",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--space",
        type=str,
        default="GanymedeNil/Qwen2-VL-7B",
        help="Hugging Face space to use (default: 'GanymedeNil/Qwen2-VL-7B')",
    )
    parser.add_argument(
        "--duplicate-space",
        action="store_true",
        help="Use Client.duplicate() to create the client in your own space",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Model to use for prediction (default: 'Qwen/Qwen2-VL-7B-Instruct')",
    )
    parser.add_argument(
        "--schema", type=str, help="Path or URL to JSON schema for validation"
    )
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="Process images even if they have already been processed",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        choices=["xattr", "file"],
        default="xattr",
        help="Method to output JSON data (default: xattr)",
    )

    args = parser.parse_args()

    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        print("Using Hugging Face API token from environment variable.")
    else:
        print(
            "No Hugging Face API token found in environment. Some features may be limited."
        )

    try:
        if args.duplicate_space:
            print(f"Duplicating space: {args.space}")
            client = Client.duplicate(args.space, hf_token=hf_token)
        else:
            print(f"Using space: {args.space}")
            client = Client(args.space, hf_token=hf_token)
    except Exception as e:
        print(f"Error creating client: {str(e)}")
        print("If you're using --duplicate-space, please note:")
        print(
            "1. The space may still be starting up. Please wait a few minutes and try again."
        )
        print(
            "2. Check the Hugging Face settings page for your space to ensure it's properly configured."
        )
        print("3. Verify that your API token has the necessary permissions.")
        print(
            "If the issue persists, try using the space directly without --duplicate-space."
        )
        sys.exit(1)

    print(f"Using model: {args.model}")

    schema = None
    if args.schema:
        try:
            schema = load_json_schema(args.schema)
            print(f"Loaded JSON schema from: {args.schema}")
        except Exception as e:
            print(f"Error loading JSON schema: {str(e)}")
            sys.exit(1)

    for image_path in args.images:
        path = Path(image_path)
        if path.is_file():
            process_image(
                path,
                client,
                args.max_size,
                args.prompt,
                args.model,
                args.debug,
                schema,
                args.repeat,
                args.json_output,
                args,
            )
        else:
            print(f"File not found: {image_path}")


if __name__ == "__main__":
    main()

# Copyright 2024 Matthew Walker
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
