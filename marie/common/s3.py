import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from platformdirs import user_cache_dir
from tqdm import tqdm

from marie.settings import settings


def join_urls(url1: str, url2: str):
    url1 = url1.rstrip("/")
    url2 = url2.lstrip("/")
    return f"{url1}/{url2}"


def get_model_name(pretrained_model_name_or_path: str):
    return pretrained_model_name_or_path.split("/")[0]


def download_file(remote_path: str, local_path: str, chunk_size: int = 1024 * 1024):
    local_path = Path(local_path)
    try:
        response = requests.get(remote_path, stream=True, allow_redirects=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

        return local_path
    except Exception as e:
        if local_path.exists():
            local_path.unlink()
        print(f"Download error for file {remote_path}: {str(e)}")
        raise


def check_manifest(local_dir: str):
    local_dir = Path(local_dir)
    manifest_path = local_dir / "manifest.json"
    if not os.path.exists(manifest_path):
        return False

    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        for file in manifest["files"]:
            if not os.path.exists(local_dir / file):
                return False
    except Exception as e:
        return False

    return True


def download_directory(remote_path: str, local_dir: str):
    model_name = get_model_name(remote_path)
    s3_url = join_urls(settings.S3_BASE_URL, remote_path)
    # Check to see if it's already downloaded
    model_exists = check_manifest(local_dir)
    if model_exists:
        return

    # Use tempfile.TemporaryDirectory to automatically clean up
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download the manifest file
        manifest_file = join_urls(s3_url, "manifest.json")
        manifest_path = os.path.join(temp_dir, "manifest.json")
        download_file(manifest_file, manifest_path)

        # List and download all files
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        pbar = tqdm(
            desc=f"Downloading {model_name} model...", total=len(manifest["files"])
        )

        with ThreadPoolExecutor(
            max_workers=settings.PARALLEL_DOWNLOAD_WORKERS
        ) as executor:
            futures = []
            for file in manifest["files"]:
                remote_file = join_urls(s3_url, file)
                local_file = os.path.join(temp_dir, file)
                futures.append(executor.submit(download_file, remote_file, local_file))

            for future in futures:
                future.result()
                pbar.update(1)

        pbar.close()

        # Move all files to new directory
        for file in os.listdir(temp_dir):
            shutil.move(os.path.join(temp_dir, file), local_dir)


class S3DownloaderMixin:

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # Allow loading models directly from the hub, or using s3
        if not pretrained_model_name_or_path.startswith("s3://"):
            return super().from_pretrained(
                pretrained_model_name_or_path, *args, **kwargs
            )
        if True:
            raise NotImplementedError("S3 download is not implemented yet")
        pretrained_model_name_or_path = pretrained_model_name_or_path.replace(
            "s3://", ""
        )
        cache_dir = Path(user_cache_dir('marie')) / "models"
        local_path = os.path.join(cache_dir, pretrained_model_name_or_path)
        os.makedirs(local_path, exist_ok=True)

        # Retry logic for downloading the model folder
        retries = 3
        delay = 5
        attempt = 0
        success = False
        while not success and attempt < retries:
            try:
                download_directory(pretrained_model_name_or_path, local_path)
                success = True  # If download succeeded
            except Exception as e:
                print(
                    f"Error downloading model from {pretrained_model_name_or_path}. Attempt {attempt+1} of {retries}. Error: {e}"
                )
                attempt += 1
                if attempt < retries:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)  # Wait before retrying
                else:
                    print(
                        f"Failed to download {pretrained_model_name_or_path} after {retries} attempts."
                    )
                    raise e  # Reraise exception after max retries

        return super().from_pretrained(local_path, *args, **kwargs)
