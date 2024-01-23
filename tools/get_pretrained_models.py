import os
import tarfile

import requests
from __init__ import ensure_exists, file_exists

__tmp_path__ = "/tmp/marie-ai/model_zoo"
__assets_path__ = os.path.join(__tmp_path__, "assets")


def download_and_extract(url, directory, filename):
    """Check if the file already exists"""

    if file_exists(directory, filename) or file_exists(
        directory, filename.split(".tar.gz")[0]
    ):
        print(f"{filename} already exists in {directory}. Skipping download.")
        return

    # Ensure the directory exists
    ensure_exists(directory)

    # Full path for the file
    full_path = os.path.join(directory, filename)

    # Download the file from the given URL
    print(f"Downloading {filename} to {directory}")
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(full_path, "wb") as f:
            f.write(response.raw.read())
        print(f"Download complete.")
    else:
        print(f"Failed to download {filename}. Status code: {response.status_code}")

    if filename.endswith(".tar.gz"):
        print(f"Extracting {filename}...")
        # Extract the tar.gz file
        with tarfile.open(os.path.join(directory, filename), "r:gz") as tar:
            tar.extractall(path=directory)
        print(f"Extraction complete.")

        # Remove .tar.gz file
        os.remove(os.path.join(directory, filename))


def main():
    # Model URLs, their respective directories, and additional assets
    models = {
        "roberta.large": {
            "url": "https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz",
            "dir": os.path.join(__tmp_path__, "fairseq/models"),
            "assets": ["https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe"],
        },
        "roberta.base": {
            "url": "https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz",
            "dir": os.path.join(__tmp_path__, "fairseq/models"),
            "assets": [],
        },
        "roberta.large.mnli": {
            "url": "https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz",
            "dir": os.path.join(__tmp_path__, "fairseq/models"),
            "assets": [],
        },
        "trocr-large-printed": {
            "url": "https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-large-printed.pt?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D",
            "dir": os.path.join(__tmp_path__, "trocr"),
            "assets": [
                "https://layoutlm.blob.core.windows.net/trocr/dictionaries/gpt2_with_mask.dict.txt?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D"
            ],
        },
        "trocr-large-str": {
            "url": "https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-large-str.pt?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D",
            "dir": os.path.join(__tmp_path__, "trocr"),
            "assets": [
                "https://layoutlm.blob.core.windows.net/trocr/dictionaries/gpt2_with_mask.dict.txt?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D"
            ],
        },
        "trocr-small-printed": {
            "url": "https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-small-printed.pt?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D",
            "dir": os.path.join(__tmp_path__, "trocr"),
            "assets": [
                "https://layoutlm.blob.core.windows.net/trocr/dictionaries/gpt2_with_mask.dict.txt?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D"
            ],
        },
        "trocr-base-printed": {
            "url": "https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-printed.pt?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D",
            "dir": os.path.join(__tmp_path__, "trocr"),
            "assets": [
                "https://layoutlm.blob.core.windows.net/trocr/dictionaries/gpt2_with_mask.dict.txt?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D"
            ],
        }
        # Add more models and their assets here as needed
    }

    # Download, extract models and download assets
    for model_name, model_info in models.items():
        model_url = model_info["url"]
        model_dir = model_info["dir"]

        if model_url:  # If there's a model to download and extract
            model_filename = model_url.split("/")[-1]
            if model_filename.endswith(
                "?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D"
            ):
                model_filename = model_filename.replace(
                    "?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D",
                    "",
                )
            download_and_extract(model_url, model_dir, model_filename)

        # Download additional assets
        for asset_url in model_info["assets"]:
            asset_filename = asset_url.split("/")[-1]
            if asset_filename.endswith(
                "?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D"
            ):
                asset_filename = asset_filename.replace(
                    "?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D",
                    "",
                )
            download_and_extract(asset_url, __assets_path__, asset_filename)


if __name__ == "__main__":
    main()
