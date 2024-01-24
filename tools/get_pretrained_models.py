import os
import tarfile
import zipfile

import gdown
import requests
from __init__ import ensure_exists, file_exists

__tmp_path__ = "/tmp/marie-ai/model_zoo"
__assets_path__ = os.path.join(__tmp_path__, "assets")


def download_and_extract(url, directory, filename):
    """Check if the file already exists"""

    # Full path for the file
    full_path = os.path.join(directory, filename)
    if file_exists(full_path) or file_exists(full_path.split(".tar.gz")[0]):
        print(f"{filename} already exists in {directory}. Skipping download.")
        return

    # Ensure the directory exists
    ensure_exists(directory)

    # Download from google-drive needs a workaround
    if url.__contains__('drive.google.com'):
        url = "https://drive.google.com/uc?id={}".format(
            url.split('/')[5]
        )  # can replace url.split('/')[5] with id of file
        gdown.download(url, os.path.join(directory, filename))

        # require gdown==4.6.3 otherwise, it will through error
        # https://github.com/wkentaro/gdown/issues/291#issuecomment-1887060708

    else:
        # Download the file from the given URL
        print(f"Downloading {filename} to {directory}")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(full_path, "wb") as f:
                # f.write(response.raw.read())
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
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

    if filename.endswith(".zip"):
        print(f"Extracting {filename}...")
        # Extract the zip file
        with zipfile.ZipFile(os.path.join(directory, filename), 'r') as zip:
            zip.extractall(path=directory)
        print(f"Extraction complete.")

        # Remove .zip file
        os.remove(os.path.join(directory, filename))


def main():
    # Model URLs, their respective directories, and additional assets
    models = {
        "TPS-ResNet-BiLSTM-Attn.pt": {
            "url": "https://www.dropbox.com/sh/j3xmli4di1zuv3s/AAArdcPgz7UFxIHUuKNOeKv_a?dl=1",
            "dir": os.path.join(__tmp_path__, "icr"),
            "assets": [],
            'filename': 'pretrained_model.zip',
        },
        "craft_mlt_25k.pt": {
            "url": "https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ",
            "dir": os.path.join(__tmp_path__, "craft"),
            "assets": [],
            'filename': 'craft_mlt_25k.pt',
        },
        "craft_ic15_20k.pt": {
            "url": "https://drive.google.com/file/d/1i2R7UIUqmkUtF0jv_3MXTqmQ_9wuAnLf",
            "dir": os.path.join(__tmp_path__, "craft"),
            "assets": [],
            'filename': 'craft_ic15_20k.pt',
        },
        "craft_refiner_CTW1500.pt": {
            "url": "https://drive.google.com/file/d/1XSaFwBkOaFOdtk4Ane3DFyJGPRw6v5bO",
            "dir": os.path.join(__tmp_path__, "craft"),
            "assets": [],
            'filename': 'craft_refiner_CTW1500.pt',
        },
        "roberta.large": {
            "url": "https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz",
            "dir": os.path.join(__tmp_path__, "fairseq/models"),
            "assets": ["https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe"],
            'filename': 'roberta.large.tar.gz',
        },
        "roberta.base": {
            "url": "https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz",
            "dir": os.path.join(__tmp_path__, "fairseq/models"),
            "assets": [],
            'filename': 'roberta.base.tar.gz',
        },
        "roberta.large.mnli": {
            "url": "https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz",
            "dir": os.path.join(__tmp_path__, "fairseq/models"),
            "assets": [],
            'filename': 'roberta.large.mnli.tar.gz',
        },
        "trocr-base-printed.pt": {
            "url": "https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-printed.pt?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D",
            "dir": os.path.join(__tmp_path__, "trocr"),
            "assets": [
                "https://layoutlm.blob.core.windows.net/trocr/dictionaries/gpt2_with_mask.dict.txt?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D"
            ],
            'filename': 'trocr-base-printed.pt',
        },
        "trocr-large-printed.pt": {
            "url": "https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-large-printed.pt?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D",
            "dir": os.path.join(__tmp_path__, "trocr"),
            "assets": [
                "https://layoutlm.blob.core.windows.net/trocr/dictionaries/gpt2_with_mask.dict.txt?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D"
            ],
            'filename': 'trocr-large-printed.pt',
        },
        "trocr-large-str.pt": {
            "url": "https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-large-str.pt?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D",
            "dir": os.path.join(__tmp_path__, "trocr"),
            "assets": [
                "https://layoutlm.blob.core.windows.net/trocr/dictionaries/gpt2_with_mask.dict.txt?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D"
            ],
            'filename': 'trocr-large-str.pt',
        },
        "trocr-small-printed.pt": {
            "url": "https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-small-printed.pt?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D",
            "dir": os.path.join(__tmp_path__, "trocr"),
            "assets": [
                "https://layoutlm.blob.core.windows.net/trocr/dictionaries/gpt2_with_mask.dict.txt?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D"
            ],
            'filename': 'trocr-small-printed.pt',
        }
        # Add more models and their assets here as needed
    }

    # Download, extract models and download assets
    for model_name, model_info in models.items():
        model_url = model_info["url"]
        model_dir = model_info["dir"]

        if model_url:
            model_filename = model_url.split("/")[-1]
            if model_filename.endswith(
                "?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D"
            ):
                model_filename = model_filename.replace(
                    "?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D",
                    "",
                )
            download_and_extract(model_url, model_dir, model_info['filename'])

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
