import os
import shutil
import tarfile
import tempfile
import zipfile

import gdown
import requests
from __init__ import ensure_exists, file_exists
from huggingface_hub import snapshot_download

__tmp_path__ = "/tmp/marie-ai/model_zoo"
__assets_path__ = os.path.join(__tmp_path__, "assets")


def download_and_extract(url, directory, filename):
    """Check if the file already exists"""

    # Full path for the file
    full_path = os.path.join(directory, filename)
    if (
        file_exists(full_path)
        or file_exists(full_path.split(".tar.gz")[0])
        or file_exists(full_path.split(".zip")[0])
    ):
        print(f"{filename} already exists in {directory}. Skipping download.")
        return

    # Ensure the directory exists
    ensure_exists(directory)

    # Download from huggingface use huggingface api
    if url.__contains__("huggingface.co"):
        print(f"Downloading {filename} to {directory}")
        repo_id = "/".join(url.split("/")[3:])
        snapshot_download(
            repo_id=repo_id, local_dir=full_path, cache_dir=full_path, revision="main"
        )

        # https://huggingface.co/docs/huggingface_hub/v0.20.3/package_reference/file_download

    # Download from google-drive needs a workaround
    elif url.__contains__("drive.google.com"):
        print(f"Downloading {filename} to {directory}")
        url = "https://drive.google.com/uc?id={}".format(
            url.split("/")[5]
        )  # can replace url.split('/')[5] with id of file
        gdown.download(url, os.path.join(directory, filename))

        # require gdown==4.6.3 otherwise, it will through error
        # https://github.com/wkentaro/gdown/issues/291#issuecomment-1887060708

    else:
        # Download the file from the given URL
        print(f"Downloading {filename} to {directory}")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # prevent incomplete download file to be saved
            with tempfile.NamedTemporaryFile(delete=False) as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            shutil.move(f.name, full_path)
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
        print(f"Extracting {filename}")
        # Extract the zip file
        with zipfile.ZipFile(os.path.join(directory, filename), "r") as zip:
            zip.extractall(path=full_path.split(".zip")[0])
        print(f"Extraction complete.")

        # Remove .zip file
        os.remove(os.path.join(directory, filename))


def main():
    # Model URLs, their respective directories, and additional assets
    models = {
        # LayoutLMv3
        # https://github.com/microsoft/unilm/tree/master/layoutlmv3
        "layoutlmv3-base": {
            "url": "https://huggingface.co/microsoft/layoutlmv3-base",
            "dir": os.path.join(__tmp_path__, "rms"),
            "assets": [],
            "filename": "layoutlmv3-base",
        },
        "layoutlmv3-large": {
            "url": "https://huggingface.co/microsoft/layoutlmv3-large",
            "dir": os.path.join(__tmp_path__, "rms"),
            "assets": [],
            "filename": "layoutlmv3-large",
        },
        # Longformer
        # https://github.com/allenai/longformer/blob/master/README.md
        "longformer-base": {
            "url": "https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-encdec-base-16384.tar.gz",
            "dir": os.path.join(__tmp_path__, "rms"),
            "assets": [],
            "filename": "longformer-encdec-base-16384.tar.gz",
        },
        "longformer-large": {
            "url": "https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-encdec-large-16384.tar.gz",
            "dir": os.path.join(__tmp_path__, "rms"),
            "assets": [],
            "filename": "longformer-encdec-large-16384.tar.gz",
        },
        # DiT
        # https://github.com/microsoft/unilm/tree/master/dit
        "funsd_dit-b_mrcnn": {
            "url": "https://layoutlm.blob.core.windows.net/dit/dit-fts/funsd_dit-b_mrcnn.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D",
            "dir": os.path.join(__tmp_path__, "unilm/dit/text_detection"),
            "assets": [],
            "filename": "funsd_dit-b_mrcnn.pth",
        },
        "funsd_dit-l_mrcnn": {
            "url": "https://layoutlm.blob.core.windows.net/dit/dit-fts/funsd_dit-l_mrcnn.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D",
            "dir": os.path.join(__tmp_path__, "unilm/dit/text_detection"),
            "assets": [],
            "filename": "funsd_dit-l_mrcnn.pth",
        },
        "td-syn_dit-b_mrcnn": {
            "url": "https://layoutlm.blob.core.windows.net/dit/dit-fts/td-syn_dit-b_mrcnn.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D",
            "dir": os.path.join(__tmp_path__, "unilm/dit/text_detection"),
            "assets": [],
            "filename": "td-syn_dit-b_mrcnn.pth",
        },
        "td-syn_dit-l_mrcnn": {
            "url": "https://layoutlm.blob.core.windows.net/dit/dit-fts/td-syn_dit-l_mrcnn.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D",
            "dir": os.path.join(__tmp_path__, "unilm/dit/text_detection"),
            "assets": [],
            "filename": "td-syn_dit-l_mrcnn.pth",
        },
        # LayoutReader
        # https://github.com/microsoft/unilm/tree/master/layoutreader
        "LayoutReader": {
            "url": "https://layoutlm.blob.core.windows.net/readingbank/model/layoutreader-base-readingbank.zip?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D",
            "dir": os.path.join(__tmp_path__, "unilm/layoutreader"),
            "assets": [],
            "filename": "layoutreader-base-readingbank.zip",
        },
        # https://github.com/doc-analysis/ReadingBank
        "ReadingBank": {
            "url": "https://layoutlm.blob.core.windows.net/readingbank/dataset/ReadingBank.zip?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D",
            "dir": os.path.join(__tmp_path__, "unilm/layoutreader"),
            "assets": [],
            "filename": "ReadingBank.zip",
        },
        # https://github.com/clovaai/deep-text-recognition-benchmark/tree/master
        "TPS-ResNet-BiLSTM-Attn.pt": {
            "url": "https://www.dropbox.com/sh/j3xmli4di1zuv3s/AAArdcPgz7UFxIHUuKNOeKv_a?dl=1",
            "dir": os.path.join(__tmp_path__, "icr"),
            "assets": [],
            "filename": "pretrained_model.zip",
        },
        # https://github.com/clovaai/CRAFT-pytorch
        "craft_mlt_25k.pt": {
            "url": "https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ",
            "dir": os.path.join(__tmp_path__, "craft"),
            "assets": [],
            "filename": "craft_mlt_25k.pt",
        },
        "craft_ic15_20k.pt": {
            "url": "https://drive.google.com/file/d/1i2R7UIUqmkUtF0jv_3MXTqmQ_9wuAnLf",
            "dir": os.path.join(__tmp_path__, "craft"),
            "assets": [],
            "filename": "craft_ic15_20k.pt",
        },
        "craft_refiner_CTW1500.pt": {
            "url": "https://drive.google.com/file/d/1XSaFwBkOaFOdtk4Ane3DFyJGPRw6v5bO",
            "dir": os.path.join(__tmp_path__, "craft"),
            "assets": [],
            "filename": "craft_refiner_CTW1500.pt",
        },
        # Roberta
        # https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.md
        "roberta.large": {
            "url": "https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz",
            "dir": os.path.join(__tmp_path__, "fairseq/models"),
            "assets": ["https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe"],
            "filename": "roberta.large.tar.gz",
        },
        "roberta.base": {
            "url": "https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz",
            "dir": os.path.join(__tmp_path__, "fairseq/models"),
            "assets": [],
            "filename": "roberta.base.tar.gz",
        },
        "roberta.large.mnli": {
            "url": "https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz",
            "dir": os.path.join(__tmp_path__, "fairseq/models"),
            "assets": [],
            "filename": "roberta.large.mnli.tar.gz",
        },
        # Trocr
        # https://github.com/microsoft/unilm/tree/master/trocr
        "trocr-base-printed.pt": {
            "url": "https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-printed.pt?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D",
            "dir": os.path.join(__tmp_path__, "trocr"),
            "assets": [
                "https://layoutlm.blob.core.windows.net/trocr/dictionaries/gpt2_with_mask.dict.txt?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D"
            ],
            "filename": "trocr-base-printed.pt",
        },
        "trocr-large-printed.pt": {
            "url": "https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-large-printed.pt?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D",
            "dir": os.path.join(__tmp_path__, "trocr"),
            "assets": [
                "https://layoutlm.blob.core.windows.net/trocr/dictionaries/gpt2_with_mask.dict.txt?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D"
            ],
            "filename": "trocr-large-printed.pt",
        },
        "trocr-large-str.pt": {
            "url": "https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-large-str.pt?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D",
            "dir": os.path.join(__tmp_path__, "trocr"),
            "assets": [
                "https://layoutlm.blob.core.windows.net/trocr/dictionaries/gpt2_with_mask.dict.txt?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D"
            ],
            "filename": "trocr-large-str.pt",
        },
        "trocr-small-printed.pt": {
            "url": "https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-small-printed.pt?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D",
            "dir": os.path.join(__tmp_path__, "trocr"),
            "assets": [
                "https://layoutlm.blob.core.windows.net/trocr/dictionaries/gpt2_with_mask.dict.txt?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D"
            ],
            "filename": "trocr-small-printed.pt",
        },
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
            download_and_extract(model_url, model_dir, model_info["filename"])

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
