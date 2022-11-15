---
sidebar_position: 1
---

# Named entity recognition
Extracting Named Entity Recognition / Key Value pair extraction

## Configuration

## Examples

### Executor setup
Basic executor setup and inference.

```python
  from marie.executor import NerExtractionExecutor
  from marie.utils.image_utils import hash_file

  # setup executor
  models_dir = ("/mnt/data/models/")
  executor = NerExtractionExecutor(models_dir)

  img_path = "/tmp/sample.png"
  checksum = hash_file(img_path)

  # invoke executor
  docs = None
  kwa = {"checksum": checksum, "img_path": img_path}
  results = executor.extract(docs, **kwa)

  print(results)
```

### Complete NER example
Setup Named Entity Recognition executor `NerExtractionExecutor` and storage backend `PostgreSQLStorage`

```python
import glob
import os
from typing import Dict

import transformers

from marie.conf.helper import storage_provider_config, load_yaml
from marie.executor import NerExtractionExecutor
from marie.executor.storage.PostgreSQLStorage import PostgreSQLStorage
from marie.logging.profile import TimeContext
from marie.registry.model_registry import ModelRegistry
from marie.utils.image_utils import hash_file, hash_bytes
from marie.utils.json import store_json_object
from marie import (
    Document,
    DocumentArray,
    __model_path__,
    __config_dir__,
)


def process_file(
    executor: NerExtractionExecutor,
    img_path: str,
    storage_enabled: bool,
    storage_conf: Dict[str, str],
):

    with TimeContext(f"### extraction info"):
        filename = img_path.split("/")[-1].replace(".png", "")
        checksum = hash_file(img_path)
        docs = None
        kwa = {"checksum": checksum, "img_path": img_path}
        payload = executor.extract(docs, **kwa)
        print(payload)
        store_json_object(payload, f"/tmp/tensors/json/{filename}.json")

        if storage_enabled:
            storage = PostgreSQLStorage(
                hostname=storage_conf["hostname"],
                port=int(storage_conf["port"]),
                username=storage_conf["username"],
                password=storage_conf["password"],
                database=storage_conf["database"],
                table="check_ner_executor",
            )

            dd2 = DocumentArray([Document(content=payload)])
            storage.add(dd2, {"ref_id": filename, "ref_type": "filename"})

        return payload


def process_dir(
    executor: NerExtractionExecutor,
    image_dir: str,
    storage_enabled: bool,
    conf: Dict[str, str],
):
    for idx, img_path in enumerate(glob.glob(os.path.join(image_dir, "*.*"))):
        try:
            process_file(executor, img_path, storage_enabled, conf)
        except Exception as e:
            print(e)
            # raise e


if __name__ == "__main__":

    # pip install git+https://github.com/huggingface/transformers
    # 4.18.0  -> 4.21.0.dev0 : We should pin it to this version
    print(transformers.__version__)
    _name_or_path = "rms/layoutlmv3-large-corr-ner"
    kwargs = {"__model_path__": __model_path__}
    _name_or_path = ModelRegistry.get_local_path(_name_or_path, **kwargs)

    print(__config_dir__)
    # Load config
    config_data = load_yaml(os.path.join(__config_dir__, "marie-debug.yml"))
    storage_conf = storage_provider_config("postgresql", config_data)
    executor = NerExtractionExecutor(_name_or_path)

    single_file = True
    img_path = f"/home/greg/tmp/image5839050414130576656-0.tif"

    if single_file:
        process_file(executor, img_path, True, storage_conf)
    else:
        process_dir(executor, img_path, True, storage_conf)

```

## Fine-Tuning LayoutLM v3
From annotation to training and inference

### Setup Development Environment
There are two separate environments one using `pip` for Marie-AI and other using `conda` for UniLM. We could mix them
however there are different dependencies needed for UniLM and Marie so it is safer to keep them segregated. Additionally, 
there is no need to use `conda` as this could have been setup with `pip` as well.  

### Load and prepare dataset
Data prep is done from tools from `marie-ai`, to setup development environment follow [getting started guide](/docs/getting-started/installation).
Data is labeled using [Computer Vision Annotation Tool (CVAT)](https://github.com/opencv/cvat) in [COCO Dataset format](https://cocodataset.org/#format-data).

Convert CVAT annotated COCO dataset into [FUNSD](https://guillaumejaume.github.io/FUNSD/) compatible format for finetuning models. 
We do this so we can check our tooling via a baseline FUNSD dataset.

#### Directory structure
There are two required directories `test_deck-raw-01` and `train_deck-raw-01`. Each directory should be in COCO segmentation 
format when exporting from CVAT

```shell
~/dataset/indexer
├── test-deck-raw
│   ├── annotations
│   │ └── instances_default.json
│   └── images
│       ├── 157303757_0.png
│       ├── 157303757_2.png
│       └── 157303757_4.png
├── train-deck-raw
│   ├── annotations
│   │   └── instances_default.json
│   └── images
│       ├── 157303758_0.png
│       ├── 157303758_2.png
│       └── 157303758_4.png
└── validation
```

The script performs few basic steps.

* STEP 1 : Convert COCO to FUNSD like format 
* STEP 2 : Text Box detection, ICR/OCR 
* STEP 3 : Data augmentation
* STEP 4 : Rescale
* STEP 5 : Visualize augmented data

```python
# STEP 1 : Convert COCO to FUNSD like format
convert_coco_to_funsd(src_dir, dst_path)
# STEP 2 : Decorate
decorate_funsd(dst_path)
# STEP 3 : Augment Data
augment_decorated_annotation(count=5, src_dir=dst_path, dest_dir=aug_dest_dir)
# STEP 4 : Rescale
rescale_annotate_frames(src_dir=aug_dest_dir, dest_dir=aug_aligned_dst_path)
# STEP 5 : Visualize augmented data
visualize_funsd(aug_dest_dir)
```

#### Utility usage

```shell
usage: coco_funsd_converter [-h] --mode MODE --dir DIR --dir_converted DIR_CONVERTED --dir_augmented DIR_AUGMENTED

COCO to FUNSD conversion utility

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Configuration file used for conversion
  --mode MODE           Conversion mode : train/test/validate
  --strip_file_name_path STRIP_FILE_NAME_PATH
                        Should full image paths be striped from annotations file
  --dir DIR             Data directory
  --dir_converted DIR_CONVERTED
                        Converted data directory
  --dir_augmented DIR_AUGMENTED
                        Augmented data directory
```

When we have datasets that don't line up with our annotations `file_image` we can use `strip_file_name_path` argument
to strip the image path and use the image file name only.


```shell
source ~/environments/pytorch/bin/activate

PYTHONPATH="$PWD" python ./marie/coco_funsd_converter.py --mode test \
--strip_file_name_path true --dir ~/dataset/private/corr-indexer \
--config ~/dataset/private/corr-indexer/config.json
```

#### Configuration  
Configuration for the tool is defined via `--config` attribute and file is in JSON format.

**Validation** 

There are couple different validations that will be performed.

* Duplicate Mapping (Validation is enabled by default)
* Key / Value aka Question/Answer in FUNSD dataset

#### Duplicate Mapping

Base validation to ensure that we don't have duplicate fields mappings. 

:::warning Duplicate field check

Duplicate pair found for image_id[25] : member_name, 4, sample.png

:::

This validation message tells us that CVAT image_id 25 (zero based) and field `member_name` had 4 duplicate 
values present on given image.

#### Key-Value validation

`question_answer_map` this fields maps KEY => Value, and it is one-to-one mapping, this mapping can be any field that
was defined in CVAT.

```json
{
    "question_answer_map" : {
        "member_name": "member_name_answer",
        "member_number": "member_number_answer",
        "pan": "pan_answer",
        "dos": "dos_answer",
        "patient_name": "patient_name_answer"
    }
}
```
When validation fails for above definition we will receive message:

:::warning Missing mapping

Missing mapping 
 Pair not found for image_id[25] : member_name [1] MISSING -> member_name_answer [8]

:::

This validation message tells us that CVAT image_id 25 (zero based) and field `member_name` is missing corresponding
`member_name_answer` field.


#### Training
Clone UniLM: Unified pre-training for language understanding (NLU) and generation (NLG) project from following repo [https://github.com/gregbugaj/unilm.git](https://github.com/gregbugaj/unilm.git) which is a fork of [https://github.com/microsoft/unilm.git](https://github.com/microsoft/unilm.git)
Fork is kept in sync, but it does contain additional changes.  

```shell
cd ~/dev
git clone https://github.com/gregbugaj/unilm.git
```

Activate your PyTorch environment that will be used to train `layoutlmv3`

```shell
conda env list
conda activate layoutlmv3
cd ~/dev/unilm/
```

## Reference
* [LayoutLMv3: Multi-modal Pre-training for Visually-Rich Document Understanding](https://arxiv.org/abs/2204.08387)
* [FUNSD: Form Understanding in Noisy Scanned Documents](https://guillaumejaume.github.io/FUNSD/)