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
models_dir = "/mnt/data/models/"
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
from marie.executor.ner import NerExtractionExecutor
from marie.executor.storage.PostgreSQLStorage import PostgreSQLStorage
from marie.logging.profile import TimeContext
from marie.registry.model_registry import ModelRegistry
from marie.utils.image_utils import hash_file, hash_bytes
from marie.utils.json import store_json_object
from marie.constants import __model_path__, __config_dir__
from marie import (
    Document,
    DocumentArray,
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
    _name_or_path = ModelRegistry.get(_name_or_path, **kwargs)

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

#### Directory structure
Each directory should be in COCO 1.0 format when exporting from CVAT.

Example structure for `test` and `train` modes, by default the data suffix of `-deck-raw` will be added to the mode to 
create a directory name.

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

Activate our marie-ai environment.
```shell
cd ~/dev/marie-ai
source ./venv/bin/activate
```

The script performs few basic steps.

* convert   : Convert COCO to FUNSD like format 
* decorate  : Text Box detection, ICR/OCR 
* augment   : Data augmentation
* rescale   : Rescale/Normalize documents to be used by UNILM
* visualize : Visualize documents
* split     : Split COCO dataset into train/test

Each command can be invoked separately, but initially they need to be invoked in following order if you don't have already 
generated intermediate assets.

```text
convert -> decorate -> augment -> rescale
```

#### Utility usage

```shell
usage: coco_funsd_converter [-h] {convert,decorate,augment,rescale,visualize,split} ...

COCO to FUNSD conversion utility

positional arguments:
  {convert,decorate,augment,rescale,visualize,split,convert-all}
                        Commands to run
    convert             Convert documents from COCO to FUNSD-Like intermediate format
    decorate            Decorate documents(Box detection, ICR)
    augment             Augment documents
    rescale             Rescale/Normalize documents to be used by UNILM
    visualize           Visualize documents
    split               Split COCO dataset into train/test
    convert-all         Run all conversion phases[convert,decorate,augment,rescale] using most defaults.

optional arguments:
  -h, --help            show this help message and exit

```

#### command : convert
Convert documents from COCO to FUNSD-Like intermediate format

```shell
usage: coco_funsd_converter convert [-h] --mode MODE [--mode-suffix MODE_SUFFIX] --strip_file_name_path STRIP_FILE_NAME_PATH --dir DIR [--dir_converted DIR_CONVERTED] [--dir_augmented DIR_AUGMENTED]

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           Conversion mode : train/test/validate/etc
  --mode-suffix MODE_SUFFIX
                        Suffix for the mode
  --strip_file_name_path STRIP_FILE_NAME_PATH
                        Should full image paths be striped from annotations file
  --dir DIR             Base data directory
```

**usage**

```shell
 PYTHONPATH="$PWD" python ./marie/coco_funsd_converter.py convert --mode test \
 --strip_file_name_path true --dir ~/dataset/private/corr-indexer \
 --config ~/dataset/private/corr-indexer/config.json
```

When we have datasets that don't line up with our annotations `file_image` we can use `strip_file_name_path` argument
to strip the image path and use the image file name only.

Default generated folder structure will look as follows (using defaults):

```text
/indexer/output
└── dataset
    └── test
        ├── annotations_tmp    
        └── images
```

#### command : decorate
This step post-processed the data that have been generated via `convert` command and will be created in `/indexer/output/dataset`

**usage**

```shell
PYTHONPATH="$PWD" python ./marie/coco_funsd_converter.py decorate --mode test --dir ~/dataset/private/corr-indexer/output/dataset
```

After the command finished output directory will have a new folder add called `annotations` which contain boxes and ICR.

```text
/indexer/output
└── dataset
    └── test
        ├── annotations        <------------ 
        ├── annotations_tmp
        └── images
```

#### command : augment
Augmentation is not necessary however it provides additional way to introduce variability into datasets that are small.

**usage**

```shell
PYTHONPATH="$PWD" python ./marie/coco_funsd_converter.py augment --mode test \
--dir ~/dataset/private/corr-indexer/output/dataset --count 1
```

After the script is run our directory structure will look as follows:

```text
/indexer/output
└── dataset
    ├── test
    │   ├── annotations
    │   ├── annotations_tmp
    │   └── images
    └── test-augmented              <------------ 
        ├── annotations
        └── images
```

#### command : rescale
Augmentation is not necessary however it provides additional way to introduce variability into datasets that are small.

**usage**

```shell
PYTHONPATH="$PWD" python ./marie/coco_funsd_converter.py rescale --mode test \
--dir ~/dataset/private/corr-indexer/output/dataset
```

[//]: # (41503)

After the script is run our directory structure will look as follows:

```text
/indexer/output
└── dataset
    ├── test
    │   ├── annotations
    │   ├── annotations_tmp
    │   └── images
    ├── test-augmented
    │   ├── annotations
    │   └── images
    └── test-rescaled             <------------ 
        ├── annotations
        └── images
```

#### command : convert-all
Run all conversion phases[convert,decorate,augment,rescale] using most defaults. This is the fastest way to test your
model and make sure that everything is configured correctly.

**usage**

```shell
PYTHONPATH="$PWD" python ./tools/coco_funsd_augmenter.py convert-all --mode test  --strip_file_name_path true --aug-count 2 --dir ~/datasets/private/corr-indexer  --config ~/datasets/private/corr-indexer/config.json ```
```


#### command : visualize
Command for visualizing FUNDS like datasets. 

**usage**

```shell
 PYTHONPATH="$PWD" python ./marie/coco_funsd_converter.py visualize --dir ~/dataset/private/corr-indexer/output/dataset/test-rescaled \
 --config ~/dataset/private/corr-indexer/visualize-config.json
```

Configuration is optional but if provided we will have constant label colors across images.

```json
{
 "label2color": {
  "pan": "blue",
  "pan_answer": "green",
  "dos": "orange"
 }
}
```

#### command : split
Split COCO dataset for training and test.

**usage**

```shell
PYTHONPATH="$PWD" python ./marie/coco_funsd_converter.py split --dir ~/dataset/private/corr-indexer/output/dataset/test-rescaled --ratio .8
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

```json title="Configuration fragment"
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


#### Linking Fields

As we are following FUNSD dataset format we have a definition for linking fields from our COCO dataset.

`id_map` config key maps arbitrary `key` to and `id`.  The `id` could be the same `id` as used in CVAT.  

```json title="Configuration fragment"
  {
   "id_map" : {
        "member_name": 0,
        "member_name_answer": 1,
        "member_number": 2,
        "member_number_answer": 3,
        "pan": 4,
        "pan_answer": 5,
        "dos": 6,
        "dos_answer": 7,
        "patient_name": 8,
        "patient_name_answer": 9,
    }
}
```

`link_map` config key maps arbitrary `key` to `id` and links the two fields together.

This tells us that `member_name` is linked to `[member_name, member_name_answer]` and vice versa as the mapping is
bidirectional.

```json
  {
    "member_name": [0,1],
    "member_name_answer": [0,1]
  }
```

To declare a field that is not linked to anything we give it a value of `-1` 

```json
  {
    "paragraph": [-1]
  }
```

```json title="Configuration fragment"
"link_map" : {
  "member_name": [
    0,
    1
  ],
  "member_name_answer": [
    0,
    1
  ],
  "member_number": [
    2,
    3
  ],
  "member_number_answer": [
    2,
    3
  ],
  "paragraph": [
    -1
  ],
  "greeting": [
    -1
  ]
}

```

### Training 

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

To start trainin we can use the following script 

```shell
screen
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python ./train.py
```


## DIT Bounding Boxes 

Convert 
```shell
 PYTHONPATH="$PWD" python ./tools/coco_funsd_converter.py convert --mode train  --strip_file_name_path true --dir ~/datasets/funsd_bboxes  --config ~/datasets/funsd_bboxes/config.json
```

Rescale

```shell
PYTHONPATH="$PWD" python ./tools/coco_funsd_converter.py rescale --mode train --dir ~/datasets/funsd_bboxes/output/dataset --suffix ''
```




## Reference
* [LayoutLMv3: Multi-modal Pre-training for Visually-Rich Document Understanding](https://arxiv.org/abs/2204.08387)
* [FUNSD: Form Understanding in Noisy Scanned Documents](https://guillaumejaume.github.io/FUNSD/)