---
sidebar_position: 1
---

# Model Zoo
Model Zoo is a collection of pre-trained, state-of-the-art models.

## Local Configuration 
Marie uses several constant variables as default, and these can be leveraged in your local dev environment as well. For ease of use it is important to set up an environment variable called ```MARIE_DEFAULT_MOUNT``` which marie looks for to set the the python variable ```marie.constants.__default_mount_point__```. This will be the path to your local model_zoo and configs. If not set, your root directory will be concidered the  ```__default_mount_point__```.
```shell
export MARIE_DEFAULT_MOUNT=path/to/parent/directory/of/models_and_configs
```
Once you have this variable you can marie can utilize the marie constants explored in other sections.

```python
from marie.constants import __model_path__, __config_dir__
```