---
description: Model Registry
sidebar_label: Registry
slug: /models/model-registry
---

# Model Registry
Model registry provides a unified way to search and retrieve models from number of sources.

Each model is downloaded to local cache and used from there, check is performed on the model to make sure that it has
not been updated since last download time.

## Providers
There are number of different providers that can be used out of the box.


| Protocol     | Description                         |
|--------------|-------------------------------------|
| zoo          | Local model zoo(native file access) |
| git          | Git                                 |
| S3           | Amazon S3                           |
| gdrive       | Google Drive                        |
| dvc          | Data Version Control                |
| mflow        | Mflow support                       |
| transformers | Transformers model                  |

### Adding provider 

Adding a new provider requires that we implement a `ModelRegistryHandler`.    
`ModelRegistryHandler` is a base class that defines common functionality for a URI protocol.
It routes I/O for a generic URI which may look like `protocol://*` or a canonical filepath `/foo/bar/baz.`


## Examples 

**Get model by name or path**
This is a basic usage of how to use the `ModelRegistry` all registered providers will be searched in order to find the
model.

```python
_name_or_path = "group/layoutlmv3-large-indexer-ner"
_name_or_path = ModelRegistry.get(_name_or_path)
```

**Customize model search directory**

We can customize model search directory by adding `__model_path__` to our `**kwargs` and passing them to the function.

```python
__model_path__ = os.path.join(
    os.path.abspath(os.path.join(__root_dir__, "..")), "model_zoo"
)
_name_or_path = "group/layoutlmv3-large-indexer-ner"
kwargs = {"__model_path__": __model_path__}
_name_or_path = ModelRegistry.get(_name_or_path, **kwargs)
```


# Proxies
Proxies are used to provide a unified way to access models from different sources.


