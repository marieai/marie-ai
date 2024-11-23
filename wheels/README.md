Custom provider wheels


# Etcd3

This is build using `protobuf>=3.20.0` as a dependency.

```bash
git clone github.com:kragniz/python-etcd3.git
cd python-etcd3

python -m build
```

## Installation

```bash
pip install ./wheels/etcd3-0.12.0-py2.py3-none-any.whl
```


# Fastwer
This is fixings issues with `pybind11` as dependency for modern python versions.


```bash
git clone github.com:marieai/fastwer.git
cd fastwer

python -m build
```

## Installation

```bash
pip install ./wheels/fastwer-0.1.3-cp312-cp312-linux_x86_64.whl
```



# References

https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html