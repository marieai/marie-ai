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


# Wheel inventory

This directory is the distributed wheel source for the Marie AI checkout. The
PyTorch 2.12 / CUDA 13 / Python 3.12 setup flow must build into and install from
this directory, not from an external artifact directory.

## Current files

| File | SHA256 | Notes |
| --- | --- | --- |
| `etcd3-0.12.0-py2.py3-none-any.whl` | `9b5c36c42a6764d4926c40d131cacd4248f5a3cefc6452fb05a2b3e1e489ed7a` | Distributed etcd3 wheel with modern protobuf dependency support. |
| `etcd3-0.12.0.tar.gz` | `46fd3624665bddbd0957823777d45ed91e2b7f7d698223db984c79bf225b64f3` | Source archive used to rebuild the distributed etcd3 wheel. |
| `fastwer-0.1.3-cp312-cp312-linux_x86_64.whl` | `1d2cbe8bce96cfced4b65c89a9f377a147e69b4886299cde342f4c382ccb46bc` | Distributed Python 3.12 fastwer wheel. |
| `fastwer-0.1.3.tar.gz` | `f411662f337b588ce21aabf51f3170e891bfed3f19f6331b946ad1974401d54d` | Source archive used by `scripts/setup-py312-torch212-cu130.sh build-fastwer`. |

## PyTorch 2.12 / CUDA 13 build outputs

The reproducible setup script writes these native wheels back into this
directory:

| File pattern | Source | Patch |
| --- | --- | --- |
| `fairseq-*.whl` | `github.com/marieai/fairseq` at the script ref, default `main` to match `Dockerfiles/cuda-312.Dockerfile` | `patches/fairseq-marie-torch212-wheel-metadata.patch` |
| `detectron2-*.whl` | `github.com/facebookresearch/detectron2` at the script ref, default `main` to match `Dockerfiles/cuda-312.Dockerfile` | none |
| `faiss_gpu_cu13-*.whl` | `github.com/facebookresearch/faiss` at the pinned script ref; imports as `faiss` | `patches/faiss-cuda13-profiler-api.patch` |

Only those three source-built native wheels belong in this directory as PyTorch
2.12 / CUDA 13 build outputs. The reproducible build baseline is NumPy 2.4.6;
resolver dependency wheels such as `fvcore`, `iopath`, `omegaconf`, `numpy`,
`pillow`, or `matplotlib` are installed through normal dependency resolution and
must not be copied into `wheels/` or a `resolver-spillover/` subdirectory.

Run `scripts/setup-py312-torch212-cu130.sh manifest` after rebuilding to
write the verified wheel list and SHA256s to the command-output manifest.

<!-- local-wheels-inventory:start -->

## Generated file inventory

Updated by scripts/setup-py312-torch212-cu130.sh wheels-readme.

| File | Size bytes | SHA256 |
| --- | ---: | --- |
| detectron2-0.6-cp312-cp312-linux_x86_64.whl | 7450799 | ecacc2035c5257394392274121d6701684cdb825744d0efac1407810e9afb484 |
| etcd3-0.12.0-py2.py3-none-any.whl | 39112 | 9b5c36c42a6764d4926c40d131cacd4248f5a3cefc6452fb05a2b3e1e489ed7a |
| etcd3-0.12.0.tar.gz | 62608 | 46fd3624665bddbd0957823777d45ed91e2b7f7d698223db984c79bf225b64f3 |
| fairseq-0.12.2+marieai.torch212cu130-cp312-cp312-linux_x86_64.whl | 30915780 | 615d9aa51a0f03b75cc6fca35b89587e636c65b3a709e3e774cd02841ab7e458 |
| faiss_gpu_cu13-1.14.1+cu130-py3-none-any.whl | 39931313 | 5676332a3e2d5e17e1e252984aedd353d1afe41b262948c70d15082355e753f9 |
| fastwer-0.1.3-cp312-cp312-linux_x86_64.whl | 1098219 | 6fc055f390e333e76394d1942b55f94a8c2cd591a4854946b9f0a40fcfd387a9 |
| fastwer-0.1.3.tar.gz | 4877 | f411662f337b588ce21aabf51f3170e891bfed3f19f6331b946ad1974401d54d |

<!-- local-wheels-inventory:end -->
