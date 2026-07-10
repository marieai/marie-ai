# uv profile locks

These files are generated with `uv 0.11.28` for the existing Python 3.12 `build.sh` profiles. They are parity artifacts for the current `setup.py` plus `extra-requirements.txt-*` flow, not the final uv workspace lock.

| build.sh profile | installer | lock artifact |
| --- | --- | --- |
| `marie-gateway-cpu` | `uv` | `requirements/uv/marie-gateway-cpu.lock.txt` |
| `marie-cuda` | `uv` | `requirements/uv/marie-cuda-torch212-cu130.lock.txt` |

The native repo wheels remain installed from `wheels/` during Docker builds:

- `etcd3-0.12.0-py2.py3-none-any.whl`
- `fastwer-0.1.3-cp312-cp312-linux_x86_64.whl`
- `fairseq-0.12.2+marieai.torch212cu130-cp312-cp312-linux_x86_64.whl`
- `detectron2-0.6-cp312-cp312-linux_x86_64.whl`
- `faiss_gpu_cu13-1.14.1+cu130-py3-none-any.whl`

`uv` defaults to a first-index strategy. These locks intentionally use `--index-strategy unsafe-best-match` because the torch constraint files include PyPI plus the PyTorch wheel index; both are expected inputs for these build profiles.

Regenerate from a staged context so each profile sees the same `extra-requirements.txt` variant that `build.sh` stages for Docker:

```bash
tmp_root="$(mktemp -d)"
root="$PWD"

mkdir -p "$tmp_root/marie-gateway-cpu" "$tmp_root/marie-cuda" requirements/uv

cp setup.py README.md requirements.txt "$tmp_root/marie-gateway-cpu/"
cp extra-requirements.txt-CPU "$tmp_root/marie-gateway-cpu/extra-requirements.txt"
cp -a requirements "$tmp_root/marie-gateway-cpu/requirements"

cp setup.py README.md requirements.txt "$tmp_root/marie-cuda/"
cp extra-requirements.txt-CUDA "$tmp_root/marie-cuda/extra-requirements.txt"
cp -a requirements "$tmp_root/marie-cuda/requirements"

(
  cd "$tmp_root/marie-gateway-cpu"
  uv pip compile setup.py requirements/torch-2.12-cpu.txt \
    --python-version 3.12 \
    --torch-backend cpu \
    --index-strategy unsafe-best-match \
    --emit-index-url \
    --output-file "$root/requirements/uv/marie-gateway-cpu.lock.txt"
)

(
  cd "$tmp_root/marie-cuda"
  uv pip compile setup.py requirements/torch-2.12-cu130.txt \
    --python-version 3.12 \
    --torch-backend cu130 \
    --index-strategy unsafe-best-match \
    --emit-index-url \
    --output-file "$root/requirements/uv/marie-cuda-torch212-cu130.lock.txt"
)

rm -rf "$tmp_root"
```

Spot check:

```bash
rg -n '^torch==2\.12\.1|^torchvision==0\.27\.1' requirements/uv/*.lock.txt
rg -n 'faiss-gpu-cu12|vllm==|flash-attn|torchaudio' requirements/uv/*.lock.txt
```
