# uv profile locks

These files are generated with `uv 0.11.28` from the dependency groups in the
repo root `pyproject.toml`. They are the authoritative dependency locks for the
Python 3.12 PyTorch 2.12/cu130 Docker lane.

| build.sh profile | dependency group | lock artifact |
| --- | --- | --- |
| `marie-gateway-cpu` | `marie-gateway-cpu` | `requirements/uv/marie-gateway-cpu.lock.txt` |
| `marie-cuda` | `marie-cuda` | `requirements/uv/marie-cuda-torch212-cu130.lock.txt` |

The native repo wheels remain installed from `wheels/` during Docker builds:

- `etcd3-0.12.0-py2.py3-none-any.whl`
- `fastwer-0.1.3-cp312-cp312-linux_x86_64.whl`
- `fairseq-0.12.2+marieai.torch212cu130-cp312-cp312-linux_x86_64.whl`
- `detectron2-0.6-cp312-cp312-linux_x86_64.whl`
- `faiss_gpu_cu13-1.14.1+cu130-py3-none-any.whl`

The CUDA lock includes the fairseq, detectron2, and FAISS wheels directly from
`wheels/`. The CPU lock includes the shared local etcd3 and fastwer wheels.

`uv` defaults to a first-index strategy. These locks intentionally use
`--index-strategy unsafe-best-match` with `--torch-backend` so the CPU and CUDA
torch wheels resolve to the expected backend.

Regenerate from the repo root:

```bash
uv pip compile --group marie-gateway-cpu \
  --python-version 3.12 \
  --torch-backend cpu \
  --index-strategy unsafe-best-match \
  --emit-index-url \
  --emit-find-links \
  --output-file requirements/uv/marie-gateway-cpu.lock.txt

uv pip compile --group marie-cuda \
  --python-version 3.12 \
  --torch-backend cu130 \
  --index-strategy unsafe-best-match \
  --emit-index-url \
  --emit-find-links \
  --output-file requirements/uv/marie-cuda-torch212-cu130.lock.txt
```

Spot check:

```bash
rg -n '^torch==2\.12\.1|^torchvision==0\.27\.1|^numpy==2\.4\.6|wheels/.*(etcd3|fastwer|fairseq|detectron2|faiss)' requirements/uv/*.lock.txt
rg -n 'faiss-gpu-cu12|vllm==|flash-attn|torchaudio|pytesseract' requirements/uv/*.lock.txt
```

The first command must show the expected pins and local wheel paths. The second
command must produce no matches.
