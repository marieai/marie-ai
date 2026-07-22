#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
MARIE_AI_ROOT="${MARIE_AI_ROOT:-$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)}"
MARIE_AI_PARENT="$(dirname "${MARIE_AI_ROOT}")"

MARIE_TORCH_WORKTREE="${MARIE_TORCH_WORKTREE:-${MARIE_AI_ROOT}}"
MARIE_VENV_BASE="${MARIE_VENV_BASE:-${MARIE_AI_PARENT}/.venvs}"
MARIE_TORCH_ENV_NAME="${MARIE_TORCH_ENV_NAME:-}"
MARIE_TORCH_VENV="${MARIE_TORCH_VENV:-}"
MARIE_WHEELS_DIR="${MARIE_WHEELS_DIR:-${MARIE_TORCH_WORKTREE}/wheels}"
MARIE_MODEL_MOUNT="${MARIE_MODEL_MOUNT:-/mnt/data/marie-ai}"

MARIE_REPRO_RUN_ID="${MARIE_REPRO_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
MARIE_REPRO_LOG_DIR="${MARIE_REPRO_LOG_DIR:-${MARIE_TORCH_WORKTREE}/command-output/repro-${MARIE_REPRO_RUN_ID}}"
MARIE_SOURCE_ROOT="${MARIE_SOURCE_ROOT:-${MARIE_AI_PARENT}/sources/torch-2.12-cu130-${MARIE_REPRO_RUN_ID}}"

MARIE_FAIRSEQ_REF="${MARIE_FAIRSEQ_REF:-main}"
MARIE_DETECTRON2_REF="${MARIE_DETECTRON2_REF:-main}"
MARIE_FAISS_REF="${MARIE_FAISS_REF:-5622e93733b64b2e033362dbdfda019b2ab33ef0}"
MARIE_FAIRSEQ_REPO="${MARIE_FAIRSEQ_REPO:-https://github.com/marieai/fairseq.git}"
MARIE_DETECTRON2_REPO="${MARIE_DETECTRON2_REPO:-https://github.com/facebookresearch/detectron2.git}"
MARIE_FAISS_REPO="${MARIE_FAISS_REPO:-https://github.com/facebookresearch/faiss.git}"
MARIE_NUMPY_VERSION="${MARIE_NUMPY_VERSION:-2.4.6}"
MARIE_TORCH_CUDA_ARCH_LIST="${MARIE_TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0}"
MARIE_FAISS_CUDA_ARCH="${MARIE_FAISS_CUDA_ARCH:-89}"
MARIE_BUILD_JOBS="${MARIE_BUILD_JOBS:-$(nproc)}"

FAIRSEQ_PATCH="${MARIE_TORCH_WORKTREE}/patches/fairseq-marie-torch212-wheel-metadata.patch"
FAISS_PATCH="${MARIE_TORCH_WORKTREE}/patches/faiss-cuda13-profiler-api.patch"

STEP_NO=0

usage() {
  cat <<'EOF'
Usage:
  scripts/setup-py312-torch212-cu130.sh <step>

Run this from, or inside, the Marie AI checkout. All repo-local paths resolve
from the Marie AI checkout, not from marie-assistant/analysis.

Steps:
  env                 Print resolved paths and versions.
  worktree            Verify the current checkout/worktree.
  venv                Create the Python 3.12 venv and base build tools, and
                      symlink .venv in the checkout to it.
  torch               Install torch 2.12.1 / torchvision 0.27.1 from cu130.
  app-deps            Install Marie editable dependencies and test/runtime helpers.
  cuda-toolkit         Install/activate CUDA 13 nvcc package and symlinks.
  build-fastwer        Build the local fastwer cp312 wheel from wheels/fastwer tarball.
  build-fairseq        Build Marie fairseq fork wheel from the configured ref.
  build-detectron2     Build detectron2 wheel from the configured ref.
  reject-faiss-cu12    Record why the current faiss-gpu-cu12 package path is rejected.
  build-faiss          Build CUDA-enabled FAISS from source with the CUDA 13 patch.
  build-wheels         Build every wheel in order (cuda-toolkit, fastwer, fairseq,
                       detectron2, faiss), then check-wheels and wheels-readme.
                       Requires an existing venv with torch installed.
  check-wheels         Verify wheels/ contains only expected top-level artifacts.
  install-wheels       Install fastwer/etcd3 plus repo-local wheels into the venv.
  verify-native        Verify torch, fairseq, detectron2 ROIAlign CUDA, and FAISS GPU.
  verify-tests         Run the focused engine unit test baseline.
  verify-gradio        Run bounding-boxes and OCR Gradio legacy gates.
  manifest             Write wheel SHA256s, uv freeze, and torch collect_env.
  wheels-readme        Update wheels/README.md with the current file inventory.
  all                  Run every setup/build/verification step in order.

Useful environment overrides:
  MARIE_AI_ROOT=/path/to/marie-ai-checkout
  MARIE_TORCH_WORKTREE=/path/to/marie-ai-checkout
  MARIE_TORCH_ENV_NAME=marie-ai-pytorch-2-12
  MARIE_VENV_BASE=/path/to/venvs
  MARIE_TORCH_VENV=/path/to/venv
  MARIE_WHEELS_DIR=/path/to/marie-ai-checkout/wheels
  MARIE_REPRO_LOG_DIR=/path/to/logs
  MARIE_MODEL_MOUNT=/mnt/data/marie-ai
  MARIE_NUMPY_VERSION=2.4.6
  MARIE_FAIRSEQ_REPO=https://github.com/marieai/fairseq.git
  MARIE_DETECTRON2_REPO=https://github.com/facebookresearch/detectron2.git
  MARIE_FAISS_REPO=https://github.com/facebookresearch/faiss.git
  MARIE_BUILD_JOBS=12

Every step writes full command output to MARIE_REPRO_LOG_DIR.
EOF
}

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

resolve_environment() {
  if [[ -z "${MARIE_TORCH_VENV}" ]]; then
    if [[ -z "${MARIE_TORCH_ENV_NAME}" ]]; then
      local default_name
      default_name="$(basename "${MARIE_TORCH_WORKTREE}")"

      if [[ ! -t 0 ]]; then
        printf 'MARIE_TORCH_ENV_NAME is required when stdin is not interactive.\n' >&2
        printf 'Example: MARIE_TORCH_ENV_NAME=%s %s all\n' "${default_name}" "$0" >&2
        exit 2
      fi

      read -r -p "Python environment name [${default_name}]: " MARIE_TORCH_ENV_NAME
      MARIE_TORCH_ENV_NAME="${MARIE_TORCH_ENV_NAME:-${default_name}}"
    fi

    if [[ -z "${MARIE_TORCH_ENV_NAME//[[:space:]]/}" ]]; then
      printf 'Environment name cannot be empty.\n' >&2
      exit 2
    fi

    if [[ "${MARIE_TORCH_ENV_NAME}" == */* ]]; then
      printf 'Environment name must be a name, not a path: %s\n' "${MARIE_TORCH_ENV_NAME}" >&2
      exit 2
    fi

    MARIE_TORCH_VENV="${MARIE_VENV_BASE}/${MARIE_TORCH_ENV_NAME}"
  elif [[ -z "${MARIE_TORCH_ENV_NAME}" ]]; then
    MARIE_TORCH_ENV_NAME="$(basename "${MARIE_TORCH_VENV}")"
  fi

  export MARIE_TORCH_ENV_NAME MARIE_TORCH_VENV
}

guard_new_environment() {
  local step="$1"

  if [[ -e "${MARIE_TORCH_VENV}" ]]; then
    printf 'Environment already present: %s\n' "${MARIE_TORCH_VENV}" >&2
    printf 'Choose another environment name or remove the existing directory before running `%s`.\n' "${step}" >&2
    exit 1
  fi
}

require_build_venv() {
  if [[ ! -x "${MARIE_TORCH_VENV}/bin/python" ]]; then
    printf 'No venv at %s. Run the `venv`, `torch`, and `app-deps` steps first.\n' "${MARIE_TORCH_VENV}" >&2
    exit 1
  fi
  if ! "${MARIE_TORCH_VENV}/bin/python" -c 'import torch' >/dev/null 2>&1; then
    printf 'torch is not installed in %s. Run the `torch` step first.\n' "${MARIE_TORCH_VENV}" >&2
    exit 1
  fi
}

write_invocation() {
  local invocation_path="${MARIE_REPRO_LOG_DIR}/invocation.sh"
  local step_name="${1:-command}"
  local step_invocation_path="${MARIE_REPRO_LOG_DIR}/invocation-${step_name}.sh"
  local arg

  mkdir -p "${MARIE_REPRO_LOG_DIR}"
  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -Eeuo pipefail\n\n'
    printf 'cd %q\n' "${MARIE_TORCH_WORKTREE}"
    printf 'export MARIE_AI_ROOT=%q\n' "${MARIE_AI_ROOT}"
    printf 'export MARIE_TORCH_WORKTREE=%q\n' "${MARIE_TORCH_WORKTREE}"
    printf 'export MARIE_VENV_BASE=%q\n' "${MARIE_VENV_BASE}"
    printf 'export MARIE_TORCH_ENV_NAME=%q\n' "${MARIE_TORCH_ENV_NAME}"
    printf 'export MARIE_TORCH_VENV=%q\n' "${MARIE_TORCH_VENV}"
    printf 'export MARIE_WHEELS_DIR=%q\n' "${MARIE_WHEELS_DIR}"
    printf 'export MARIE_MODEL_MOUNT=%q\n' "${MARIE_MODEL_MOUNT}"
    printf 'export MARIE_REPRO_RUN_ID=%q\n' "${MARIE_REPRO_RUN_ID}"
    printf 'export MARIE_REPRO_LOG_DIR=%q\n' "${MARIE_REPRO_LOG_DIR}"
    printf 'export MARIE_SOURCE_ROOT=%q\n' "${MARIE_SOURCE_ROOT}"
    printf 'export MARIE_FAIRSEQ_REF=%q\n' "${MARIE_FAIRSEQ_REF}"
    printf 'export MARIE_DETECTRON2_REF=%q\n' "${MARIE_DETECTRON2_REF}"
    printf 'export MARIE_FAISS_REF=%q\n' "${MARIE_FAISS_REF}"
    printf 'export MARIE_FAIRSEQ_REPO=%q\n' "${MARIE_FAIRSEQ_REPO}"
    printf 'export MARIE_DETECTRON2_REPO=%q\n' "${MARIE_DETECTRON2_REPO}"
    printf 'export MARIE_FAISS_REPO=%q\n' "${MARIE_FAISS_REPO}"
    printf 'export MARIE_NUMPY_VERSION=%q\n' "${MARIE_NUMPY_VERSION}"
    printf 'export MARIE_TORCH_CUDA_ARCH_LIST=%q\n' "${MARIE_TORCH_CUDA_ARCH_LIST}"
    printf 'export MARIE_FAISS_CUDA_ARCH=%q\n' "${MARIE_FAISS_CUDA_ARCH}"
    printf 'export MARIE_BUILD_JOBS=%q\n' "${MARIE_BUILD_JOBS}"
    printf '\n'
    printf '%q' "${MARIE_TORCH_WORKTREE}/scripts/setup-py312-torch212-cu130.sh"
    for arg in "$@"; do
      printf ' %q' "${arg}"
    done
    printf '\n'
  } > "${invocation_path}"
  cp "${invocation_path}" "${step_invocation_path}"
  chmod +x "${invocation_path}"
  chmod +x "${step_invocation_path}"
  log "wrote invocation: ${invocation_path}"
  log "wrote step invocation: ${step_invocation_path}"
}

next_log() {
  local label="$1"
  STEP_NO=$((STEP_NO + 1))
  printf '%s/%02d-%s.log' "${MARIE_REPRO_LOG_DIR}" "${STEP_NO}" "${label}"
}

run_cmd() {
  local label="$1"
  local command="$2"
  local log_file
  log_file="$(next_log "${label}")"
  mkdir -p "${MARIE_REPRO_LOG_DIR}"

  {
    printf '### STEP: %s\n' "${label}"
    printf '### START: %s\n' "$(date -Is)"
    printf '### MARIE_TORCH_WORKTREE: %s\n' "${MARIE_TORCH_WORKTREE}"
    printf '### COMMAND:\n%s\n\n' "${command}"
  } | tee -a "${log_file}"

  set +e
  bash -lc "set -Eeo pipefail
${command}" 2>&1 | tee -a "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e

  {
    printf '\n### EXIT: %s\n' "${status}"
    printf '### END: %s\n' "$(date -Is)"
  } | tee -a "${log_file}"

  if [[ "${status}" -ne 0 ]]; then
    log "failed step ${label}; see ${log_file}"
    return "${status}"
  fi
}

cuda_home_for_venv() {
  local venv_cuda="${MARIE_TORCH_VENV}/lib/python3.12/site-packages/nvidia/cu13"
  if [[ -x "${venv_cuda}/bin/nvcc" ]]; then
    printf '%s\n' "${venv_cuda}"
    return 0
  fi

  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    printf '%s\n' "${CUDA_HOME}"
    return 0
  fi

  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    printf '%s\n' /usr/local/cuda
    return 0
  fi

  return 1
}

venv_prefix() {
  local prefix
  prefix="source \"${MARIE_TORCH_VENV}/bin/activate\"; export VIRTUAL_ENV=\"${MARIE_TORCH_VENV}\"; export VENV_PYTHON=\"${MARIE_TORCH_VENV}/bin/python\"; "

  local cuda_home=""
  if cuda_home="$(cuda_home_for_venv 2>/dev/null)"; then
    prefix+="export CUDA_HOME=\"${cuda_home}\"; "
    prefix+="export PATH=\"${cuda_home}/bin:\${PATH}\"; "
    prefix+="export LD_LIBRARY_PATH=\"${cuda_home}/lib:${cuda_home}/lib64:\${LD_LIBRARY_PATH:-}\"; "
  fi

  printf '%s' "${prefix}"
}

run_venv() {
  local label="$1"
  local command="$2"
  run_cmd "${label}" "$(venv_prefix) ${command}"
}

apply_patch_once_cmd() {
  local patch_path="$1"
  cat <<EOF
if git apply --check "${patch_path}"; then
  git apply "${patch_path}"
elif git apply --reverse --check "${patch_path}"; then
  echo "patch already applied: ${patch_path}"
else
  echo "patch cannot be applied cleanly: ${patch_path}" >&2
  exit 1
fi
EOF
}

step_env() {
  run_cmd env "cat <<EOF
MARIE_AI_ROOT=${MARIE_AI_ROOT}
MARIE_TORCH_WORKTREE=${MARIE_TORCH_WORKTREE}
MARIE_VENV_BASE=${MARIE_VENV_BASE}
MARIE_TORCH_ENV_NAME=${MARIE_TORCH_ENV_NAME}
MARIE_TORCH_VENV=${MARIE_TORCH_VENV}
MARIE_WHEELS_DIR=${MARIE_WHEELS_DIR}
MARIE_REPRO_LOG_DIR=${MARIE_REPRO_LOG_DIR}
MARIE_SOURCE_ROOT=${MARIE_SOURCE_ROOT}
MARIE_MODEL_MOUNT=${MARIE_MODEL_MOUNT}
MARIE_FAIRSEQ_REF=${MARIE_FAIRSEQ_REF}
MARIE_DETECTRON2_REF=${MARIE_DETECTRON2_REF}
MARIE_FAISS_REF=${MARIE_FAISS_REF}
MARIE_FAIRSEQ_REPO=${MARIE_FAIRSEQ_REPO}
MARIE_DETECTRON2_REPO=${MARIE_DETECTRON2_REPO}
MARIE_FAISS_REPO=${MARIE_FAISS_REPO}
MARIE_NUMPY_VERSION=${MARIE_NUMPY_VERSION}
MARIE_TORCH_CUDA_ARCH_LIST=${MARIE_TORCH_CUDA_ARCH_LIST}
MARIE_FAISS_CUDA_ARCH=${MARIE_FAISS_CUDA_ARCH}
EOF
python3.12 --version
uv --version
git -C \"${MARIE_TORCH_WORKTREE}\" rev-parse --show-toplevel
git -C \"${MARIE_TORCH_WORKTREE}\" status --short --branch
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv || true"
}

step_worktree() {
  run_cmd worktree "test -d \"${MARIE_TORCH_WORKTREE}\"
git -C \"${MARIE_TORCH_WORKTREE}\" rev-parse --is-inside-work-tree
git -C \"${MARIE_TORCH_WORKTREE}\" status --short --branch
git -C \"${MARIE_TORCH_WORKTREE}\" rev-parse HEAD
test -d \"${MARIE_WHEELS_DIR}\"
test -f \"${MARIE_WHEELS_DIR}/fastwer-0.1.3.tar.gz\"
test -f \"${MARIE_WHEELS_DIR}/etcd3-0.12.0-py2.py3-none-any.whl\"
test -f \"${FAIRSEQ_PATCH}\"
test -f \"${FAISS_PATCH}\""
}

step_venv() {
  run_cmd venv "mkdir -p \"$(dirname "${MARIE_TORCH_VENV}")\"
test ! -e \"${MARIE_TORCH_VENV}\"
command -v uv
uv venv --python 3.12 \"${MARIE_TORCH_VENV}\"
source \"${MARIE_TORCH_VENV}/bin/activate\"
export VENV_PYTHON=\"${MARIE_TORCH_VENV}/bin/python\"
\"\${VENV_PYTHON}\" --version
uv pip install --python \"\${VENV_PYTHON}\" -U 'setuptools<81' wheel
uv pip install --python \"\${VENV_PYTHON}\" -U packaging ninja cmake pybind11 swig \"numpy==${MARIE_NUMPY_VERSION}\"
cd \"${MARIE_TORCH_WORKTREE}\"
if [[ -L .venv || ! -e .venv ]]; then
  ln -sfn \"${MARIE_TORCH_VENV}\" .venv
  echo \"linked ${MARIE_TORCH_WORKTREE}/.venv -> ${MARIE_TORCH_VENV}\"
else
  echo \"skipping .venv symlink: ${MARIE_TORCH_WORKTREE}/.venv is a real directory\" >&2
fi"
}

step_torch() {
  run_venv torch "uv pip install --python \"\${VENV_PYTHON}\" --torch-backend cu130 torch==2.12.1 torchvision==0.27.1
python - <<'PY'
import torch
import torchvision
from torchvision.ops import nms

assert torch.__version__.startswith('2.12.1'), torch.__version__
assert torch.version.cuda == '13.0', torch.version.cuda
print('torch', torch.__version__)
print('torchvision', torchvision.__version__)
print('torch cuda', torch.version.cuda)
print('cuda available', torch.cuda.is_available())
boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=torch.float32)
print('nms', nms(boxes, torch.tensor([0.9, 0.8]), 0.5))
if torch.cuda.is_available():
    x = torch.ones((64, 64), device='cuda')
    y = x @ x
    torch.cuda.synchronize()
    print('gpu', torch.cuda.get_device_name(0), float(y[0, 0].cpu()))
PY"
}

step_app_deps() {
  run_venv app-deps "cd \"${MARIE_TORCH_WORKTREE}\"
uv pip install --python \"\${VENV_PYTHON}\" -e .
uv pip install --python \"\${VENV_PYTHON}\" pytest openai 'gradio==6.20.0' 'timm==1.0.27' nltk
uv pip uninstall --python \"\${VENV_PYTHON}\" Pillow-SIMD || true
uv pip install --python \"\${VENV_PYTHON}\" --reinstall 'Pillow>=11.0,<13'
uv pip install --python \"\${VENV_PYTHON}\" -U 'opencv-python==5.0.0.93' 'opencv-python-headless==5.0.0.93'
uv pip install --python \"\${VENV_PYTHON}\" -U \"numpy==${MARIE_NUMPY_VERSION}\"
uv pip check --python \"\${VENV_PYTHON}\" || true
python - <<'PY'
import importlib.util
for name in ('pytesseract', 'torch', 'torchvision', 'timm', 'nltk'):
    print(name, importlib.util.find_spec(name))
PY"
}

step_cuda_toolkit() {
  run_venv cuda-toolkit "uv pip install --python \"\${VENV_PYTHON}\" 'nvidia-cuda-nvcc==13.0.88' 'nvidia-nvvm==13.0.88' 'nvidia-cuda-crt==13.0.88' 'nvidia-cuda-cccl==13.0.85'
export CUDA_HOME=\"${MARIE_TORCH_VENV}/lib/python3.12/site-packages/nvidia/cu13\"
export PATH=\"\${CUDA_HOME}/bin:\${PATH}\"
export LD_LIBRARY_PATH=\"\${CUDA_HOME}/lib:\${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH:-}\"
test -x \"\${CUDA_HOME}/bin/nvcc\"
test -f \"\${CUDA_HOME}/include/nv/target\"
mkdir -p \"\${CUDA_HOME}/lib\"
test -e \"\${CUDA_HOME}/lib64\" || ln -s lib \"\${CUDA_HOME}/lib64\"
cd \"\${CUDA_HOME}/lib\"
test -e libcudart.so || ln -s libcudart.so.13 libcudart.so
if test -e libcublas.so.13 && ! test -e libcublas.so; then ln -s libcublas.so.13 libcublas.so; fi
if test -e libcublasLt.so.13 && ! test -e libcublasLt.so; then ln -s libcublasLt.so.13 libcublasLt.so; fi
nvcc --version
python - <<'PY'
import importlib.metadata as metadata
import os
import torch
from torch.utils.cpp_extension import CUDA_HOME

for dist, version in (
    ('nvidia-cuda-nvcc', '13.0.88'),
    ('nvidia-nvvm', '13.0.88'),
    ('nvidia-cuda-crt', '13.0.88'),
    ('nvidia-cuda-cccl', '13.0.85'),
):
    actual = metadata.version(dist)
    print(dist, actual)
    assert actual == version, (dist, actual)
print('env CUDA_HOME', os.environ.get('CUDA_HOME'))
print('torch CUDA_HOME', CUDA_HOME)
assert torch.cuda.is_available()
assert os.environ.get('CUDA_HOME')
PY"
}

step_build_fastwer() {
  run_venv build-fastwer "cd \"${MARIE_TORCH_WORKTREE}\"
mkdir -p \"${MARIE_WHEELS_DIR}\"
uv build --wheel --python \"\${VENV_PYTHON}\" --out-dir \"${MARIE_WHEELS_DIR}\" \"${MARIE_WHEELS_DIR}/fastwer-0.1.3.tar.gz\"
ls -l \"${MARIE_WHEELS_DIR}\"/fastwer-0.1.3-cp312-*.whl
sha256sum \"${MARIE_WHEELS_DIR}\"/fastwer-0.1.3-cp312-*.whl"
}

step_build_fairseq() {
  run_venv build-fairseq "mkdir -p \"${MARIE_SOURCE_ROOT}\" \"${MARIE_WHEELS_DIR}\"
cd \"${MARIE_SOURCE_ROOT}\"
if test -d fairseq/.git; then
  echo \"reusing source checkout: ${MARIE_SOURCE_ROOT}/fairseq\"
else
  git clone \"${MARIE_FAIRSEQ_REPO}\" fairseq
fi
cd fairseq
git fetch --tags origin '+refs/heads/*:refs/remotes/origin/*'
git checkout \"${MARIE_FAIRSEQ_REF}\"
git rev-parse HEAD
$(apply_patch_once_cmd "${FAIRSEQ_PATCH}")
uv pip install --python \"\${VENV_PYTHON}\" -U \"numpy==${MARIE_NUMPY_VERSION}\"
uv build -v --wheel --python \"\${VENV_PYTHON}\" --no-build-isolation --out-dir \"${MARIE_WHEELS_DIR}\" .
ls -l \"${MARIE_WHEELS_DIR}\"/fairseq-*.whl
sha256sum \"${MARIE_WHEELS_DIR}\"/fairseq-*.whl"
}

step_build_detectron2() {
  run_venv build-detectron2 "mkdir -p \"${MARIE_SOURCE_ROOT}\" \"${MARIE_WHEELS_DIR}\"
cd \"${MARIE_SOURCE_ROOT}\"
if test -d detectron2/.git; then
  echo \"reusing source checkout: ${MARIE_SOURCE_ROOT}/detectron2\"
else
  git clone \"${MARIE_DETECTRON2_REPO}\" detectron2
fi
cd detectron2
git fetch --tags origin '+refs/heads/*:refs/remotes/origin/*'
git checkout \"${MARIE_DETECTRON2_REF}\"
git rev-parse HEAD
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST=\"${MARIE_TORCH_CUDA_ARCH_LIST}\"
uv pip install --python \"\${VENV_PYTHON}\" -U \"numpy==${MARIE_NUMPY_VERSION}\"
uv pip install --python \"\${VENV_PYTHON}\" git+https://github.com/facebookresearch/fvcore
uv build -v --wheel --python \"\${VENV_PYTHON}\" --no-build-isolation --out-dir \"${MARIE_WHEELS_DIR}\" .
ls -l \"${MARIE_WHEELS_DIR}\"/detectron2-*.whl
sha256sum \"${MARIE_WHEELS_DIR}\"/detectron2-*.whl"
}

step_reject_faiss_cu12() {
  run_venv reject-faiss-cu12 "cd \"${MARIE_TORCH_WORKTREE}\"
if rg -n \"faiss-gpu-cu12\" pyproject.toml uv.lock; then
  echo \"faiss-gpu-cu12 must not be present in the torch 2.12/cu130 lane\" >&2
  exit 1
fi
uv pip install --python \"\${VENV_PYTHON}\" --dry-run --ignore-installed faiss-gpu-cu12 || true
uv pip install --python \"\${VENV_PYTHON}\" --dry-run --ignore-installed 'faiss-gpu-cu12' \"numpy==${MARIE_NUMPY_VERSION}\" || true"
}

step_build_faiss() {
  run_venv build-faiss "mkdir -p \"${MARIE_SOURCE_ROOT}\" \"${MARIE_WHEELS_DIR}\"
cd \"${MARIE_SOURCE_ROOT}\"
if test -d faiss/.git; then
  echo \"reusing source checkout: ${MARIE_SOURCE_ROOT}/faiss\"
else
  git clone \"${MARIE_FAISS_REPO}\" faiss
fi
cd faiss
git fetch --tags origin '+refs/heads/*:refs/remotes/origin/*'
git checkout \"${MARIE_FAISS_REF}\"
git rev-parse HEAD
$(apply_patch_once_cmd "${FAISS_PATCH}")
uv pip install --python \"\${VENV_PYTHON}\" -U \"numpy==${MARIE_NUMPY_VERSION}\" packaging swig cmake ninja
blas_library=\"\$(ldconfig -p | awk '/libopenblas\\.so / {print \$NF; found=1; exit} /libblas\\.so\\.3 / {fallback=\$NF} END {if (!found && fallback) print fallback}')\"
if [[ -z \"\${blas_library}\" ]]; then
  echo \"Missing BLAS library. The Dockerfiles install libopenblas-dev; install it with: sudo apt-get install -y libopenblas-dev\" >&2
  exit 1
fi
lapack_library=\"\$(ldconfig -p | awk '/liblapack\\.so\\.3 / {print \$NF; exit}')\"
if [[ -z \"\${lapack_library}\" ]]; then
  echo \"Missing LAPACK library. The Dockerfiles install libopenblas-dev; install it with: sudo apt-get install -y libopenblas-dev\" >&2
  exit 1
fi
echo \"BLAS_LIBRARIES=\${blas_library}\"
echo \"LAPACK_LIBRARIES=\${lapack_library};\${blas_library}\"
cmake -B build -S . \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_PYTHON=ON \
  -DFAISS_ENABLE_C_API=OFF \
  -DBUILD_TESTING=OFF \
  -DFAISS_OPT_LEVEL=generic \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=\"${MARIE_FAISS_CUDA_ARCH}\" \
  -DBLAS_LIBRARIES=\"\${blas_library}\" \
  -DLAPACK_LIBRARIES=\"\${lapack_library};\${blas_library}\" \
  -DCUDAToolkit_ROOT=\"\${CUDA_HOME}\" \
  -DCMAKE_CUDA_COMPILER=\"\${CUDA_HOME}/bin/nvcc\" \
  -DPython_EXECUTABLE=\"\$(python -c 'import sys; print(sys.executable)')\"
cmake --build build -j \"${MARIE_BUILD_JOBS}\"
cd build/faiss/python
uv build --wheel --python \"\${VENV_PYTHON}\" --no-build-isolation --out-dir \"${MARIE_WHEELS_DIR}\" .
ls -l \"${MARIE_WHEELS_DIR}\"/faiss*.whl
sha256sum \"${MARIE_WHEELS_DIR}\"/faiss*.whl"
}

step_check_wheels() {
  run_cmd check-wheels "cd \"${MARIE_TORCH_WORKTREE}\"
test ! -d \"${MARIE_WHEELS_DIR}/resolver-spillover\"
shopt -s nullglob
fairseq=(\"${MARIE_WHEELS_DIR}\"/fairseq-*.whl)
detectron2=(\"${MARIE_WHEELS_DIR}\"/detectron2-*.whl)
faiss=(\"${MARIE_WHEELS_DIR}\"/faiss*.whl)
if [[ \${#fairseq[@]} -ne 1 ]]; then echo \"expected exactly one fairseq wheel, found \${#fairseq[@]}\" >&2; exit 1; fi
if [[ \${#detectron2[@]} -ne 1 ]]; then echo \"expected exactly one detectron2 wheel, found \${#detectron2[@]}\" >&2; exit 1; fi
if [[ \${#faiss[@]} -ne 1 ]]; then echo \"expected exactly one faiss wheel, found \${#faiss[@]}\" >&2; exit 1; fi
unexpected=0
for path in \"${MARIE_WHEELS_DIR}\"/*.whl; do
  base=\"\$(basename \"\${path}\")\"
  case \"\${base}\" in
    etcd3-0.12.0-py2.py3-none-any.whl|fastwer-0.1.3-cp312-*.whl|fairseq-*.whl|detectron2-*.whl|faiss*.whl) ;;
    *) echo \"unexpected wheel in wheels/: \${base}\" >&2; unexpected=1 ;;
  esac
done
if [[ \${unexpected} -ne 0 ]]; then exit 1; fi
find \"${MARIE_WHEELS_DIR}\" -maxdepth 1 -type f -name '*.whl' -printf '%f\\n' | sort"
}

step_install_wheels() {
  run_venv install-wheels "cd \"${MARIE_TORCH_WORKTREE}\"
export UV_PROJECT_ENVIRONMENT=\"${MARIE_TORCH_VENV}\"
uv sync --locked --extra cu130 --group dev --group legacy-gradio --python \"\${VENV_PYTHON}\"
python patches/patch-omegaconf-py312.py --no-confirm
python patches/patch-detectron2-metadata.py --no-confirm
uv pip check --python \"\${VENV_PYTHON}\""
}

step_verify_native() {
  run_venv verify-native "python - <<'PY'
import importlib.metadata as metadata
import importlib.util
import numpy as np
import torch
import torchvision
import fairseq
import detectron2
import faiss
from detectron2 import _C
from detectron2.layers import ROIAlign
from marie.utils.patches import patchify

print('python packages')
for dist in ('torch', 'torchvision', 'numpy', 'fairseq', 'detectron2', 'faiss-gpu-cu13', 'fastwer', 'timm'):
    print(dist, metadata.version(dist))
print('pytesseract spec', importlib.util.find_spec('pytesseract'))
print('faiss import version', getattr(faiss, '__version__', 'unknown'))

assert torch.__version__.startswith('2.12.1'), torch.__version__
assert torch.version.cuda == '13.0', torch.version.cuda
assert torch.cuda.is_available()
x = torch.randn(1, 1, 8, 8, device='cuda')
boxes = torch.tensor([[0, 0, 0, 7, 7]], dtype=torch.float32, device='cuda')
out = ROIAlign((2, 2), 1.0, 2, True)(x, boxes)
assert out.is_cuda, out.device
ngpu = faiss.get_num_gpus()
assert ngpu > 0, ngpu
res = faiss.StandardGpuResources()
index = faiss.GpuIndexFlatL2(res, 16)
xb = np.random.random((128, 16)).astype('float32')
index.add(xb)
distances, indices = index.search(xb[:2], 2)
image = np.random.default_rng(0).integers(0, 255, (512, 512, 3), dtype=np.uint8)
patches = patchify(image, (128, 128, 3), step=128)
reconstructed = patches[:, :, 0].transpose(0, 2, 1, 3, 4).reshape(image.shape)
assert patches.shape == (4, 4, 1, 128, 128, 3), patches.shape
assert np.array_equal(image, reconstructed)
print('detectron2_ext', _C.__name__, tuple(out.shape), out.device)
print('faiss_gpus', ngpu, 'search', indices.tolist(), distances.tolist())
print('patchify_roundtrip', patches.shape, bool(np.array_equal(image, reconstructed)))
PY"
}

step_verify_tests() {
  run_venv verify-tests "cd \"${MARIE_TORCH_WORKTREE}\"
python -m pytest tests/unit/engine/ -x -q"
}

step_verify_gradio() {
  run_venv verify-gradio "export MARIE_DEFAULT_MOUNT=\"${MARIE_MODEL_MOUNT}\"
test -d \"${MARIE_MODEL_MOUNT}/model_zoo\"
cd \"${MARIE_TORCH_WORKTREE}/workspaces/bounding-boxes-gradio\"
python - <<'PY'
import importlib.util
from pathlib import Path
from PIL import Image

spec = importlib.util.spec_from_file_location('bb_app', 'app.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sample = Path('../../assets/psm/block/block-001.png').resolve()
boxes_img, lines_img, box_count, line_count = mod.processor.process_image(Image.open(sample).convert('RGB'))
print('bounding-boxes sample', sample)
print('box_count', box_count)
print('line_count', line_count)
print('boxes_img', getattr(boxes_img, 'size', None))
print('lines_img', getattr(lines_img, 'size', None))
assert box_count > 0
assert line_count > 0
PY
cd \"${MARIE_TORCH_WORKTREE}/workspaces/ocr-gradio\"
python - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('ocr_app', 'app.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sample = Path('../../assets/psm/block/block-001.png').resolve()
bboxes_img, overlay_image, lines_img, result_json, text = mod.process_image(str(sample))
print('ocr sample', sample)
print('bboxes_img', getattr(bboxes_img, 'size', None))
print('overlay_shape', getattr(overlay_image, 'shape', None))
print('lines_img', getattr(lines_img, 'size', None))
print('result_json_len', len(result_json))
print('text_len', len(text))
print('text_preview', text[:160].replace('\\n', ' '))
assert len(result_json) > 1000
assert len(text) > 100
PY"
}

step_manifest() {
  run_venv manifest "mkdir -p \"${MARIE_REPRO_LOG_DIR}\"
{
  echo '# Reproducible PyTorch 2.12/cu130 py312 run'
  echo
  echo '## Paths'
  echo \"worktree=${MARIE_TORCH_WORKTREE}\"
  echo \"env_name=${MARIE_TORCH_ENV_NAME}\"
  echo \"venv=${MARIE_TORCH_VENV}\"
  echo \"wheels_dir=${MARIE_WHEELS_DIR}\"
  echo \"source_root=${MARIE_SOURCE_ROOT}\"
  echo
  echo '## Git'
  git -C \"${MARIE_TORCH_WORKTREE}\" status --short --branch
  git -C \"${MARIE_TORCH_WORKTREE}\" rev-parse HEAD
  echo
  echo '## Wheels'
  ls -l \"${MARIE_WHEELS_DIR}\"/*.whl
  sha256sum \"${MARIE_WHEELS_DIR}\"/*.whl
} > \"${MARIE_REPRO_LOG_DIR}/manifest.md\"
uv pip freeze --python \"\${VENV_PYTHON}\" > \"${MARIE_REPRO_LOG_DIR}/uv-freeze.txt\"
python -m torch.utils.collect_env > \"${MARIE_REPRO_LOG_DIR}/torch-collect-env.txt\"
cat \"${MARIE_REPRO_LOG_DIR}/manifest.md\""
}

step_wheels_readme() {
  run_cmd wheels-readme "cd \"${MARIE_TORCH_WORKTREE}\"
MARIE_WHEELS_DIR=\"${MARIE_WHEELS_DIR}\" python3 - <<'PY'
from pathlib import Path
import hashlib
import os

wheels_dir = Path(os.environ['MARIE_WHEELS_DIR'])
readme = wheels_dir / 'README.md'
start = '<!-- local-wheels-inventory:start -->'
end = '<!-- local-wheels-inventory:end -->'

rows = []
for path in sorted(wheels_dir.iterdir(), key=lambda item: item.name):
    if path.suffix not in {'.whl', '.gz'}:
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows.append(f'| {path.name} | {path.stat().st_size} | {digest} |')

block = '\\n'.join([
    start,
    '',
    '## Generated file inventory',
    '',
    'Updated by scripts/setup-py312-torch212-cu130.sh wheels-readme.',
    '',
    '| File | Size bytes | SHA256 |',
    '| --- | ---: | --- |',
    *rows,
    '',
    end,
]) + '\\n'

text = readme.read_text()
if start in text and end in text:
    before = text.split(start, 1)[0].rstrip()
    after = text.split(end, 1)[1].lstrip()
    text = f'{before}\\n\\n{block}\\n{after}'
else:
    text = f'{text.rstrip()}\\n\\n{block}'

readme.write_text(text)
print(block)
PY"
}

run_all() {
  step_env
  step_worktree
  step_venv
  step_torch
  step_app_deps
  step_cuda_toolkit
  step_build_fastwer
  step_build_fairseq
  step_build_detectron2
  step_reject_faiss_cu12
  step_build_faiss
  step_check_wheels
  step_install_wheels
  step_verify_native
  step_verify_tests
  step_verify_gradio
  step_manifest
  step_wheels_readme
}

run_build_wheels() {
  require_build_venv
  step_cuda_toolkit
  step_build_fastwer
  step_build_fairseq
  step_build_detectron2
  step_reject_faiss_cu12
  step_build_faiss
  step_check_wheels
  step_wheels_readme
}

main() {
  local step="${1:-help}"
  case "${step}" in
    help|-h|--help)
      usage
      return 0
      ;;
  esac

  resolve_environment
  write_invocation "${step}"

  case "${step}" in
    venv|all) guard_new_environment "${step}" ;;
  esac

  case "${step}" in
    env) step_env ;;
    worktree) step_worktree ;;
    venv) step_venv ;;
    torch) step_torch ;;
    app-deps) step_app_deps ;;
    cuda-toolkit) step_cuda_toolkit ;;
    build-fastwer) step_build_fastwer ;;
    build-fairseq) step_build_fairseq ;;
    build-detectron2) step_build_detectron2 ;;
    reject-faiss-cu12) step_reject_faiss_cu12 ;;
    build-faiss) step_build_faiss ;;
    build-wheels) run_build_wheels ;;
    check-wheels) step_check_wheels ;;
    install-wheels) step_install_wheels ;;
    verify-native) step_verify_native ;;
    verify-tests) step_verify_tests ;;
    verify-gradio) step_verify_gradio ;;
    manifest) step_manifest ;;
    wheels-readme) step_wheels_readme ;;
    all) run_all ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
