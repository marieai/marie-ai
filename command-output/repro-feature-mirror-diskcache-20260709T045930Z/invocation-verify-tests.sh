#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/greg/dev/marieai/worktrees/marie-ai-pytorch-2-12
export MARIE_AI_ROOT=/home/greg/dev/marieai/worktrees/marie-ai-pytorch-2-12
export MARIE_TORCH_WORKTREE=/home/greg/dev/marieai/worktrees/marie-ai-pytorch-2-12
export MARIE_VENV_BASE=/home/greg/dev/marieai/worktrees/.venvs
export MARIE_TORCH_ENV_NAME=marie-ai-pytorch-2-12-validate-main-20260709T040215Z
export MARIE_TORCH_VENV=/home/greg/dev/marieai/worktrees/.venvs/marie-ai-pytorch-2-12-validate-main-20260709T040215Z
export MARIE_WHEELS_DIR=/home/greg/dev/marieai/worktrees/marie-ai-pytorch-2-12/wheels
export MARIE_MODEL_MOUNT=/mnt/data/marie-ai
export MARIE_REPRO_RUN_ID=feature-mirror-diskcache-20260709T045930Z
export MARIE_REPRO_LOG_DIR=/home/greg/dev/marieai/worktrees/marie-ai-pytorch-2-12/command-output/repro-feature-mirror-diskcache-20260709T045930Z
export MARIE_SOURCE_ROOT=/home/greg/dev/marieai/worktrees/sources/torch-2.12-cu130-feature-mirror-diskcache-20260709T045930Z
export MARIE_FAIRSEQ_REF=main
export MARIE_DETECTRON2_REF=main
export MARIE_FAISS_REF=5622e93733b64b2e033362dbdfda019b2ab33ef0
export MARIE_FAIRSEQ_REPO=https://github.com/marieai/fairseq.git
export MARIE_DETECTRON2_REPO=https://github.com/facebookresearch/detectron2.git
export MARIE_FAISS_REPO=https://github.com/facebookresearch/faiss.git
export MARIE_NUMPY_VERSION=2.4.6
export MARIE_TORCH_CUDA_ARCH_LIST=7.5\;8.0\;8.6\;8.9\;9.0
export MARIE_FAISS_CUDA_ARCH=89
export MARIE_BUILD_JOBS=24

/home/greg/dev/marieai/worktrees/marie-ai-pytorch-2-12/tools/scripts/setup-py312-torch212-cu130.sh verify-tests
