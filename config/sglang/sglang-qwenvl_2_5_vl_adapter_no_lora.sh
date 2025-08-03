docker run --gpus '"device=0"' \
  --shm-size 32g \
  -p 8000:8000 \
  -e OMP_NUM_THREADS=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt/data/training/saves:/adapter \
  --ipc=host \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --trust-remote-code \
    --chat-template qwen2-vl \
    --attention-backend flashinfer \
    --disable-radix-cache
