docker run --gpus '"device=0,1,2,3,4,5,6,7"' \
  --shm-size 32g \
  -p 8000:8000 \
  -e OMP_NUM_THREADS=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --ipc=host \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-VL-72B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --trust-remote-code \
    --attention-backend flashinfer \
    --quantization fp8 \
    --chat-template qwen2-vl \
    --attention-backend flashinfer \
    --dp-size 4 \
    --tp 2 \
    --pp-size 1 \
    --schedule-policy fcfs \
    --schedule-conservativeness 0.1 \
    --max-running-requests 4 \
    --mem-fraction-static 0.7 \
    --disable-radix-cache \
    --disable-cuda-graph \
    --context-len 32000 \
    --max-total-tokens 32768 \
    --max-prefill-tokens 32000