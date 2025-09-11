docker run --gpus '"device=0,1,2,3,4,5,6,7"' \
  --shm-size 64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /var/log/sglang:/logs \
  --ipc=host \
  --privileged \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
    --model-path Qwen/QwQ-32B \
    --host 0.0.0.0 --port 8000 \
    --trust-remote-code \
    --attention-backend flashinfer \
    --quantization fp8 \
    --chat-template qwen2-vl \
    --dp-size 4 \
    --tp 2 \
    --max-running-requests 256 \
    --max-queued-requests 10000 \
    --mem-fraction-static 0.7 \
    --schedule-policy fcfs \
    --schedule-conservativeness 0.1 \
    --log-level info \
    --log-level-http error \
    --log-requests \
    --log-requests-level 1 \
    --enable-metrics \
    --enable-metrics-for-all-schedulers \
    --crash-dump-folder /logs/crash_dumps

