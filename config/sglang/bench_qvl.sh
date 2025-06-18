docker run --gpus all \
  --shm-size 32g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --ipc=host \
  lmsysorg/sglang:latest \
  python3 -m sglang.bench_serving \
    --backend sglang \
    --num-prompts 100 \
    --dataset-name random --random-input-len 4000 --random-output-len 500 --random-range-ratio 1 --request-rate 100 \
    --host 172.17.0.1 \
    --port 8000
