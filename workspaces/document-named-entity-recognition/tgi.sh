model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B 
volume=~/.cache/huggingface

docker run --gpus all --shm-size 64g -p 8090:80 -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:3.1.1 \
    --model-id $model --quantize eetq