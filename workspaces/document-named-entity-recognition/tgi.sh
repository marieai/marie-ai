model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B 
# model=/home/gbugaj/dev/marieai/marie-ai/workspaces/document-named-entity-recognition/DeepSeek-R1-Distill-Qwen-14B-exl2-8_0
volume=~/.cache/huggingface

docker run --gpus all --shm-size 64g -p 8090:80 -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:3.1.1 \
    --model-id $model --quantize eetq