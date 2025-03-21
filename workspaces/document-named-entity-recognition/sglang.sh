# DeepSeek-R1-Distill-Qwen-14B-Q6_K_L.gguf
# DeepSeek-R1-Distill-Qwen-14B-Q8_0.gguf

docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /home/greg/Downloads/DeepSeek-R1-Distill-Qwen-14B-Q6_K_L.gguf:/model/DeepSeek-R1-Distill-Qwen.gguf \
    --ipc=host \
    lmsysorg/sglang:v0.4.3.post3-cu124 \
    python3 -m sglang.launch_server --model-path /model/DeepSeek-R1-Distill-Qwen.gguf --load-format gguf \
    --quantization gguf  --tokenizer-path deepseek-ai/DeepSeek-R1-Distill-Qwen-14B   --host 0.0.0.0 --port 30000 \
    --reasoning-parser deepseek-r1
#    --grammar-backend xgrammar

# --enable-torch-compile --enable-dp-attention

# launch_server.py: error: argument --quantization: invalid choice: 'eetq' (choose from 'awq', 'fp8', 'gptq', 'marlin', 'gptq_marlin', 'awq_marlin', 'bitsandbytes', 'gguf', 'modelopt', 'w8a8_int8')
# https://github.com/sgl-project/sglang/pull/2215/files#diff-19c9f69429f350e2828dfc8b02381630b7b03067678731e726c5c26c57d9be87


# usage: launch_server.py [-h] --model-path MODEL_PATH
#                         [--tokenizer-path TOKENIZER_PATH] [--host HOST]
#                         [--port PORT] [--tokenizer-mode {auto,slow}]
#                         [--skip-tokenizer-init]
#                         [--load-format {auto,pt,safetensors,npcache,dummy,gguf,bitsandbytes,layered}]
#                         [--trust-remote-code]
#                         [--dtype {auto,half,float16,bfloat16,float,float32}]
#                         [--kv-cache-dtype {auto,fp8_e5m2,fp8_e4m3}]
#                         [--quantization-param-path QUANTIZATION_PARAM_PATH]
#                         [--quantization {awq,fp8,gptq,marlin,gptq_marlin,awq_marlin,bitsandbytes,gguf,modelopt,w8a8_int8}]
#                         [--context-length CONTEXT_LENGTH]
#                         [--device {cuda,xpu,hpu,cpu}]
#                         [--served-model-name SERVED_MODEL_NAME]
#                         [--chat-template CHAT_TEMPLATE] [--is-embedding]
#                         [--revision REVISION]
#                         [--mem-fraction-static MEM_FRACTION_STATIC]
#                         [--max-running-requests MAX_RUNNING_REQUESTS]
#                         [--max-total-tokens MAX_TOTAL_TOKENS]
#                         [--chunked-prefill-size CHUNKED_PREFILL_SIZE]
#                         [--max-prefill-tokens MAX_PREFILL_TOKENS]
#                         [--schedule-policy {lpm,random,fcfs,dfs-weight}]
#                         [--schedule-conservativeness SCHEDULE_CONSERVATIVENESS]
#                         [--cpu-offload-gb CPU_OFFLOAD_GB]
#                         [--prefill-only-one-req PREFILL_ONLY_ONE_REQ]
#                         [--tensor-parallel-size TENSOR_PARALLEL_SIZE]
#                         [--stream-interval STREAM_INTERVAL] [--stream-output]
#                         [--random-seed RANDOM_SEED]
#                         [--constrained-json-whitespace-pattern CONSTRAINED_JSON_WHITESPACE_PATTERN]
#                         [--watchdog-timeout WATCHDOG_TIMEOUT]
#                         [--download-dir DOWNLOAD_DIR]
#                         [--base-gpu-id BASE_GPU_ID] [--log-level LOG_LEVEL]
#                         [--log-level-http LOG_LEVEL_HTTP] [--log-requests]
#                         [--show-time-cost] [--enable-metrics]
#                         [--decode-log-interval DECODE_LOG_INTERVAL]
#                         [--api-key API_KEY]
#                         [--file-storage-pth FILE_STORAGE_PTH]
#                         [--enable-cache-report]
#                         [--data-parallel-size DATA_PARALLEL_SIZE]
#                         [--load-balance-method {round_robin,shortest_queue}]
#                         [--expert-parallel-size EXPERT_PARALLEL_SIZE]
#                         [--dist-init-addr DIST_INIT_ADDR] [--nnodes NNODES]
#                         [--node-rank NODE_RANK]
#                         [--json-model-override-args JSON_MODEL_OVERRIDE_ARGS]
#                         [--lora-paths [LORA_PATHS ...]]
#                         [--max-loras-per-batch MAX_LORAS_PER_BATCH]
#                         [--lora-backend LORA_BACKEND]
#                         [--attention-backend {flashinfer,triton,torch_native}]
#                         [--sampling-backend {flashinfer,pytorch}]
#                         [--grammar-backend {xgrammar,outlines}]
#                         [--enable-flashinfer-mla]
#                         [--speculative-algorithm {EAGLE,NEXTN}]
#                         [--speculative-draft-model-path SPECULATIVE_DRAFT_MODEL_PATH]
#                         [--speculative-num-steps SPECULATIVE_NUM_STEPS]
#                         [--speculative-num-draft-tokens SPECULATIVE_NUM_DRAFT_TOKENS]
#                         [--speculative-eagle-topk {1,2,4,8}]
#                         [--enable-double-sparsity]
#                         [--ds-channel-config-path DS_CHANNEL_CONFIG_PATH]
#                         [--ds-heavy-channel-num DS_HEAVY_CHANNEL_NUM]
#                         [--ds-heavy-token-num DS_HEAVY_TOKEN_NUM]
#                         [--ds-heavy-channel-type DS_HEAVY_CHANNEL_TYPE]
#                         [--ds-sparse-decode-threshold DS_SPARSE_DECODE_THRESHOLD]
#                         [--disable-radix-cache] [--disable-jump-forward]
#                         [--disable-cuda-graph] [--disable-cuda-graph-padding]
#                         [--enable-nccl-nvls] [--disable-outlines-disk-cache]
#                         [--disable-custom-all-reduce] [--disable-mla]
#                         [--disable-overlap-schedule] [--enable-mixed-chunk]
#                         [--enable-dp-attention] [--enable-ep-moe]
#                         [--enable-torch-compile]
#                         [--torch-compile-max-bs TORCH_COMPILE_MAX_BS]
#                         [--cuda-graph-max-bs CUDA_GRAPH_MAX_BS]
#                         [--cuda-graph-bs CUDA_GRAPH_BS [CUDA_GRAPH_BS ...]]
#                         [--torchao-config TORCHAO_CONFIG]
#                         [--enable-nan-detection] [--enable-p2p-check]
#                         [--triton-attention-reduce-in-fp32]
#                         [--triton-attention-num-kv-splits TRITON_ATTENTION_NUM_KV_SPLITS]
#                         [--num-continuous-decode-steps NUM_CONTINUOUS_DECODE_STEPS]
#                         [--delete-ckpt-after-loading] [--enable-memory-saver]
#                         [--allow-auto-truncate]
#                         [--enable-custom-logit-processor]
#                         [--return-hidden-states]
#                         [--tool-call-parser {qwen25,mistral,llama3}]
#                         [--enable-hierarchical-cache]
