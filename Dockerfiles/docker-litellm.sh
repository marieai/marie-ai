

data_dir="${DATA_DIR:-${MARIE_DEFAULT_MOUNT:-/mnt/data/marie-ai}}"

docker run -it \
  -v "$data_dir/config/litellm/config.prod.yml:/app/config.yaml" \
  -p 4000:4000 \
  ghcr.io/berriai/litellm:litellm_stable_release_branch-v1.74.3-stable.patch.2 \
  --port 4000 --config /app/config.yaml  --num_workers 8
