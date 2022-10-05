# Docker container monitoring via cAdvisor and DCGM-Exporter

# http://localhost:8077/metrics
VERSION=v0.45.0 # use the latest release version from https://github.com/google/cadvisor/releases
# VERSION=latest
sudo docker run \
  --volume=/:/rootfs:ro \
  --volume=/var/run:/var/run:ro \
  --volume=/sys:/sys:ro \
  --volume=/var/lib/docker/:/var/lib/docker:ro \
  --volume=/dev/disk/:/dev/disk:ro \
  --publish=8077:8080 \
  --detach=true \
  --name=cadvisor \
  --privileged \
  --device=/dev/kmsg \
  gcr.io/cadvisor/cadvisor:$VERSION

#  https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/dcgm-exporter.html

# http://localhost:9400/metrics
DCGM_EXPORTER_VERSION=2.4.6-2.6.10 && \
docker run -d --rm \
   --gpus all \
   --net host \
   --cap-add SYS_ADMIN \
   nvcr.io/nvidia/k8s/dcgm-exporter:${DCGM_EXPORTER_VERSION}-ubuntu20.04

# Setup up monitoring via node-exporter
# http://localhost:9100/metrics
docker run -d \
  --net="host" \
  --pid="host" \
  -v "/:/host:ro,rslave" \
  quay.io/prometheus/node-exporter:latest \
  --path.rootfs=/host

# Setup Grafana Loki and Promtail
# http://localhost:3100/metrics
docker run --rm --net host -v $(pwd)/../config/grafana/promtail:/mnt/config \
  -v /var/log/marie/:/var/log/marie \
  grafana/promtail:2.6.1 \
  --config.file=/mnt/config/promtail-config.yaml

