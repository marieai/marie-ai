## Traefik

Password should be generated using `htpasswd` (md5, sha1 or bcrypt)

For passwords stored in `user-credential` passwords need to be generated without escaping `$`
```sh
htpasswd -nb admin secure_password
```

### Sample config

Set up the `auth` middleware to be basicauth that takes a file for users
```yaml
  middlewares:
    auth:
      basicAuth:
        removeHeader: true
        usersFile: /user-credentials
#        users:
#          - "dashboard:$apr1$2TgJEKkl$.fyRx.XI5l0AIm/bef4Rw."
#    redirect-to-https:
#      redirectScheme:
#        scheme:

```

For password hardcoded in middlewares directly via `users` node.

```sh
echo $(htpasswd -nB dashboard) | sed -e s/\\$/\\$\\$/g
```

### Sample config

```yaml
  middlewares:
    auth:
      basicAuth:
        removeHeader: true
        users:
          - "dashboard:$$2y$$05$$6zECIStqygUCGeKl/zog/up2Hu2vADiDJfw6SLd0cCSepU80czGS2"
```

Bootstrap 
```sh
docker compose down && docker compose -f docker-compose.yml --project-directory . up  traefik whoami  --build  --remove-orphans
docker compose down && docker compose -f docker-compose.yml --project-directory . up consul-server grafana prometheus traefik whoami --build  --remove-orphans
```

## Endpoints

http://localhost:8500/ui/dc1/services
http://traefik.localhost:7777/metrics
http://traefik.localhost:7777/dashboard/#/http/routers
http://localhost:7777/ping
http://localhost:3000/?orgId=1
http://localhost:9090/targets?search=

### Metrics
* [metrics - cadvisor](http://localhost:8077/metrics)  http://localhost:8077/metrics
* [metrics - DCGM](http://localhost:9400/metrics)   http://localhost:9400/metrics
* [metrics - node-exporter](http://localhost:9400/metrics) http://localhost:9400/metrics

### Monitoring
* [Promtail UI](http://localhost:9080/targets)
* [Alertmanager UI](http://localhost:9093/#/status)

## Resources

https://github.com/Einsteinish/Docker-Compose-Prometheus-and-Grafana
https://github.com/kpritam/prometheus-consul-grafana/blob/master/docker-compose.yml
https://medium.com/javarevisited/monitoring-setup-with-docker-compose-part-1-prometheus-3d2c9089ee82
https://github.com/vegasbrianc/docker-traefik-prometheus/blob/master/56k_Cloud_Traefik_Monitoring.pdf
https://traefik.io/blog/capture-traefik-metrics-for-apps-on-kubernetes-with-prometheus/
https://github.com/TheYkk/traefik-whoami/blob/master/docker-compose.yml
https://github.com/nightmareze1/traefik-prometheus-metrics
https://medium.com/trendyol-tech/consul-prometheus-monitoring-service-discovery-7190bae50516
https://github.com/rfmoz/grafana-dashboards/blob/master/prometheus/node-exporter-full.json
https://yetiops.net/posts/prometheus-consul-node_exporter/
https://grafana.com/grafana/dashboards/1860-node-exporter-full/

## Tracking

https://github.com/jina-ai/clip-as-service/blob/main/server/setup.py
https://dev.to/aleksk1ng/go-kafka-grpc-and-mongodb-microservice-with-metrics-and-tracing-448d
Zabbix
https://www.jaegertracing.io/

sudo mount /mnt/data/marie-ai && docker container ls -aq | xargs --no-run-if-empty docker stop && docker rm $(docker ps --filter status=exited -q) && cd ~/dev/marie-ai/docker-util && ./run-all.sh


