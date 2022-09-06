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
```

https://medium.com/javarevisited/monitoring-setup-with-docker-compose-part-1-prometheus-3d2c9089ee82
https://github.com/vegasbrianc/docker-traefik-prometheus/blob/master/56k_Cloud_Traefik_Monitoring.pdf
https://traefik.io/blog/capture-traefik-metrics-for-apps-on-kubernetes-with-prometheus/
https://github.com/TheYkk/traefik-whoami/blob/master/docker-compose.yml
https://github.com/nightmareze1/traefik-prometheus-metrics