---
sidebar_position: 2
---

# Traefik

Configuration is located in  `config/traefik` 

```
./traefik/
├── file-provider.yml
├── prometheus-auth.yaml
├── README.md
├── traefik.yml
└── user-credentials
```


## Authentication

### Basic Authentication

Password should be generated using `htpasswd` (md5, sha1 or bcrypt)

For passwords stored in `user-credential` passwords need to be generated without escaping `$`
```sh
htpasswd -nb admin secure_password
```

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

### Hardcoded Authentication

For password hardcoded in middlewares directly via `users` node.

```sh
echo $(htpasswd -nB dashboard) | sed -e s/\\$/\\$\\$/g
```


```yaml
  middlewares:
    auth:
      basicAuth:
        removeHeader: true
        users:
          - "dashboard:$$2y$$05$$6zECIStqygUCGeKl/zog/up2Hu2vADiDJfw6SLd0cCSepU80czGS2"
```

## Testing configuration

```sh
docker compose down && docker compose -f docker-compose.yml --project-directory . up  traefik whoami  --build  --remove-orphans
```
