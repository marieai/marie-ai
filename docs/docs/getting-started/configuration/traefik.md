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

## Certificates

Default configuration is located in `./traefik/provider/tls.yml` if you need to add domain or change certificates you 
can add them here.

###  Localhost
For local development we follow a [guide](https://letsencrypt.org/docs/certificates-for-localhost/) created by [letsencrypt.org](http://letsencrypt.org) 

#### Making and trusting your own certificates
The simplest way to generate a private key and self-signed certificate for `localhost` is with this openssl command:

```shell
openssl req -x509 -out traefik.localhost.crt -keyout traefik.localhost.key \
  -newkey rsa:2048 -nodes -sha256 \
  -subj '/CN=traefik.localhost' -extensions EXT -config <( \
   printf "[dn]\nCN=traefik.localhost\n[req]\ndistinguished_name = dn\n[EXT]\nsubjectAltName=DNS:traefik.localhost\nkeyUsage=digitalSignature\nextendedKeyUsage=serverAuth")
```

Config for this option will look as follow:

```yaml
tls:
  stores:
    default:
      defaultCertificate:
        certFile: /letsencrypt/traefik.localhost.crt
        keyFile: /letsencrypt/traefik.localhost.key
```

### Let's Encrypt

This process allows us to configure FQDN use Lets Encrypt as default ACME provider.

```yaml
# Dynamic Transport Layer Security configuration
# https://doc.traefik.io/traefik/https/tls/

tls:
  stores:
    default:
      defaultGeneratedCert:
        resolver: http-resolver
        domain:
          main: marie-ai.com
          sans:
            - ops-001.marie-ai.com

  # When testing certs, enable this so traefik doesn't use its own self-signed cert for unknown domains.
  options:
    default:
      sniStrict: false
```

### Use Your Own Certificates

You wil need your certificate (.crt) and private key (.key) for each domain you like to use.
Then add them in the `tls.yml` configuration as follows.

```yaml
tls:
  certificates:
    - certFile: /certs/ops-001.marie-ai.com.crt
      keyFile: /certs/ops-001.marie-ai.com.key
```


### Testing certificates

For testing, we will setup a virtual domain `ops-001.marie-ai.com` and add it to our `/etc/hosts`. 

```shell
cat /etc/hosts

127.0.0.1	localhost
127.0.0.1   ops-001.marie-ai.com
```

When testing certs, enable `sniStrict` so traefik doesn't use its own self-signed cert for unknown domains. 
If the validation fail for the domain you will get following error.

```
SSL peer has no certificate for the requested DNS name
Error code: SSL_ERROR_UNRECOGNIZED_NAME_ALERT
```

This error means that the name on the certificate is not recognized and is usually caused by a SSL configuration error. 
Check `sniStrict: false` is present.


```yaml
# Dynamic Transport Layer Security configuration
# https://doc.traefik.io/traefik/https/tls/

tls:
  stores:
    default:
      defaultCertificate:
        certFile: /letsencrypt/traefik.localhost.crt
        keyFile: /letsencrypt/traefik.localhost.key
#      ENABLE TO Auto Generate CERT
#      defaultGeneratedCert:
#        resolver: http-resolver
#        domain:
#          main: marie-ai.com
#          sans:
#            - ops-001.marie-ai.com

  # When testing certs, enable this so traefik doesn't use its own self-signed cert for unknown domains.
  options:
    default:
      sniStrict: false
```

## Testing configuration

```sh
docker compose down && docker compose -f docker-compose.yml --project-directory . up  traefik whoami  --build  --remove-orphans
```
