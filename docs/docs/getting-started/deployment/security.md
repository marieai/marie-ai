---
sidebar_position: 6
---

# Security

Secure your Marie-AI deployment with authentication, authorization, network policies, and secrets management.

## Overview

Marie-AI security consists of multiple layers:

| Layer | Components | Purpose |
|-------|------------|---------|
| Authentication | API keys, Bearer tokens | Verify client identity |
| Authorization | Key types, permissions | Control access to resources |
| Network | TLS, Network policies | Encrypt and restrict traffic |
| Secrets | Kubernetes secrets, env vars | Protect sensitive data |
| Runtime | Pod security, RBAC | Restrict container capabilities |

```text
┌─────────────────────────────────────────────────────────────┐
│                    Security Layers                          │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Network Security (TLS)                  │   │
│  │  ┌───────────────────────────────────────────────┐  │   │
│  │  │           Authentication (API Keys)            │  │   │
│  │  │  ┌─────────────────────────────────────────┐  │  │   │
│  │  │  │        Authorization (Permissions)       │  │  │   │
│  │  │  │  ┌───────────────────────────────────┐  │  │  │   │
│  │  │  │  │    Application (Marie-AI Core)    │  │  │  │   │
│  │  │  │  └───────────────────────────────────┘  │  │  │   │
│  │  │  └─────────────────────────────────────────┘  │  │   │
│  │  └───────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Authentication

### API key authentication

Marie-AI uses API keys for authenticating requests to the Gateway.

#### Key format

API keys follow a specific format:

| Prefix | Type | Use Case |
|--------|------|----------|
| `mau_` | User-to-server | Client applications, SDKs |
| `mas_` | Server-to-server | Backend services, integrations |

Keys are 58 characters total: 4-character prefix + 54-character token.

**Example keys:**

```text
mas_0aPJ9Q9nUO1Ac1vJTfffXEXs9FyGLf9BzfYgZ_RaHm707wmbfHJNPQ
mau_Gcp_GvCMrVVgp-BwGKLyELE3BaKtpmCrlwdIB-VWWWXwpm3k1CwVIg
```

#### Generating keys

Generate new API keys using Python:

```python
from marie.auth.api_key_manager import KeyGenerator

# Generate server-to-server key
server_key = KeyGenerator.generate_key("mas_")
print(f"Server key: {server_key}")

# Generate user-to-server key
user_key = KeyGenerator.generate_key("mau_")
print(f"User key: {user_key}")
```

Or via command line:

```bash
python -c "from marie.auth.api_key_manager import KeyGenerator; print(KeyGenerator.generate_key('mas_'))"
```

#### Configuring API keys

Add keys to your Marie configuration:

```yaml
# config/marie.yml
auth:
  keys:
    - name: production-backend
      api_key: mas_0aPJ9Q9nUO1Ac1vJTfffXEXs9FyGLf9BzfYgZ_RaHm707wmbfHJNPQ
      enabled: true

    - name: web-application
      api_key: mau_Gcp_GvCMrVVgp-BwGKLyELE3BaKtpmCrlwdIB-VWWWXwpm3k1CwVIg
      enabled: true

    - name: disabled-key
      api_key: mas_XeuXeznfHd_n0qRqavWSu9EVD0OrcwnJwvl_NOz0ucBG5R3creEWmw
      enabled: false
```

### Using Bearer tokens

Include the API key in the `Authorization` header:

```bash
curl -X POST http://localhost:54322/api/v1/invoke \
  -H "Authorization: Bearer mas_0aPJ9Q9nUO1Ac1vJTfffXEXs9FyGLf9BzfYgZ_RaHm707wmbfHJNPQ" \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {
      "command": "extract",
      "asset_key": "s3://bucket/document.pdf"
    }
  }'
```

**Python client:**

```python
import requests

headers = {
    "Authorization": "Bearer mas_0aPJ9Q9nUO1Ac1vJTfffXEXs9FyGLf9BzfYgZ_RaHm707wmbfHJNPQ",
    "Content-Type": "application/json"
}

response = requests.post(
    "http://localhost:54322/api/v1/invoke",
    headers=headers,
    json={"parameters": {"command": "extract", "asset_key": "s3://bucket/doc.pdf"}}
)
```

### Authentication errors

| HTTP Code | Error | Cause |
|-----------|-------|-------|
| 401 | Invalid token | Key not found or disabled |
| 403 | Invalid scheme | Not using Bearer scheme |
| 403 | Invalid authorization | Missing or malformed header |

## Secrets management

### Kubernetes secrets

Store sensitive configuration in Kubernetes secrets:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: marie-secrets
type: Opaque
stringData:
  # Database credentials
  POSTGRES_PASSWORD: "your-secure-password"

  # API keys
  API_KEY_BACKEND: "mas_0aPJ9Q9nUO1Ac1vJTfffXEXs9FyGLf9BzfYgZ_RaHm707wmbfHJNPQ"

  # Storage credentials
  S3_ACCESS_KEY: "your-access-key"
  S3_SECRET_KEY: "your-secret-key"
```

Reference secrets in deployments:

```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: marie
          env:
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: marie-secrets
                  key: POSTGRES_PASSWORD
            - name: API_KEY
              valueFrom:
                secretKeyRef:
                  name: marie-secrets
                  key: API_KEY_BACKEND
```

### Helm secret configuration

Configure secrets through Helm values:

```yaml
# values.yaml
postgresql:
  auth:
    existingSecret: marie-db-secret
    existingSecretPasswordKey: password

storage:
  s3:
    existingSecret: marie-storage-secret
```

### External secret management

Integrate with external secret managers:

**AWS Secrets Manager with External Secrets Operator:**

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: marie-secrets
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: ClusterSecretStore
  target:
    name: marie-secrets
  data:
    - secretKey: POSTGRES_PASSWORD
      remoteRef:
        key: marie/production
        property: db_password
```

**HashiCorp Vault:**

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: marie-secrets
spec:
  secretStoreRef:
    name: vault-backend
    kind: ClusterSecretStore
  target:
    name: marie-secrets
  data:
    - secretKey: api_key
      remoteRef:
        key: secret/marie/api-keys
        property: backend
```

## Network security

### TLS configuration

Enable TLS for encrypted communication:

**Helm values:**

```yaml
security:
  tls:
    enabled: true
    secretName: marie-tls-cert
```

**Generate self-signed certificates (development only):**

```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key \
  -out tls.crt \
  -subj "/CN=marie.example.com"

kubectl create secret tls marie-tls-cert \
  --cert=tls.crt \
  --key=tls.key
```

**Let's Encrypt with cert-manager:**

```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: marie-cert
spec:
  secretName: marie-tls-cert
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
    - marie.example.com
```

### Network policies

Restrict pod-to-pod communication:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: marie-gateway-policy
spec:
  podSelector:
    matchLabels:
      app: marie-gateway
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Allow traffic from ingress controller
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8080
        - protocol: TCP
          port: 52000
  egress:
    # Allow traffic to executors
    - to:
        - podSelector:
            matchLabels:
              app: marie-executor
      ports:
        - protocol: TCP
          port: 52010
    # Allow traffic to PostgreSQL
    - to:
        - podSelector:
            matchLabels:
              app: postgresql
      ports:
        - protocol: TCP
          port: 5432
    # Allow DNS
    - to:
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53
```

**Executor network policy:**

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: marie-executor-policy
spec:
  podSelector:
    matchLabels:
      app: marie-executor
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Only allow traffic from gateway
    - from:
        - podSelector:
            matchLabels:
              app: marie-gateway
      ports:
        - protocol: TCP
          port: 52010
  egress:
    # Allow S3/storage access
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
      ports:
        - protocol: TCP
          port: 443
```

### Ingress security

Configure secure ingress:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: marie-ingress
  annotations:
    # Force HTTPS
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    # Rate limiting
    nginx.ingress.kubernetes.io/limit-rps: "100"
    nginx.ingress.kubernetes.io/limit-connections: "10"
    # Security headers
    nginx.ingress.kubernetes.io/configuration-snippet: |
      add_header X-Frame-Options "SAMEORIGIN" always;
      add_header X-Content-Type-Options "nosniff" always;
      add_header X-XSS-Protection "1; mode=block" always;
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - marie.example.com
      secretName: marie-tls-cert
  rules:
    - host: marie.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: marie-gateway
                port:
                  number: 8080
```

## Kubernetes RBAC

### ServiceAccount

Create a dedicated service account:

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: marie-sa
  annotations:
    # For AWS IRSA
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789:role/marie-role
```

### Role and RoleBinding

Grant minimal permissions:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: marie-role
rules:
  # Read secrets for configuration
  - apiGroups: [""]
    resources: ["secrets"]
    resourceNames: ["marie-secrets", "marie-tls-cert"]
    verbs: ["get"]
  # Read config maps
  - apiGroups: [""]
    resources: ["configmaps"]
    resourceNames: ["marie-config"]
    verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: marie-rolebinding
subjects:
  - kind: ServiceAccount
    name: marie-sa
roleRef:
  kind: Role
  name: marie-role
  apiGroup: rbac.authorization.k8s.io
```

### Helm configuration

```yaml
serviceAccount:
  create: true
  name: "marie-sa"
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789:role/marie-role
  automountServiceAccountToken: true
```

## Pod security

### Security context

Configure pod and container security:

```yaml
security:
  podSecurityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000

  securityContext:
    allowPrivilegeEscalation: false
    readOnlyRootFilesystem: true
    capabilities:
      drop:
        - ALL
```

### Pod Security Standards

Apply Pod Security Standards (PSS):

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: marie
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

### GPU workloads

GPU pods may require elevated permissions:

```yaml
securityContext:
  # Required for NVIDIA GPU
  privileged: false
  capabilities:
    drop:
      - ALL
    add:
      - SYS_ADMIN  # Only if required by GPU driver
```

## Audit logging

### Request logging

Enable request logging for audit trails:

```yaml
# config/marie.yml
logging:
  level: INFO
  format: json
  # Include request details
  log_requests: true
  # Redact sensitive fields
  redact_fields:
    - authorization
    - api_key
    - password
```

### Kubernetes audit

Configure Kubernetes audit policy:

```yaml
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
  - level: Metadata
    namespaces: ["marie"]
    resources:
      - group: ""
        resources: ["secrets"]
  - level: Request
    namespaces: ["marie"]
    resources:
      - group: ""
        resources: ["pods", "services"]
```

## Security checklist

### Development

- [ ] Use non-production API keys
- [ ] Enable TLS for external access
- [ ] Set resource limits on pods

### Staging

- [ ] Rotate API keys regularly
- [ ] Enable network policies
- [ ] Configure audit logging
- [ ] Test authentication flows

### Production

- [ ] Use external secret management
- [ ] Enable Pod Security Standards
- [ ] Configure RBAC with minimal permissions
- [ ] Enable TLS everywhere
- [ ] Set up network policies
- [ ] Configure rate limiting
- [ ] Enable audit logging
- [ ] Regular security scanning
- [ ] API key rotation schedule

## Best practices

1. **Rotate keys regularly**: Set up a key rotation schedule (e.g., every 90 days)

2. **Use separate keys per service**: Don't share API keys between different services

3. **Disable unused keys**: Mark keys as `enabled: false` instead of deleting for audit trails

4. **Monitor authentication failures**: Alert on repeated 401/403 errors

5. **Least privilege**: Grant minimum required permissions to service accounts

6. **Encrypt at rest**: Enable encryption for Kubernetes secrets and database

7. **Network segmentation**: Use network policies to restrict pod communication

## Next steps

- [Observability](./observability.md) - Monitor security events
- [Troubleshooting](./troubleshooting.md) - Debug authentication issues
- [Configuration](../job-management/configuration.md) - Service configuration
