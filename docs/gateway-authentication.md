# Configure gateway authentication

Marie gateway startup requires at least one enabled API key. Missing keys, malformed keys and the published example keys in `APIKeyManager` documentation cause startup to fail with a redacted `auth_boot_*` error. Authentication is loaded before HTTP/gRPC listener setup.

Provide the existing authentication block in your gateway configuration:

```yaml
auth:
  keys:
    - name: gateway
      api_key: ${{ ENV.MARIE_GATEWAY_API_KEY }}
      enabled: true
```

Generate a key once and provision it through your deployment's existing secret delivery:

```bash
python -c "import secrets; print('mas_' + secrets.token_urlsafe(40))"
```

Keep the generated value out of source control and logs. Use the same provisioned credential in authorized clients. Restart the gateway after correcting its configuration.

Local development also requires a generated key. This first step adds no authentication bypass, automatic key generation at startup or credential rotation. Existing scopes and fabric restrictions still apply; startup validation does not replace request authorization.
