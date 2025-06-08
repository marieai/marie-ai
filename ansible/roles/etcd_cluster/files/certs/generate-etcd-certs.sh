#!/bin/bash
set -euo pipefail

CERT_DIR="$(dirname "$0")"
DAYS_VALID=3650
COMMON_NAME="etcd-cluster"

echo ">> Generating shared TLS certs for ETCD cluster in: $CERT_DIR"

mkdir -p "$CERT_DIR"
cd "$CERT_DIR"

# Step 1: Generate CA key and certificate
openssl genrsa -out ca-key.pem 4096
openssl req -x509 -new -nodes -key ca-key.pem -subj "/CN=etcd-ca" \
  -days $DAYS_VALID -out ca.pem

# Step 2: Generate shared ETCD server key and CSR
openssl genrsa -out etcd-key.pem 4096
openssl req -new -key etcd-key.pem -subj "/CN=${COMMON_NAME}" -out etcd.csr

# Step 3: Create extfile with SANs (optional)
cat > etcd.ext <<EOF
basicConstraints = CA:FALSE
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
IP.1 = 127.0.0.1
# Add cluster IPs if you want stronger SAN validation
EOF

# Step 4: Sign CSR with CA
openssl x509 -req -in etcd.csr -CA ca.pem -CAkey ca-key.pem -CAcreateserial \
  -out etcd.pem -days $DAYS_VALID -extfile etcd.ext

# Cleanup
rm -f etcd.csr etcd.ext ca.srl

echo ">> Done. Files generated:"
ls -l ca.pem etcd.pem etcd-key.pem
