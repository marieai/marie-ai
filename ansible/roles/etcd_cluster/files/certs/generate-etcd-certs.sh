#!/usr/bin/env bash
set -euo pipefail

#
# Usage:
#   ./generate-etcd-certs.sh OUTPUT_DIR NODE1 NODE2 NODE3 …
#
OUTDIR="$1"; shift
NODES=( "$@" )

mkdir -p "${OUTDIR}"
cd "${OUTDIR}"

echo "## 1) Generate CA key + cert"
openssl genrsa -out ca.key 4096
openssl req -x509 -new -nodes -key ca.key \
  -subj "/CN=etcd-ca" -days 3650 -out ca.pem

echo "## 2) Generate shared server key + CSR"
openssl genrsa -out server.key 4096
openssl req -new -key server.key \
  -subj "/CN=etcd-server" -out server.csr

echo "## 3) Build SAN config for all nodes"
cat > openssl-san.cnf <<EOF
[ req ]
distinguished_name = req_distinguished_name
req_extensions     = v3_req

[ req_distinguished_name ]

[ v3_req ]
basicConstraints = CA:FALSE
keyUsage         = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth, clientAuth
subjectAltName   = @alt_names

[ alt_names ]
EOF

# append each node as DNS entry
for i in "${!NODES[@]}"; do
  idx=$((i+1))
  echo "DNS.${idx} = ${NODES[i]}" >> openssl-san.cnf
done

echo "## 4) Sign the shared cert"
openssl x509 -req \
  -in server.csr \
  -CA ca.pem -CAkey ca.key -CAcreateserial \
  -out server.pem \
  -days 3650 \
  -extfile openssl-san.cnf -extensions v3_req

echo "## 5) Duplicate for peer use"
cp server.key peer.key
cp server.pem peer.pem

echo "✔ Certificates generated in ${OUTDIR}:"
ls -1 "${OUTDIR}"
