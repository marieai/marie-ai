#!/usr/bin/env bash
set -euo pipefail

# Load from .env if present
[[ -f ".env" ]] && source .env

# Prompt if not set
if [[ -z "${SSHPASS:-}" ]]; then
  read -s -p "Enter SSH password: " SSHPASS
  echo
fi

# List of target hosts
servers=(
  mariectl-001
  mariectl-002
  mariectl-003
  mariectl-004
  mariectl-005
  mariectl-006
  mariectl-007
  mariectl-008
)

PUBKEY=~/.ssh/id_rsa.pub
[[ -f "$PUBKEY" ]] || { echo "❌ Public key not found: $PUBKEY"; exit 1; }

for host in "${servers[@]}"; do
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🔍 Checking: $host"

  # Step 1: Check if key already works
  if ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" 'exit' 2>/dev/null; then
    echo "✅ SSH key already works on $host. Skipping."
    continue
  fi

  # Step 2: Check if password authentication works
  echo "🔐 Verifying password for $host..."
  if ! SSHPASS="$SSHPASS" sshpass -e ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$host" 'echo "Password OK"' >/dev/null 2>&1; then
    echo "❌ ERROR: Password authentication failed for $host"
    echo "🛑 Aborting. Please check the password or reachability."
    exit 1
  fi

  # Step 3: Install the SSH key
  echo "→ Installing SSH key to $host..."
  if SSHPASS="$SSHPASS" sshpass -e ssh-copy-id -i "$PUBKEY" "$host"; then
    # Step 4: Confirm the key works
    if ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" 'echo "✅ Verified key-based login."' 2>/dev/null; then
      echo "✅ SSH key successfully deployed and verified on $host"
    else
      echo "❌ SSH key install may have failed — cannot verify login for $host"
      exit 1
    fi
  else
    echo "❌ ssh-copy-id failed on $host"
    exit 1
  fi

  echo
done

unset SSHPASS
