#!/usr/bin/env bash

# Usage: source cluster_utils.sh && prompt_for_cluster

function ensure_yq_installed() {
  if ! command -v yq >/dev/null 2>&1; then
    echo "🔧 'yq' not found. Attempting to install..."

    if command -v apt >/dev/null 2>&1; then
      sudo apt update && sudo apt install -y yq
    elif command -v brew >/dev/null 2>&1; then
      brew install yq
    else
      echo "❌ Cannot auto-install yq. Please install manually: https://github.com/mikefarah/yq"
      exit 1
    fi
  fi
}

function prompt_for_cluster() {
  local inventory_file="${1:-./inventories/hosts.yml}"
  ensure_yq_installed

  # Extract available clusters
  available_clusters=$(yq 'keys | .[]' "$inventory_file" | sed 's/^"\(.*\)"$/\1/')

  if [[ -z "$available_clusters" ]]; then
    echo "❌ No clusters found in inventory: $inventory_file"
    exit 1
  fi

  echo "📦 Available clusters:"
  echo "$available_clusters" | nl -w2 -s'. '

  read -rp "Enter the cluster name to use to: " CLUSTER_NAME
  if [[ -z "$CLUSTER_NAME" ]]; then
    echo "❌ Cluster name is required. Aborting."
    exit 1
  fi

  if ! echo "$available_clusters" | grep -qx "$CLUSTER_NAME"; then
    echo "❌ '$CLUSTER_NAME' is not a valid cluster in inventory. Aborting."
    exit 1
  fi

  export CLUSTER_NAME
  echo "🚀 Using cluster: $CLUSTER_NAME"
}
