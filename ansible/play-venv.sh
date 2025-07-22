#!/usr/bin/env bash
set -euo pipefail

source ./env.sh
source ./cluster_utils.sh

INVENTORY="./inventories/hosts.yml"

prompt_for_cluster "$INVENTORY"

ansible-playbook ./playbook/venv-node.yml \
  -i "$INVENTORY" \
  -u "$ANSIBLE_USER" \
  --become --become-user "$ANSIBLE_BECOME_USER" \
  -e "$ANSIBLE_PASSWORD_FILE" \
  --vault-password-file="$ANSIBLE_VAULT_PASSWORD_FILE" \
  --extra-vars "target_group=$CLUSTER_NAME"


