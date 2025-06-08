#!/bin/bash
source ./env.sh

if [ $# -eq 0 ]; then
    echo "No arguments provided"
    exit 1
fi

grep_param=$1
echo "Grep Param: $grep_param"

ansible-playbook  -f 10 ./playbook/grep-node.yml -i ./inventories/hosts.yml -u $ANSIBLE_USER --become --become-user $ANSIBLE_BECOME_USER \
 -e $ANSIBLE_PASSWORD_FILE  --vault-password-file=$ANSIBLE_VAULT_PASSWORD_FILE  -e "grep_param=${grep_param}"

