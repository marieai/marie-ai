#!/bin/bash

export ANSIBLE_VAULT_PASSWORD_FILE=./vault.txt
ansible-playbook -vvv ./playbook/start-node.yml -i ./inventories/hosts-single.yml -u gpu-svc --become --become-user gpu-svc \
 -e '@password.yml'  --vault-password-file=vault.txt