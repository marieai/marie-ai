#!/bin/bash

ansible-playbook -vvv ./playbook/info.yml -i ./inventories/hosts.yml -u gpu-svc --become --become-user gpu-svc -e '@password.yml' --vault-password-file=vault.txt

