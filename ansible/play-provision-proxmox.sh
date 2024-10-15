#!/bin/bash
source ./env.sh
# this will allow us to self login to the local machine and run the playbook
ansible-playbook -vvv ./playbook/provision-proxmox.yml -i ./inventories/proxmox.yml  -i localhost, -c local
