#!/bin/bash
source ./env.sh

ansible-playbook ./playbook/validate.yml -i ./inventories/hosts.yml -u $ANSIBLE_USER
