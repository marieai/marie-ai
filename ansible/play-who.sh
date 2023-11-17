#!/bin/bash
source ./env.sh

ansible-playbook  ./playbook/who.yml -i ./inventories/hosts.yml -u $ANSIBLE_USER