#!/bin/bash

ansible-playbook ./playbook/who.yml -i ./inventories/hosts.yml -u gpu-svc
