#!/bin/bash

ansible-playbook ./playbook/validate.yml -i ./inventories/hosts.yml -u gpu-svc
