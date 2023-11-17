#!/bin/bash

# .env loading in the shell raise an error if the file does not exist

if [ ! -f .env ]; then
    echo "ERROR: .env file not found. Please create it from .env.example file."
    exit 1
fi

set -a
[ -f .env ] && . .env
set +a

echo $ANSIBLE_USER
echo $ANSIBLE_BECOME_USER

echo $ANSIBLE_PASSWORD_FILE
echo $ANSIBLE_VAULT_PASSWORD_FILE
