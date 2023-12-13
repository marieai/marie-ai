 #!/bin/bash
source ./env.sh

ansible-playbook ./playbook/restart-daemon.yml -i ./inventories/hosts.yml -u $ANSIBLE_USER --become --become-user $ANSIBLE_BECOME_USER \
 -e $ANSIBLE_PASSWORD_FILE  --vault-password-file=$ANSIBLE_VAULT_PASSWORD_FILE
