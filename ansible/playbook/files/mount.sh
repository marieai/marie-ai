#!/usr/bin/env bash
echo 'Mount setup'

CAN_I_RUN_SUDO=$(sudo -n uptime 2>&1|grep "load"|wc -l)
if [ ${CAN_I_RUN_SUDO} -eq 0 ]
then
    echo "I can't run the SUDO command"
    exit 2
fi

sudo -n uptime
mkdir -p /mnt/data/marie-ai
 the @
cat >>/etc/fstab <<'EOF'
# MARIE_AI_START
127.0.0.1:/mnt/shares/data  /mnt/data/marie-ai       nfs defaults,nfsvers=3 0 0
# MARIE_AI_END
EOF

mount /mnt/data/marie-ai

# check if mounted:
# check if mounted
if mountpoint -q /mnt/data/marie-ai/; then
    echo "Destination reachable."
    exit 0
fi
# check if entry added
if grep "/mnt/data/marie-ai" /etc/fstab
then
    echo "Mount already defined"
    exit 1
fi

echo "Setting up cifs/nfs mounts"
sudo apt-get install -qy nfs-common cifs-utils

# single quote prevents parameter expansion for
if mountpoint -q /mnt/data/marie-ai/; then
    echo "Destination reachable."
    exit 0
else
    echo "Destination unreachable. Exiting."
    exit 2
fi
