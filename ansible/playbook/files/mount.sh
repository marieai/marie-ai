#!/usr/bin/env bash
echo 'Mount setup'

CAN_I_RUN_SUDO=$(sudo -n uptime 2>&1|grep "load"|wc -l)
if [ ${CAN_I_RUN_SUDO} -eq 0 ]
then
    echo "I can't run the SUDO command"
    exit 2
fi

echo "Setting up cifs/nfs mounts"
if ! dpkg -l cifs-utils > /dev/null
then
    echo "cifs-utils not installed. Installing..."
    sudo apt-get install -qy nfs-common cifs-utils
fi

# check if entry added
if grep "/mnt/data/marie-ai" /etc/fstab
then
    echo "Mount already defined"
else
    echo "Mount not defined. Adding..."

sudo -n uptime
mkdir -p /mnt/data/marie-ai
cat >>/etc/fstab <<'EOF'
# MARIE_AI_START
127.0.0.1:/mnt/shares/data  /mnt/data/marie-ai       nfs defaults,nfsvers=3 0 0
# MARIE_AI_END
EOF

fi

mount /mnt/data/marie-ai

# single quote prevents parameter expansion for
if mountpoint -q /mnt/data/marie-ai/; then
    echo "Destination reachable."
    exit 0
else
    echo "Destination unreachable. Exiting."
    exit 2
fi
