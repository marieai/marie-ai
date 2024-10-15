
Add local `id_rsa.pub` to the remote machine's `authorized_keys` file.

```bash
ssh-copy-id -i ~/.ssh/id_rsa.pub gpu-svc@<remote-ip>
```


## gpu-svc user setup on the remote machine

This will create the `gpu-svc` user, add it to the sudo group, configure passwordless sudo, and set up SSH keys for passwordless login.

```bash
sudo adduser gpu-svc && \
sudo usermod -aG sudo gpu-svc && \
echo 'gpu-svc ALL=(ALL) NOPASSWD:ALL' | sudo tee /etc/sudoers.d/gpu-svc && \
sudo mkdir -p /home/gpu-svc/.ssh && \
sudo chmod 700 /home/gpu-svc/.ssh && \
sudo ssh-keygen -t rsa -b 2048 -f /home/gpu-svc/.ssh/id_rsa && \
sudo cp /home/gpu-svc/.ssh/id_rsa.pub /home/gpu-svc/.ssh/authorized_keys && \
sudo chmod 600 /home/gpu-svc/.ssh/authorized_keys && \
sudo chown -R gpu-svc:gpu-svc /home/gpu-svc/.ssh
```

We need to copy the private key to the local machine to use it for passwordless login and manage the remote machine.

```bash
scp gpu-svc@<remote-ip>:/home/gpu-svc/.ssh/id_rsa ~/keys/gpu-cluster/gpu-svc.pem
```

## Verify passwordless sudo

```bash
ssh gpu-svc@<remote-ip> 
ssh gpu-svc@192.168.1.28 -i ~/keys/gpu-cluster/gpu-svc.pem sudo ls /
```


## Reference
https://www.digitalocean.com/community/tutorials/how-to-use-vault-to-protect-sensitive-ansible-data
https://www.shellhacks.com/ansible-sudo-a-password-is-required/
https://github.com/priximmo
https://github.com/Pro-Tweaker/SEEDbox

