### Connecting to MarieAI

To start new instance of MarieAI server, use the following command:

```bash
server --start --uses /mnt/data/marie-ai/config/service/marie-dev.yml
```

To start new job on MarieAI server, use the following command:

```bash
job --start --uses /mnt/data/marie-ai/config/service/marie-dev.yml
```


### Connecting to specific etcd server instance via MarieAI

```bash
marie server watch --etcd-host 127.0.0.1 --etcd-port 2379
```


Loading help screen...

```bash
marie --help
```
