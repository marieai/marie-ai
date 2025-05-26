# Service Discovery

Service discovery is a mechanism that allows services to find and communicate with each other. 
Here we are going to show how to use the `EtcdServiceResolver` to resolve services from etcd.

## Install etcd

```bash
docker run  -d  -p 2379:2379  --name etcd \
-v /usr/share/ca-certificates/:/etc/ssl/certs \
quay.io/coreos/etcd:v3.5.14 /usr/local/bin/etcd -advertise-client-urls \
http://0.0.0.0:2379 -listen-client-urls http://0.0.0.0:2379 \
--log-level=info \
--log-outputs=stdout
```

Verify the installation by running the following command:

```bash  
docker exec -it etcd etcdctl version
```

```bash
docker exec -it etcd etcdctl put 'hello' 'value-1'
docker exec -it etcd etcdctl put 'world' 'value-2'
 
docker exec -it etcd etcdctl get "" --prefix=true
docker exec -it etcd etcdctl get "" --from-key
```


## Purge the etcd data

```bash
docker exec -it etcd etcdctl del "" --from-key=true
```
  

## Install the `etcd3` package

Install the `etcd3` package from source code  as the `GRPC` version or `marie` is not compatible with the current version of the `etcd3` package.

```bash
git clone  git@github.com:kragniz/python-etcd3.git
cd python-etcd3
python setup.py install
pip show etcd3
```

# Test the registry and resolver

Start the resolver and the registry services.

```bash
python ./marie/serve/discovery/resolver.py --port 2379 --host 0.0.0.0 --service-key gateway/service_test
```

```bash
python ./registry.py --port 2379 --host 0.0.0.0 --service-key gateway/service_test --service-addr 127.0.0.1:5001 --my-id service001
```


