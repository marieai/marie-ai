# PostgreSQLStorage

**PostgreSQLStorage** is Indexer wrapper around the PostgreSQL DBMS. Postgres is an open source object-relational database. You can read more about it here: https://www.postgresql.org/




## Prerequisites


Additionally, you will need a running PostgreSQL database. This can be a local instance, a Docker image, or a virtual machine in the cloud. Make sure you have the credentials and connection parameters. 

You can start one in a Docker container, like so: 

```bash
docker run -e POSTGRES_PASSWORD=123456  -p 127.0.0.1:5432:5432/tcp postgres:14.4 
```

📕 **Note on docker network for macOS users**:  
If you run both the database and the `PostgresSQLStorage` docker container on the same machine 
localhost in the `PostgresSQLStorage` resolves to a separate network created by Docker which cannot see the database running on the host network.  
Use `host.docker.internal` to access localhost on the host machine. You can pass this parameter 
to the `PostgresSQLStorage` storage by using `uses_with={'hostname': 'host.docker.internal''}` when
calling the `flow.add(...)` function.


This indexer assumes a PRIMARY KEY on the `id` field, thus you cannot add two `Document` of the same id. Make sure you clean up any existing data if you want to start fresh.

## Reference

- https://www.postgresql.org/
- https://github.com/jina-ai/executors/tree/main/jinahub/indexers/storage/PostgreSQLStorage