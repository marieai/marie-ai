---
sidebar_position: 1
---

# Basic config

Service configuration is done via configuring `marie.yml` file.


Shown below is an example `marie.yml` file:
```yaml
RegistryEnabled: True
DebugWebserver: False
WatchdogInterval: 60

ConsulEndpoint:
  Host: 10.0.13.70
  Port: 8500
  Scheme: http

ServiceEndpoint: 
  Port: 5100
  Host: 

executors:
- uses: NerExtractionExecutor
  _name_or_path: "provider/corr-ner"

# Storage where the document will be stored
storage:
- provider: postgresql
  hostname: 127.0.0.1
  port: 5432
  username: postgres
  password: 123456
  database: postgres
  table: default_table
```
