# Marie-AI System Bootstrap Script Documentation

This document describes the functionality, options, and usage of the `bootstrap.sh` script to deploy the Marie-AI system.

---

## 🎯 Purpose

The script automates:

* Validation of Docker environment and compose files
* Detection of running containers and clean-up options
* Deployment of infrastructure services (RabbitMQ, MinIO, ETCD, etc.)
* Deployment of Marie services (Gateway and Extract Executors)
* Display of service endpoints and status

---

## 🛠 Requirements

* **Docker** and **Docker Compose** installed

* `.env.dev` environment file:

  ```bash
  ./config/.env.dev
  ```

* Compose files in `./Dockerfiles/`:

  | File                          | Description                  |
  | ----------------------------- | ---------------------------- |
  | docker-compose.yml            | Base infrastructure          |
  | docker-compose.storage.yml    | Storage backend              |
  | docker-compose.monitoring.yml | Monitoring stack             |
  | docker-compose.s3.yml         | MinIO S3 storage             |
  | docker-compose.rabbitmq.yml   | RabbitMQ                     |
  | docker-compose.etcd.yml       | ETCD cluster                 |
  | docker-compose.gateway.yml    | Gateway API (optional)       |
  | docker-compose.extract.yml    | Extract Executors (optional) |

---

## 🚀 Usage

Run the script:

```bash
./bootstrap.sh [options]
```

### Options

| Option                  | Description                              |
| ----------------------- | ---------------------------------------- |
| `--no-gateway`          | Skip Gateway deployment                  |
| `--no-extract`          | Skip Extract Executor deployment         |
| `--no-infrastructure`   | Skip infrastructure services             |
| `--infrastructure-only` | Deploy only infrastructure               |
| `--services-only`       | Deploy only Gateway and Extract services |
| `-h`, `--help`          | Show help                                |

### Examples

* **Deploy everything**:

  ```bash
  ./bootstrap.sh
  ```

* **Deploy only infrastructure**:

  ```bash
  ./bootstrap.sh --infrastructure-only
  ```

* **Deploy only services**:

  ```bash
  ./bootstrap.sh --services-only
  ```

* **Deploy infrastructure + gateway only**:

  ```bash
  ./bootstrap.sh --no-extract
  ```

---

## ⚠️ Cleanup Process

If running services are detected, you will be prompted to:

1. Stop and remove all containers (recommended)
2. Stop Compose services only
3. Continue without cleanup (may cause conflicts)
4. Exit

---

## 📊 Service Endpoints

| Service              | URL                                                              |
| -------------------- | ---------------------------------------------------------------- |
| RabbitMQ Management  | [http://localhost:15672](http://localhost:15672) (`guest/guest`) |
| MinIO Console        | [http://localhost:8001](http://localhost:8001)                   |
| Monitoring (Grafana) | [http://localhost:3000](http://localhost:3000)                   |
| HTTP Gateway         | [http://localhost:52000](http://localhost:52000)                 |
| GRPC Gateway         | grpc://localhost:51000                                           |
| Extract Executor     | [http://localhost:8080](http://localhost:8080)                   |

> **Note:** Services depend on deployment options.

---

## 🛠 Useful Docker Commands

* **View logs:**

  ```bash
  docker compose logs -f [service_name]
  ```

* **Stop services:**

  ```bash
  docker compose down
  ```

* **Stop and remove volumes:**

  ```bash
  docker compose down --volumes --remove-orphans
  ```

* **Scale executors:**

  ```bash
  docker compose up -d --scale marie-extract-executor=3
  ```

---

## Workflow Summary

1. Validate environment and compose files
2. Detect and clean up running containers if needed
3. Build Docker Compose command
4. Start services
5. Display status and endpoints

---

## Support

* Ensure Docker is installed and running
* Verify `.env.dev` exists
* Check logs if deployment fails

---

End of documentation
