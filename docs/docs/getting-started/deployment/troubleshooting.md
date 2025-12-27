---
sidebar_position: 7
---

# Troubleshooting

Diagnose and resolve common issues with Marie-AI deployments.

## Quick diagnosis

### Check system health

Run these commands to quickly assess your deployment:

```bash
# Check pod status
kubectl get pods -l app.kubernetes.io/name=marie

# Check service endpoints
kubectl get endpoints

# View recent events
kubectl get events --sort-by='.lastTimestamp' | tail -20

# Check resource usage
kubectl top pods -l app.kubernetes.io/name=marie
```

### Health endpoints

```bash
# Gateway health
curl http://localhost:54322/health

# Gateway readiness
curl http://localhost:54322/ready

# Debug information
curl http://localhost:54322/api/debug

# Capacity status
curl http://localhost:54322/api/capacity
```

## Pod issues

### Pods not starting

**Symptoms:** Pods stuck in `Pending`, `ContainerCreating`, or `CrashLoopBackOff`

#### Check pod events

```bash
kubectl describe pod <pod-name>
```

#### Common causes and solutions

| Status | Cause | Solution |
|--------|-------|----------|
| `Pending` | Insufficient resources | Scale cluster or reduce requests |
| `Pending` | No matching nodes | Check node selectors and taints |
| `ImagePullBackOff` | Image not found | Verify image name and registry access |
| `CrashLoopBackOff` | Application error | Check container logs |
| `ContainerCreating` | Volume mount issues | Check PVC status |

#### Insufficient resources

```bash
# Check node resources
kubectl describe nodes | grep -A 5 "Allocated resources"

# Check pending pods
kubectl get pods --field-selector=status.phase=Pending
```

**Solutions:**
- Reduce resource requests in values
- Add more nodes to the cluster
- Enable cluster autoscaler

#### Image pull failures

```bash
# Check image pull errors
kubectl describe pod <pod-name> | grep -A 10 "Events"
```

**Solutions:**
- Verify image exists: `docker pull marieai/marie:3.0-cuda`
- Check imagePullSecrets for private registries
- Verify network access to registry

### CrashLoopBackOff

**Check logs:**

```bash
# Current logs
kubectl logs <pod-name>

# Previous crash logs
kubectl logs <pod-name> --previous

# Follow logs
kubectl logs -f <pod-name>
```

**Common causes:**

| Error | Cause | Solution |
|-------|-------|----------|
| `Connection refused` | Dependency not ready | Check PostgreSQL, etcd status |
| `Out of memory` | Memory limit too low | Increase memory limit |
| `Permission denied` | Security context | Check runAsUser, fsGroup |
| `Config not found` | Missing ConfigMap | Verify ConfigMap exists |

### GPU pods not scheduling

```bash
# Check GPU availability
kubectl describe nodes | grep nvidia.com/gpu

# Check GPU pod events
kubectl describe pod <gpu-pod-name>
```

**Solutions:**

1. Verify GPU node pool exists and is ready
2. Check NVIDIA device plugin is running:
   ```bash
   kubectl get pods -n kube-system | grep nvidia
   ```
3. Verify tolerations match node taints:
   ```yaml
   tolerations:
     - key: nvidia.com/gpu
       operator: Exists
       effect: NoSchedule
   ```

## Connection issues

### Cannot connect to Gateway

**Symptoms:** Connection refused, timeout

```bash
# Check service
kubectl get svc marie-gateway

# Check endpoints
kubectl get endpoints marie-gateway

# Test from within cluster
kubectl run debug --rm -it --image=curlimages/curl -- \
  curl http://marie-gateway:8080/health
```

**Solutions:**

1. Verify service selector matches pod labels
2. Check network policies aren't blocking traffic
3. Verify correct port configuration

### Database connection failures

**Symptoms:** `Connection refused` or timeout to PostgreSQL

```bash
# Check PostgreSQL pod
kubectl get pods -l app.kubernetes.io/name=postgresql

# Test database connection
kubectl run debug --rm -it --image=postgres:15 -- \
  psql -h marie-postgresql -U marie -d marie -c "SELECT 1"
```

**Solutions:**

1. Verify PostgreSQL pod is running
2. Check credentials in secrets
3. Verify network policy allows database access
4. Check PostgreSQL logs for authentication errors

### etcd connection failures

**Symptoms:** Service discovery not working

```bash
# Check etcd status
kubectl get pods -l app.kubernetes.io/name=etcd

# Test etcd connectivity
kubectl run debug --rm -it --image=bitnami/etcd -- \
  etcdctl --endpoints=http://marie-etcd:2379 endpoint health
```

## Job processing issues

### Jobs stuck in CREATED state

**Symptoms:** Jobs submitted but not processing

```bash
# Check job status
curl http://localhost:54322/api/jobs/CREATED

# Check scheduler debug info
curl http://localhost:54322/api/debug
```

**Possible causes:**

1. **No executor capacity:**
   ```bash
   curl http://localhost:54322/api/capacity
   ```
   Solution: Scale up executors or check executor health

2. **Scheduler not running:**
   Check gateway logs for scheduler errors

3. **Database lock issues:**
   ```sql
   -- Check for locks
   SELECT * FROM pg_locks WHERE relation = 'marie_scheduler.job'::regclass;
   ```

### Jobs failing repeatedly

```bash
# Check failed jobs
curl http://localhost:54322/api/jobs/FAILED

# Get specific job details
curl http://localhost:54322/api/jobs/<job-id>
```

**Investigate through logs:**

```bash
# Gateway logs for job submission
kubectl logs -l app=marie-gateway | grep <job-id>

# Executor logs for processing errors
kubectl logs -l app=marie-executor | grep <job-id>
```

**Common failure causes:**

| Error | Cause | Solution |
|-------|-------|----------|
| `Asset not found` | Invalid S3 path | Verify asset exists |
| `Timeout` | Processing took too long | Increase timeout or optimize |
| `Out of memory` | Document too large | Increase executor memory |
| `GPU error` | CUDA out of memory | Reduce batch size |

### DAG not progressing

**Symptoms:** DAG stuck, dependent jobs not starting

```sql
-- Check DAG status
SELECT dag_id, state, COUNT(*)
FROM marie_scheduler.job
WHERE dag_id = 'your-dag-id'
GROUP BY dag_id, state;

-- Check blocked jobs
SELECT id, name, state, dependencies
FROM marie_scheduler.job
WHERE dag_id = 'your-dag-id' AND state = 'created';
```

**Solutions:**

1. Check if dependency jobs completed successfully
2. Verify dependency job IDs are correct
3. Look for failed jobs in the DAG chain

## Performance issues

### Slow job processing

**Diagnosis:**

```bash
# Check throughput metrics
curl http://localhost:9090/api/v1/query?query=rate(marie_jobs_total[5m])

# Check P95 latency
curl http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,rate(marie_job_duration_seconds_bucket[5m]))
```

**Common causes and solutions:**

| Symptom | Cause | Solution |
|---------|-------|----------|
| Low throughput | Not enough executors | Scale up replicas |
| High latency | Large documents | Optimize processing or add memory |
| Increasing queue | Executor bottleneck | Add GPU executors |
| Sporadic slowdowns | Resource contention | Check node resources |

### High memory usage

```bash
# Check pod memory
kubectl top pods -l app.kubernetes.io/name=marie

# Check for OOM events
kubectl get events | grep OOM
```

**Solutions:**

1. Increase memory limits
2. Reduce concurrent jobs per executor
3. Implement batch processing for large documents

### Database performance

```sql
-- Check slow queries
SELECT query, calls, mean_time, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- Check table sizes
SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
FROM pg_stat_user_tables
ORDER BY pg_total_relation_size(relid) DESC;
```

**Solutions:**

1. Add indexes for frequently queried columns
2. Run VACUUM ANALYZE regularly
3. Archive old jobs to reduce table size

## Authentication issues

### 401 Unauthorized

**Symptoms:** API requests return 401

```bash
# Test authentication
curl -v -H "Authorization: Bearer your-api-key" \
  http://localhost:54322/api/jobs
```

**Causes and solutions:**

| Cause | Solution |
|-------|----------|
| Invalid key | Verify key format (58 chars, starts with `mau_` or `mas_`) |
| Key disabled | Check `enabled: true` in config |
| Key not loaded | Restart Gateway after config change |
| Wrong header format | Use `Bearer <key>` not just `<key>` |

### 403 Forbidden

**Symptoms:** API requests return 403

**Causes:**
- Missing Authorization header
- Wrong authentication scheme (not Bearer)
- Key validation error

## Storage issues

### S3 access errors

**Symptoms:** Cannot read/write documents

```bash
# Test S3 connectivity from pod
kubectl exec -it <pod-name> -- aws s3 ls s3://your-bucket/
```

**Solutions:**

1. Verify IAM credentials or service account
2. Check bucket policy allows access
3. Verify endpoint URL for custom S3 (MinIO)
4. Check network egress allows S3 access

### Volume mount failures

**Symptoms:** Pods stuck in `ContainerCreating`

```bash
# Check PVC status
kubectl get pvc

# Describe PVC for events
kubectl describe pvc <pvc-name>
```

**Solutions:**

1. Check storage class exists
2. Verify PV is available
3. Check node has access to storage backend

## Debugging tools

### Interactive debugging

```bash
# Shell into running pod
kubectl exec -it <pod-name> -- /bin/bash

# Run debug container
kubectl debug <pod-name> -it --image=busybox
```

### Network debugging

```bash
# DNS resolution
kubectl run debug --rm -it --image=busybox -- nslookup marie-gateway

# TCP connectivity
kubectl run debug --rm -it --image=busybox -- nc -zv marie-postgresql 5432

# HTTP request
kubectl run debug --rm -it --image=curlimages/curl -- \
  curl -v http://marie-gateway:8080/health
```

### Log aggregation

```bash
# All Marie-AI logs
kubectl logs -l app.kubernetes.io/name=marie --all-containers

# Stream logs from all pods
kubectl logs -f -l app.kubernetes.io/name=marie --all-containers --max-log-requests=10
```

### Database queries

```bash
# Connect to PostgreSQL
kubectl exec -it <postgresql-pod> -- psql -U marie -d marie

# Quick status check
kubectl exec -it <postgresql-pod> -- psql -U marie -d marie -c "
  SELECT state, COUNT(*) FROM marie_scheduler.job GROUP BY state;
"
```

## Recovery procedures

### Reset stuck jobs

```sql
-- Reset jobs stuck in ACTIVE state (timeout recovery)
UPDATE marie_scheduler.job
SET state = 'retry', started_on = NULL
WHERE state = 'active'
  AND started_on < NOW() - INTERVAL '1 hour';
```

### Clear job queue

```sql
-- Archive and clear old completed jobs
INSERT INTO marie_scheduler.archive
SELECT * FROM marie_scheduler.job
WHERE state IN ('completed', 'failed')
  AND completed_on < NOW() - INTERVAL '7 days';

DELETE FROM marie_scheduler.job
WHERE state IN ('completed', 'failed')
  AND completed_on < NOW() - INTERVAL '7 days';
```

### Restart components

```bash
# Rolling restart of Gateway
kubectl rollout restart deployment marie-gateway

# Rolling restart of Executors
kubectl rollout restart deployment marie-executor

# Force pod recreation
kubectl delete pod -l app=marie-gateway
```

### Database recovery

```bash
# Backup database
kubectl exec <postgresql-pod> -- pg_dump -U marie marie > backup.sql

# Restore database
kubectl exec -i <postgresql-pod> -- psql -U marie marie < backup.sql
```

## Getting help

### Gather diagnostic information

When requesting support, collect:

```bash
# Pod status and events
kubectl get pods -o wide
kubectl describe pods -l app.kubernetes.io/name=marie > pod-describe.txt

# Logs
kubectl logs -l app.kubernetes.io/name=marie --all-containers > logs.txt

# Resource usage
kubectl top pods > resources.txt

# Configuration (redact secrets)
helm get values marie > helm-values.txt
```

### Support resources

- GitHub Issues: https://github.com/marieai/marie-ai/issues
- Documentation: Check related guides for specific features

## Next steps

- [Observability](./observability.md) - Set up monitoring and alerting
- [Scaling](./scaling.md) - Handle capacity issues
- [Security](./security.md) - Debug authentication problems
