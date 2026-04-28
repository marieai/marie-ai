# Sample usage

```shell
./create_jobs.sh grpc://127.0.0.1:52000 1 mau_t6qDi1BcL1NkLI8I6iM8z1va0nZP01UQ6LWecpbDz6mbxWgIIIZPfQ
```

Use the reusable mock planner metadata template:

```shell
./create_jobs.sh grpc://127.0.0.1:52000 100 YOUR_API_KEY ./mock_parallel_subgraphs.metadata.json
```

The metadata template supports per-job placeholders:

- `{{request_id}}` -> `job-1`, `job-2`, ...
- `{{job_index}}` -> `1`, `2`, ...
- `{{timestamp}}`
- `{{random}}`
