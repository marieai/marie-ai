# Docker Container Stresser

A Python script to stress-test Docker containers by randomly or periodically restarting, stopping, or killing them.\\
This tool helps you verify application resilience and reconnection behavior.

---

## 📋 Features

- Random stress testing with configurable intervals and duration.
- Periodic stress testing with fixed intervals and action count.
- Burst stress testing (rapid restart sequences with rest periods).
- Chaos testing with completely random actions and timing.
- Logs to console and a `container_stresser.log` file.
- Graceful shutdown via `SIGINT` and `SIGTERM`.
- Prints detailed test statistics on exit.

---

## 🛠️ Requirements

- Python 3.6+
- Docker CLI installed and accessible to the user running the script

---

## 🚀 Usage

Run the script using Python:

```bash
python3 container_stresser.py --container <container_name> [OPTIONS]
```

---

## 🔧 Options

### Required:

- `--container`: **(Required)** Docker container name.

---

### Modes:

Choose the stress mode:

- `--mode random` (default)
- `--mode periodic`
- `--mode burst`
- `--mode chaos`

---

### Random Mode:

Random restarts/stops/kills with random intervals.

| Option           | Default | Description                                |
| ---------------- | ------- | ------------------------------------------ |
| `--duration`     | 10      | Duration in minutes                        |
| `--min-interval` | 5       | Minimum interval between actions (seconds) |
| `--max-interval` | 30      | Maximum interval between actions (seconds) |

Example:

```bash
python3 container_stresser.py --container my_app --mode random --duration 20 --min-interval 10 --max-interval 60
```

---

### Periodic Mode:

Fixed number of actions at regular intervals.

| Option       | Default | Description                               |
| ------------ | ------- | ----------------------------------------- |
| `--count`    | 10      | Number of actions                         |
| `--interval` | 15      | Interval between actions (seconds)        |
| `--action`   | restart | Action type: `restart`, `stop`, or `kill` |

Example:

```bash
python3 container_stresser.py --container my_app --mode periodic --count 5 --interval 20 --action stop
```

---

### Burst Mode:

Rapid sequences of restarts followed by rest periods.

| Option             | Default | Description                                    |
| ------------------ | ------- | ---------------------------------------------- |
| `--burst-count`    | 5       | Restarts per burst                             |
| `--burst-interval` | 2       | Interval between restarts in a burst (seconds) |
| `--rest-period`    | 30      | Rest period between bursts (seconds)           |

Example:

```bash
python3 container_stresser.py --container my_app --mode burst --burst-count 3 --burst-interval 5 --rest-period 60
```

---

### Chaos Mode:

Completely random actions and timings.

| Option       | Default | Description         |
| ------------ | ------- | ------------------- |
| `--duration` | 15      | Duration in minutes |

Example:

```bash
python3 container_stresser.py --container my_app --mode chaos --duration 30
```

---

### General Options:

- `--verbose` or `-v`: Enable debug logging.

---

## 📈 Output

- Logs are written to `container_stresser.log`.
- On exit, statistics are printed, including:
  - Total starts, stops, restarts, failures.
  - Duration.
  - Current container status.

---

## 🛑 Graceful Shutdown

Press `Ctrl+C` to stop the stress test gracefully. The script ensures:

- Final stats are printed.
- The container is restarted before exiting if it was stopped.

---

## 🧩 Example Commands

Random mode for 5 minutes:

```bash
python3 container_stresser.py --container test_container --mode random --duration 5
```

Periodic mode with 3 kills every 10 seconds:

```bash
python3 container_stresser.py --container test_container --mode periodic --count 3 --interval 10 --action kill
```

Burst mode with 4 restarts per burst:

```bash
python3 container_stresser.py --container test_container --mode burst --burst-count 4 --burst-interval 3 --rest-period 20
```

Chaos mode for 10 minutes:

```bash
python3 container_stresser.py --container test_container --mode chaos --duration 10
```


---