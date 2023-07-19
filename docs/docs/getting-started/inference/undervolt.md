---
sidebar_position: 3
---

# Undervolt NVIDIA GPUs

List all NVIDIA GPUs:

```shell    
nvidia-smi -L
```

List all available clocks:

```shell
nvidia-smi -q -d SUPPORTED_CLOCKS
```

To review the current GPU clock speed, default clock speed, and maximum possible clock speed, run:

```shell
nvidia-smi -q -d CLOCK
```

## Stress test GPU

Start monitoring GPU usage:
```shell
nvidia-smi --query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr --format=csv -l 1
```

Start stress test:

```shell
docker run --rm --gpus all gpu_burn
```


## Enable persistence mode

```shell
sudo nvidia-smi -pm 1
```

```text
Enabled persistence mode for GPU 00000000:03:00.0.
Enabled persistence mode for GPU 00000000:04:00.0.
```


## Adjust GPU power limit

```shell
sudo nvidia-smi -pl 350
```

```text
Power limit for GPU 00000000:03:00.0 was set to 200.00 W from 450.00 W.
Power limit for GPU 00000000:04:00.0 was set to 200.00 W from 450.00 W.
```

## Adjust GPU clock speed
4
```shell
nvidia-smi -i 0 -lgc 0,<base_clock>

sudo nvidia-smi -lgc 0,2100
```

```text
GPU clocks set to "(gpuClkMin 0, gpuClkMax 1860)" for GPU 00000000:03:00.0
GPU clocks set to "(gpuClkMin 0, gpuClkMax 1860)" for GPU 00000000:04:00.0
GPU clocks set to "(gpuClkMin 0, gpuClkMax 1860)" for GPU 00000000:05:00.0
GPU clocks set to "(gpuClkMin 0, gpuClkMax 1860)" for GPU 00000000:06:00.0

```


## References
https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries
https://www.microway.com/hpc-tech-tips/nvidia-smi_control-your-gpus/
https://linustechtips.com/topic/1259546-how-to-undervolt-nvidia-gpus-in-linux/

https://github.com/wilicc/gpu-burn
https://github.com/waggle-sensor/gpu-stress-test


 Device 0 [NVIDIA GeForce RTX 3090 Ti] PCIe GEN 3@ 8x RX: 19.53 MiB/s TX: 8.789 MiB/s Device 1 [NVIDIA GeForce RTX 3090 Ti] PCIe GEN 3@ 8x RX: 28.32 MiB/s TX: 8.789 MiB/s
 GPU 1815MHz MEM 10251MH TEMP  58°C FAN  69% POW 447 / 450 W                          GPU 1740MHz MEM 10251MH TEMP  57°C FAN  59% POW 446 / 450 W
 GPU[||||||||||||||||||||||||||||||||100%] MEM[||||||||||||||||||||22.158Gi/23.988Gi] GPU[||||||||||||||||||||||||||||||||100%] MEM[||||||||||||||||||||22.226Gi/23.988Gi]

 Device 2 [NVIDIA GeForce RTX 3090 Ti] PCIe GEN 3@ 8x RX: 27.34 MiB/s TX: 8.789 MiB/s Device 3 [NVIDIA GeForce RTX 3090 Ti] PCIe GEN 3@ 8x RX: 22.46 MiB/s TX: 6.836 MiB/s
 GPU 1830MHz MEM 10251MH TEMP  59°C FAN  71% POW 447 / 450 W                          GPU 1770MHz MEM 10251MH TEMP  58°C FAN  66% POW 448 / 450 W
 GPU[||||||||||||||||||||||||||||||||100%] MEM[||||||||||||||||||||22.260Gi/23.988Gi] GPU[||||||||||||||||||||||||||||||||100%] MEM[||||||||||||||||||||21.918Gi/23.988Gi]
