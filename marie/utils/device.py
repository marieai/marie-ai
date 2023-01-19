import os
import subprocess


def gpu_device_count():
    """get gpu device count"""
    try:
        num_devices = str(subprocess.check_output(['nvidia-smi', '-L'])).count('UUID')
    except:
        num_devices = int(os.environ.get('CUDA_TOTAL_DEVICES', 0))
    return num_devices
