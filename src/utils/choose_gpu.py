import subprocess
from io import BytesIO
import pandas as pd
import torch


def get_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.free"])
    gpu_df = pd.read_csv(BytesIO(gpu_stats),
                         names=['memory.free'],
                         skiprows=1)
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: int(x.rstrip(' [MiB]')))
    return gpu_df['memory.free'].idxmax()


def create_device(gpu_number):
    if torch.cuda.is_available():
        if gpu_number is None:
            free_gpu_id = get_free_gpu()
        else:
            free_gpu_id = gpu_number
        device = torch.device(f"cuda:{free_gpu_id}")
        pin_memory = False
    else:
        device = torch.device("cpu")
        pin_memory = False
    return device, pin_memory