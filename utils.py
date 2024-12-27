import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
import numpy as np
import subprocess
import torch

def _find_free_gpus(threshold=10):
    try:
        # assert(torch.cuda.device_count() == 8)
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )

        utilizations = [int(x.strip()) for x in result.stdout.split('\n') if x.strip()]
        print(utilizations)

        free_gpus = [i for i, util in enumerate(utilizations) if util < 10]
        return free_gpus

    except Exception as e:
        print(f"something went wrong getting free gpus: {e}")


def get_free_gpus(ngpu):
    free_gpus = _find_free_gpus()
    if free_gpus:
        print(f"Available GPUs are: {free_gpus}")
        selected_gpus = free_gpus[-ngpu:]
        print(selected_gpus)

        # for gpu_idx in selected_gpus:
        #     print(torch.cuda.device(gpu_idx))
        #     print(torch.cuda.get_device_properties(gpu_idx))
    
    return selected_gpus

    # device = torch.device(f"cuda:{selected_gpus[0]}" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
