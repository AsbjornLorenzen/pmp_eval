import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
import numpy as np
import subprocess
import torch
import GPUtil
import wandb

def _find_free_gpus(threshold=10):
    try:
        # assert(torch.cuda.device_count() == 8)
        util_result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )

        memory_result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )

        utilizations = [int(x.strip()) for x in util_result.stdout.split('\n') if x.strip()]
        free_memory = [int(x.strip()) for x in memory_result.stdout.split('\n') if x.strip()]

        # print(utilizations)

        free_gpus = [i for i, (util, mem) in enumerate(zip(utilizations, free_memory)) if util < 20 and mem >= 10000]
        return free_gpus

    except Exception as e:
        print(f"something went wrong getting free gpus: {e}")


def get_free_gpus(ngpu):
    free_gpus = _find_free_gpus()
    if free_gpus:
        selected_gpus = free_gpus[-ngpu:]
    return selected_gpus


def log_gpu_utilization():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        wandb.log({
            f"gpu_{gpu.id}_memory": gpu.memoryUsed,
            f"gpu_{gpu.id}_utilization": gpu.load * 100,
        })

def get_output_shapes(model, batch_size, device):
    shapes = {}
    def hook_fn(module, input, output):
        shapes[id(module)] = output.shape

    hooks = []
    for i, layer in enumerate(model.features):
        hooks.append(layer.register_forward_hook(hook_fn))

    # Pass data through the model
    x = torch.randn(batch_size, 3, 32, 32).to(device)  # Example input tensor
    output = model(x)

    # Print the shapes
    for i, layer in enumerate(model.features):
        print(f"Layer {i} ({layer}): {shapes[id(layer)]}")

    # Remove hooks (cleanup)
    for hook in hooks:
        hook.remove()
