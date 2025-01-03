import torch
import threading
import time
import random
import argparse
import wandb
import os
import numpy as np
import torch.nn as nn
from torchvision import models
from dataloading import data_loader
from utils import get_free_gpus, log_gpu_utilization
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, PipelineStage, ScheduleGPipe, ScheduleInterleavedZeroBubble, ScheduleInterleaved1F1B

global rank, device, pp_group, stage_index, world_size, training_devices
def init_distributed(num_gpus=4):
    global rank, device, pp_group, stage_index, world_size, training_devices
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    stage_index = rank


    free_devices = get_free_gpus(num_gpus)
    device = torch.device(f"cuda:{free_devices[rank]}")
    training_devices = free_devices[0:world_size]
    print(f'rank {rank} getting device {free_devices[rank]}')

    dist.init_process_group(device_id=device)
    pp_group = dist.new_group()


def init_wandb(config, identifier="stages"):
    rank = int(os.environ["LOCAL_RANK"])
    wandb.init(
        # project=os.getenv("WANDB_PROJECT"),
        # id=os.getenv("WANDB_RUN_ID"),
        project="pmp_sync_test",
        group=identifier,
        name=f"stage_{rank}",
        config=config,
    )
    wandb.log({"initialization": "success"})


def get_model_stages(model, stage_index, input_example, device, num_gpus=4):
    input_example = input_example.to(device)
    if num_gpus == 4:
        if stage_index == 0:
            model_stage = nn.Sequential(*list(model.features.children())[:6])
        if stage_index == 1:
            model_stage = nn.Sequential(*list(model.features.children())[6:11])
        if stage_index == 2:
            model_stage = nn.Sequential(*list(model.features.children())[11:])
        if stage_index == 3:
            model_stage = nn.Sequential(model.avgpool, nn.Flatten(), *list(model.classifier.children()))
    elif num_gpus == 2:
        if stage_index == 0:
            model_stage = nn.Sequential(*list(model.features.children())[:11])
        if stage_index == 1:
            model_stage = nn.Sequential(*list(model.features.children())[11:], model.avgpool, nn.Flatten(), *list(model.classifier.children()))
    else:
        raise Exception("Only supports 2 or 4 GPUs")

    # print(f"Model stage {stage_index} contains:")
    # for m in model_stage.children():
    #     print(m)
    model_stage = model_stage.to(device)

    stage = PipelineStage(
        model_stage,
        stage_index,
        world_size,
        device,
        input_args=input_example
    )

    return stage

def initialize_training(schedule_name, batch_size, num_workers, n_microbatches, num_epochs, async_training, num_gpus):
    global device, world_size
    train_loader, valid_loader = data_loader(data_dir='./data',
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             valid_size=0.1)
                                             # TEMP: ONLY FOR TESTING QUICKLY WITH SMALL SET

    test_loader = data_loader(data_dir='./data',
                              batch_size=batch_size,
                              num_workers=num_workers,
                              test=True)
    
    sync_dataloader()
    num_batches = len(train_loader)
    num_classes = 100 # 100 classes for CIFAR100
    loss_fn = nn.CrossEntropyLoss()
    

    def get_schedule(schedule_name, n_microbatches, stage, loss_fn=None):
        # for multi-stage schedules, put stage in a list such that it is subscriptable 
        if schedule_name == 'GPipe':
            schedule = ScheduleGPipe(stage, n_microbatches, loss_fn=loss_fn)
        if schedule_name == 'InterleavedZeroBubble':
            schedule = ScheduleInterleavedZeroBubble([stage], n_microbatches, loss_fn=loss_fn)
        if schedule_name == 'Interleaved1F1B':
            schedule = ScheduleInterleaved1F1B([stage], n_microbatches, loss_fn=loss_fn)
        # if schedule == 'ZBVZeroBubble':
        #     Schedule = ScheduleZBVZeroBubble
        return schedule
    

    # start training...
    model = models.vgg11(progress=True, num_classes=num_classes)
    
    micro_batch_size = batch_size // n_microbatches
    if num_gpus == 4:
        if rank == 0:
            input_example = torch.rand(micro_batch_size, 3, 32, 32)
        if rank == 1:
            input_example = torch.rand(micro_batch_size, 128, 8, 8)
        if rank == 2:
            input_example = torch.rand(micro_batch_size, 256, 4, 4)
        if rank == 3:
            input_example = torch.rand(micro_batch_size, 512, 1, 1)
    if num_gpus == 2:
        if rank == 0:
            input_example = torch.rand(micro_batch_size, 3, 32, 32)
        if rank == 1:
            input_example = torch.rand(micro_batch_size, 256, 4, 4)

    stage = get_model_stages(model, rank, input_example, device, num_gpus)
    schedule = get_schedule(schedule_name, n_microbatches, stage, loss_fn)
    optimizer = torch.optim.SGD(stage.submod.parameters(), lr=0.01)
    total_time_start = time.time()
    total_training_time = 0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        stage.submod.train()
        print(f"Starting epoch with rank {rank} and device {device}")
        dist.barrier(group=pp_group)
        all_predictions = torch.empty((num_batches, batch_size), device=device, dtype=torch.long)
        all_targets = torch.empty((num_batches, batch_size), device=device, dtype=torch.long)
        all_losses =  torch.empty((num_batches, ), device=device, dtype=torch.float32)
        print(f"Epoch {epoch + 1} / {num_epochs}")

        for i, (x, target) in enumerate(train_loader):
            batch_start = time.time()
            x, target = x.to(device), target.to(device)
            broadcast_start = time.time()
            dist.broadcast(target, src=0)
            broadcast_time = time.time() - broadcast_start
            logging_calc_time = 0
            print(f"DeBugTarget: rank {rank} batch {i} has target {target[0:3]}")
            if rank == 0:
                step_start = time.time()
                schedule.step(x)
                step_time = time.time() - step_start
                if (i % 50 == 0):
                    print(f"Completed batch {i} ")
                    threading.Thread(target=log_gpu_utilization, daemon=True).start()
            elif rank == world_size - 1: #last rank
                losses = []
                step_start = time.time()
                output = schedule.step(target=target, losses=losses)
                step_time = time.time() - step_start
                log_calc_time_start = time.time()
                pred = output.argmax(dim=1)
                all_predictions[i] = pred
                all_targets[i] = target
                conv_start = time.time()
                # conv_loss = torch.tensor(losses).mean()
                conv_loss = sum(losses) / len(losses)
                print(f"DEBUG conv loss takes time: {time.time() - conv_start}")
                all_losses[i] = conv_loss
                logging_calc_time = time.time() - log_calc_time_start
            else:
                step_start = time.time()
                schedule.step()
                step_time = time.time() - step_start
            if (i % 100 == 0):
                gradient_norm = sum(p.grad.norm() for _, p in stage.submod.named_parameters() if p.grad is not None)
                print(f"DEBUG batch {i} batch_gradient_norm {gradient_norm}")
            optimizer.step()
            optimizer.zero_grad()
            batch_time = time.time() - batch_start
            bubble_start = time.time()
            dist.barrier(group=pp_group)
            bubble_time = time.time() - bubble_start

            wandb.log({
                "epoch": epoch,
                "batch_idx": i,
                "broadcast_time": broadcast_time,
                "batch_time": batch_time,
                "bubble_time": bubble_time,
                "step_time": step_time,
                "logging_calc_time": logging_calc_time,
                # "batch_gradient_norm": gradient_norm
            })

        epoch_time = time.time() - epoch_start
        total_training_time += epoch_time
        log_throughput(num_batches, batch_size, epoch_time)

        end_epoch_start = time.time()
        # compute acc
        if rank == world_size - 1:
            all_predictions = all_predictions.flatten().cpu()
            all_targets = all_targets.flatten().cpu()
            all_losses = all_losses.flatten().cpu()
            log_train(all_predictions, all_targets, all_losses, total_training_time, epoch)

        dist.barrier(group=pp_group)
        stage.submod.eval()
        num_val_batches = len(valid_loader)
        val_start = time.time()
        val_schedule = get_schedule(schedule_name, n_microbatches, stage, loss_fn=loss_fn)
        val_predictions = torch.empty((num_val_batches, batch_size), device=device, dtype=torch.long)
        val_targets = torch.empty((num_val_batches, batch_size), device=device, dtype=torch.long)
        val_latencies = []
        val_losses =  torch.empty((num_val_batches), device=device, dtype=torch.float32)
        dist.barrier(group=pp_group)
        for i, (x, target) in enumerate(valid_loader):
            latency_start = time.time()
            x, target = x.to(device), target.to(device)
            dist.broadcast(target, src=0)
            if rank == 0:
                val_schedule.step(x)
            elif rank == world_size - 1:
                # ignore target and val_losses
                batch_losses = []
                output = val_schedule.step(target=target, losses=batch_losses)
                pred = output.argmax(dim=1)
                val_predictions[i] = pred
                val_targets[i] = target
                batch_avg_loss = sum(batch_losses) / len(batch_losses)
                val_losses[i] = batch_avg_loss
                latency_time = latency_start - time.time()
                val_latencies.append(latency_time)
            else:
                val_schedule.step()
            dist.barrier(group=pp_group)

        if rank == world_size - 1:
            val_predictions = val_predictions.flatten()
            val_targets = val_targets.flatten()
            log_val(val_predictions, val_targets, val_latencies, val_losses, epoch)
            val_time = time.time() - val_start
            print(f"DEBUG Total val time is {val_time}")

        end_epoch_time = time.time() - end_epoch_start
        wandb.log({"end_epoch_time": end_epoch_time})

        optimizer.zero_grad()

        dist.barrier(group=pp_group)

    total_time = time.time() - total_time_start
    wandb.log({"total_time": total_time})

    dist.destroy_process_group()

def sync_dataloader():
    if dist.is_initialized():
        dist.barrier()
    

def log_throughput(num_batches, batch_size, epoch_time):
    throughput = num_batches * batch_size / epoch_time
    wandb.log({
        "throughput": throughput,
    })

def log_val(pred, target, latencies, losses, epoch):
    print("Calculating val Accuracy...")
    # print(f"DEBUG: pred = {pred[0:3]} \n target \ {target[0:3]}")
    # print(f"DEBUG: pred shape = {pred.shape} \n target \ {target.shape}")
    correct = (pred == target).sum().item()
    total = target.size(0)
    acc = correct / total
    avg_latency = sum(latencies) / len(latencies)
    avg_loss = sum(losses) / len(losses)
    wandb.log({
        "val_acc": acc,
        "val_losses": avg_loss,
        "val_latency": avg_latency
    })
    print(f"Epoch {epoch + 1} Val Accuracy: {acc}")


def log_train(pred, target, loss, cumulative_training_time, epoch):
    loss_values = [t.item() for t in loss]
    avg_loss = np.average(loss_values)
    correct = (pred == target).sum().item()
    total = target.size(0)
    acc = correct / total
    wandb.log({
        "train_acc": acc,
        "train_loss": avg_loss,
        "train_time": cumulative_training_time,
    })
    print(f"Epoch {epoch + 1} Accuracy: {acc}")


def print_accuracy(pred, target):
    all_predictions = torch.cat(pred)
    all_targets = torch.cat(target)
    correct = (all_predictions == all_targets).sum().item()
    total = all_targets.size(0)
    acc = correct / total
    wandb.log({
        "train_acc": acc,
    })
    print(f"Epoch Accuracy: {acc}")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_microbatches", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--async_training", type=bool, required=False, default=False)
    parser.add_argument("--identifier", type=str, required=False, default="default")
    parser.add_argument("--num_gpus", type=int, required=False, default=2)

    args = parser.parse_args()

    print(f"Schedule: {args.schedule}, Batch Size: {args.batch_size}, Learning Rate: {args.learning_rate}")

    wandb_config = {
        "batch_size": args.batch_size,
        "n_microbatches": args.n_microbatches,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "async": args.async_training,
        "num_gpus": args.num_gpus,
        "schedule": args.schedule
    }

    set_seed(1337)
    init_wandb(wandb_config, identifier=args.identifier)
    init_distributed(args.num_gpus)
    torch.set_num_threads(args.num_gpus)

    initialize_training(args.schedule, args.batch_size, num_workers=0, n_microbatches=args.n_microbatches, num_epochs=args.num_epochs, async_training=args.async_training, num_gpus=args.num_gpus)
