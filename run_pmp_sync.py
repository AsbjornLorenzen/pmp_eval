import torch
import argparse
import wandb
import os
import torch.nn as nn
from torchvision import models
from dataloading import data_loader
from utils import get_free_gpus
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, PipelineStage, ScheduleGPipe

global rank, device, pp_group, stage_index, num_stages
def init_distributed():
    global rank, device, pp_group, stage_index, num_stages
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    free_devices = get_free_gpus(4)
    device = torch.device(f"cuda:{free_devices[rank]}")
    print(f'rank {rank} getting device {free_devices[rank]}')

    dist.init_process_group()
    pp_group = dist.new_group()

    stage_index = rank
    num_stages = world_size


def init_wandb():
    wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        id=os.getenv("WANDB_RUN_ID"),
        resume="allow",
    )

    wandb.log({"initialization": "success"})


def get_model_stages(model, stage_index, input_example):
    if stage_index == 0:
        model_stage = nn.Sequential(*list(model.features.children())[:6])
    if stage_index == 1:
        model_stage = nn.Sequential(*list(model.features.children())[6:11])
    if stage_index == 2:
        model_stage = nn.Sequential(*list(model.features.children())[11:])
    if stage_index == 3:
        model_stage = nn.Sequential(model.avgpool, *list(model.classifier.children()))

    # print(f"Model stage {stage_index} contains:")
    # for m in model_stage.children():
    #     print(m)

    stage = PipelineStage(
        model_stage,
        stage_index,
        num_stages,
        device,
        # input_args=input_example
    )

    return stage


def initialize_training(batch_size, num_workers, n_microbatches, num_epochs):
    global device
    train_loader, valid_loader = data_loader(data_dir='./data',
                                             batch_size=batch_size,
                                             num_workers=num_workers)

    test_loader = data_loader(data_dir='./data',
                              batch_size=batch_size,
                              num_workers=num_workers,
                              test=True)

    total_steps = len(train_loader)
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

    # start training...
    # MAYBE SEND TO TORCH.META??
    model = models.vgg11(progress=True, num_classes=100)
    
    micro_batch_size = batch_size // n_microbatches
    if rank == 0:
        input_example = torch.rand(micro_batch_size, 3, 32, 32)
    if rank == 1:
        input_example = torch.rand(micro_batch_size, 64, 32, 32)
    if rank == 2:
        input_example = torch.rand()
    if rank == 1:
        input_example = torch.rand(1,2,3,4)
    stage = get_model_stages(model, rank, input_example)
    schedule = ScheduleGPipe(stage, n_microbatches, loss_fn=loss_fn)
    # wandb.watch(stage, log="all", log_freq=10)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} / {num_epochs}")
        for i, (x, target) in enumerate(train_loader):
            print(f"Rank {rank}: Preparing batch {i}")
            x, target = x.to(device), target.to(device)
            print(f"Rank {rank}: Loaded batch {i}")
            print(f"Rank {rank} processing batch {i}")
            if rank == 0:
                print(f"Rank {rank}, Stage {stage_index}: Stepping with input.")
                schedule.step(x)
                print(f"Rank {rank}, Stage {stage_index}: Step completed.")
            elif rank == 3:
                losses = []
                print(f"Rank {rank}, Stage {stage_index}: Stepping with input.")
                schedule.step(target=target, losses=losses)
                print(f"Rank {rank}, Stage {stage_index}: Step completed.")
                print(f"Batch {i + 1} / {total_steps}")
            else:
                print(f"Rank {rank}, Stage {stage_index}: Stepping with input.")
                schedule.step()
                print(f"Rank {rank}, Stage {stage_index}: Step completed.")
            dist.barrier(group=pp_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_microbatches", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)

    args = parser.parse_args()

    print(f"Schedule: {args.schedule}, Batch Size: {args.batch_size}, Learning Rate: {args.learning_rate}")

    init_wandb()
    init_distributed()
    torch.set_num_threads(4)

    initialize_training(args.batch_size, num_workers=4, n_microbatches=args.n_microbatches, num_epochs=args.num_epochs)
