import torch
import argparse
import wandb
import os
import torch.nn as nn
from dataloading import data_loader
from models import VGG11
from utils import get_free_gpus
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, PipelineStage, ScheduleGPipe

global rank, device, pp_group, stage_index, num_stages
def init_distributed():
    global rank, device, pp_group, stage_index, num_stages
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{device}")

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


def initialize_training(batch_size, num_workers):
    train_loader, valid_loader = data_loader(data_dir='./data',
                                             batch_size=batch_size,
                                             num_workers=num_workers)

    test_loader = data_loader(data_dir='./data',
                              batch_size=batch_size,
                              num_workers=num_workers,
                              test=True)

    total_steps = len(train_loader)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)  

    # start training...
    model = VGG11(progress=True, num_classes=100)
    stage = get_model_stages(model, rank)
    wandb.watch(stage, log="all", log_freq=10)



def get_model_stages(model, stage_index):
    if stage_index == 0:
        model_stage = nn.Sequential(*list(model.features.children())[:6])
    if stage_index == 1:
        model_stage = nn.Sequential(*list(model.features.children())[6:11])
    if stage_index == 2:
        model_stage = nn.Sequential(*list(model.features.children())[11:])
    if stage_index == 3:
        model_stage = nn.Sequential(model.avgpool, *list(model.classifier.children()))

    print(f"Model stage {stage_index} contains:")
    for m in model_stage.children():
        print(m)

    stage = PipelineStage(
        model,
        stage_index,
        num_stages,
        device,
    )

    return stage






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--microbatch_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--device", type=int, required=True)

    args = parser.parse_args()
    global device
    device = args.device

    print(f"Schedule: {args.schedule}, Batch Size: {args.batch_size}, Learning Rate: {args.learning_rate}")

    init_wandb()
    init_distributed()
    torch.set_num_threads(4)

    initialize_training(args.batch_size, num_workers=4)