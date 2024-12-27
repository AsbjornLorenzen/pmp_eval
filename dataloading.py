from torchvision import datasets
from torchvision import transforms as tt
from torchvision import models
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np

def data_loader(data_dir,
                batch_size,
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test=False,
                num_workers=4):
  
    normalize = tt.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    transform = tt.Compose([
            tt.RandomCrop(32, padding=4,padding_mode='reflect'),
            tt.RandomHorizontalFlip(),
            # tt.Resize((224,224)),
            tt.ToTensor(),
            normalize,
    ])

    if test:
        dataset = datasets.CIFAR100(
          root=data_dir, train=False,
          download=True, transform=transform,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

        return data_loader

    # load the dataset
    train_dataset = datasets.CIFAR100( root=data_dir, train=True, download=True, transform=transform)

    valid_dataset = datasets.CIFAR100(root=data_dir, train=True,download=True, transform=transform,)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=num_workers)
 
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler, pin_memory=True, num_workers=num_workers)

    return (train_loader, valid_loader)