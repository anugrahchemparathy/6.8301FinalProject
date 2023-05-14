import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import torchvision
import torchvision.transforms as T
from PIL import Image


import os
import math
import time
import copy
import argparse


from device_tools import get_device, t2np
import datasets
import losses
import models
import utils
# from utils import knn_monitor, fix_seed

import matplotlib.pyplot as plt
import numpy as np

normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
single_transform = T.Compose([T.ToTensor(), normalize])

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main(args):
    train_loader = torch.utils.data.DataLoader(
        dataset=datasets.dataset_class_mapper(torchvision.datasets.CIFAR10(
            '../data', train=True, transform=single_transform, download=True
        ), args.classes),
        shuffle=True,
        batch_size=16,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    
    print(args.classes)

    dataiter = iter(train_loader)
    images, labels = next(dataiter) # [nparr[16, 3, 32, 32], nparr[16, 3, 32, 32]]

    # print labels
    print(' '.join(f'{args.classes[labels[j]]:5s}' for j in range(16)))
    print(' '.join(f'{labels[j]}' for j in range(16)))
    # show images
    imshow(torchvision.utils.make_grid(images))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup_epochs', default=10, type=int)
    
    parser.add_argument('--lr', default=0.03, type=float)
    parser.add_argument('--bsz', default=512, type=int)
    parser.add_argument('--wd', default=0.0005, type=float)


    parser.add_argument('--dim_proj', default='2048,3', type=str)
    parser.add_argument('--dim_pred', default=512, type=int)
    parser.add_argument('--loss', default='simclr', type=str, choices=['simclr', 'simsiam'])
    parser.add_argument('--path_dir', default='sample', type=str)
    parser.add_argument('--save_every', default=5, type=int)

    
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--fp16', action='store_true')


    parser.add_argument('--classes', default=None, type=str)
    args = parser.parse_args()
    if args.classes != None:
        args.classes = args.classes.split(",") # comma-sep

    main(args)