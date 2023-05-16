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

import matplotlib.pyplot as plt

from device_tools import get_device, t2np
import datasets
import losses
import models
import utils
# from utils import knn_monitor, fix_seed

normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
single_transform = T.Compose([T.ToTensor(), normalize])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = get_device()
print('Using device:', device)


def SSL_loop(args, encoder = None):
    os.makedirs('saved_experiments/' + args.path_dir, exist_ok=True)
    # os.makedirs('saved_plots/' + args.path_dir, exist_ok=True)
    file_to_update = open(os.path.join('saved_experiments/' + args.path_dir, 'train_and_eval.log'), 'w')

    train_loader = torch.utils.data.DataLoader(
        dataset=datasets.dataset_class_mapper(torchvision.datasets.CIFAR10(
            '../data', train=True, transform=datasets.ContrastiveLearningTransform(), download=True
        ), args.classes),
        shuffle=True,
        batch_size=args.bsz,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    # to calculate KNN accuracy

    memory_loader = torch.utils.data.DataLoader(
        dataset=datasets.dataset_class_mapper(torchvision.datasets.CIFAR10(
            '../data', train=True, transform=single_transform, download=True
        ), args.classes),
        shuffle=False,
        batch_size=args.bsz,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=datasets.dataset_class_mapper(torchvision.datasets.CIFAR10(
            '../data', train=False, transform=single_transform, download=True,
        ), args.classes),
        shuffle=False,
        batch_size=args.bsz,
        pin_memory=True,
        num_workers=args.num_workers
    )

    main_branch = models.Branch(args, encoder=encoder)
    main_branch.to(device)
    backbone = main_branch.encoder
    projector = main_branch.projector
    torch.save(dict(epoch=0, state_dict=main_branch.state_dict()), os.path.join('saved_experiments/' + args.path_dir, '0.pth'))

    loss_inst = losses.SupConLoss(device, temperature = args.temperature, base_temperature = args.temperature)

    optimizer = torch.optim.SGD(main_branch.parameters(), momentum=0.9, lr=args.lr * args.bsz / 256, weight_decay=args.wd)
    scaler = GradScaler()
    
    loss_list = []
    acc_list = []

    start = time.time()
    for e in range(1, args.epochs + 1):
        main_branch.train()

        for it, (inputs, y) in enumerate(train_loader, start=(e - 1) * len(train_loader)):
            print(f"Step {it % len(train_loader) + 1}/{len(train_loader)}...", end="\r")
            lr = utils.adjust_learning_rate(epochs=args.epochs,
                                      warmup_epochs=args.warmup_epochs,
                                      base_lr=args.lr * args.bsz / 256,
                                      optimizer=optimizer,
                                      loader=train_loader,
                                      step=it)
            
            main_branch.zero_grad()

            def forward_step():
                x1 = inputs[0].to(device)
                x2 = inputs[1].to(device)
                b1 = backbone(x1)
                b2 = backbone(x2)
                z1 = projector(b1)
                z2 = projector(b2)

                z = torch.stack((z1, z2), axis = 1)
                loss = loss_inst(z, labels = y, thresh = args.threshold)
                # loss = losses.info_nce_loss(z1, z2, device=device) / 2 + losses.info_nce_loss(z2, z1, device=device) / 2


                return loss

            # optimization step
            if args.fp16:
                with autocast():
                    loss = forward_step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                loss = forward_step()
                loss.backward()

                optimizer.step()

        loss_list.append(loss.item())

        if args.fp16:
            with autocast():
                knn_acc = utils.knn_loop(backbone, memory_loader, test_loader, device)
        else:
            knn_acc = utils.knn_loop(backbone, memory_loader, test_loader, device)

        acc_list.append(knn_acc)

        line_to_print = (
            f'epoch: {e} | knn_acc: {knn_acc:.3f} | '
            f'loss: {loss.item():.3f} | lr: {lr:.6f} | '
            f'time_elapsed: {time.time() - start:.3f}'
        )
        if file_to_update:
            file_to_update.write(line_to_print + '\n')
            file_to_update.flush()
        print(line_to_print)

        if e % args.save_every == 0:
            torch.save(dict(epoch=e, state_dict=main_branch.state_dict()),
                       os.path.join('saved_experiments/' + args.path_dir, f'{e}.pth'))

    loss_np = np.array(loss_list)
    acc_np = np.array(acc_list)
    np.save(os.path.join('saved_experiments/' + args.path_dir, 'loss.npy'), loss_np)
    np.save(os.path.join('saved_experiments/' + args.path_dir, 'acc.npy'), acc_np)

    plt.plot(np.arange(len(loss_np)), loss_np)
    plt.savefig(os.path.join('saved_experiments/' + args.path_dir, 'loss_plot.png'))
    plt.clf()
    plt.plot(np.arange(len(acc_np)), acc_np)
    plt.savefig(os.path.join('saved_experiments/' + args.path_dir, 'knn_acc_plot.png'))

    return main_branch.encoder, file_to_update



def main(args):
    utils.fix_seed(args.seed)
    SSL_loop(args)




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
    parser.add_argument('--temperature', default=0.5, type=float)

    parser.add_argument('--threshold', default = 0.0, type = float) # default unsupervised


    parser.add_argument('--classes', default=None, type=str)
    args = parser.parse_args()
    if args.classes != None:
        args.classes = args.classes.split(",") # comma-sep

    main(args)