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



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = get_device()
print('Using device:', device)



def SSL_loop(args, encoder = None):
    os.makedirs('saved_experiments/' + args.path_dir, exist_ok=True)
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


    main_branch = models.Branch(args, encoder=encoder)
    main_branch.to(device)
    backbone = main_branch.encoder
    projector = main_branch.projector
    torch.save(dict(epoch=0, state_dict=main_branch.state_dict()), os.path.join('saved_experiments/' + args.path_dir, '0.pth'))

    optimizer = torch.optim.SGD(main_branch.parameters(), momentum=0.9, lr=args.lr * args.bsz / 256, weight_decay=args.wd)
    scaler = GradScaler()

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

                loss = losses.info_nce_loss(z1, z2, device=device) / 2 + losses.info_nce_loss(z2, z1, device=device) / 2

                if args.lmbd > 0:
                    rotated_images, rotated_labels = datasets.rotate_images(inputs[2])
                    b = backbone(rotated_images)
                    logits = main_branch.predictor2(b)
                    rot_loss = F.cross_entropy(logits, rotated_labels)
                    loss += args.lmbd * rot_loss

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

        line_to_print = (
            f'epoch: {e} | '
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


    parser.add_argument('--lmbd', default=0.0, type=float)
    parser.add_argument('--dim_proj', default='2048,3', type=str)
    parser.add_argument('--dim_pred', default=512, type=int)
    parser.add_argument('--loss', default='simclr', type=str, choices=['simclr', 'simsiam'])
    parser.add_argument('--path_dir', default='sample', type=str)
    parser.add_argument('--save_every', default=5, type=int)

    
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--fp16', action='store_true')


    parser.add_argument('--classes', default=None, type=str)
    args = parser.parse_args()

    main(args)