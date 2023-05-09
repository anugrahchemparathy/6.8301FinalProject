import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler



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



# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = get_device()
print('Using device:', device)



def SSL_loop(args, encoder = None):
    main_branch = models.Branch(args, encoder=encoder)
    main_branch.to(device)
    print(main_branch)



def main(args):
    utils.fix_seed(args.seed)
    SSL_loop(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_proj', default='2048,2048', type=str)
    parser.add_argument('--dim_pred', default=512, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--loss', default='simclr', type=str, choices=['simclr', 'simsiam'])
    args = parser.parse_args()

    main(args)