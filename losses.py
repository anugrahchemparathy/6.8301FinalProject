import torch
import torch.nn as nn
import torch.nn.functional as F

def info_nce_loss(z1, z2, temperature=0.5, device = 'cpu'):
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)

    logits = z1 @ z2.T
    logits /= temperature
    n = z2.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).to(device)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss
