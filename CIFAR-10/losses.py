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

def rmseNCE(z1, z2, temperature=0.1, device = 'cpu'):
    z1 = torch.unsqueeze(z1,0) #(nxr -> 1xnxr)
    z2 = torch.unsqueeze(z2,0)
    
    euclidean_dist = -torch.cdist(z1,z2,p=2.0) #p=2.0 for standard euclidean distance
    euclidean_dist = torch.squeeze(euclidean_dist)

    logits = euclidean_dist
    logits /= temperature
    n = z2.shape[1] #since its unsqueezed (1xnxr), the first real dimension is the second one
    labels = torch.arange(0, n, dtype=torch.long).to(device)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss