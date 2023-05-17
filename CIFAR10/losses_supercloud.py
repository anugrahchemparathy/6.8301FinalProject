import torch
import torch.nn as nn
import torch.nn.functional as F

import math

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


def get_deterministic_quantile(tens, thresh, dim, interpolation='linear'):
    n = tens.size(dim)
    if interpolation == 'linear':
        return torch.quantile(tens, thresh, dim=dim)
    elif interpolation == 'lower':
        new_thresh = math.floor(n * thresh) / n
        return torch.quantile(tens, new_thresh, dim=dim)
    elif interpolation == 'higher':
        new_thresh = math.ceil(n * thresh) / n
        return torch.quantile(tens, new_thresh, dim=dim)

def get_prob_quantile(tens, thresh, dim):
    lower = get_deterministic_quantile(tens, thresh, dim, interpolation='lower')
    higher = get_deterministic_quantile(tens, thresh, dim, interpolation='higher')

    n = tens.size(dim)
    prob = n * thresh - math.floor(n * thresh)
    
    mask = torch.distributions.uniform.Uniform(torch.zeros(lower.size()), torch.full(lower.size(), 1.0)).sample().to('cuda')
    mask = mask < prob

    return lower * ~mask + higher * mask


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, device, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels, thresh=0.0):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
            thresh: scalar in range 0 to 1, for what fraction of the class should be
                considered as positive pairs
                0 is completely unsupervised
                1 is fully supervised
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None:
            labels = labels.contiguous()
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            # mask = torch.eq(labels, labels.T).float().to(device)
        else:
            raise TypeError("Labels cannot be None.")
        
        # normalize features
        features = torch.nn.functional.normalize(features, dim=2)

        contrast_count = features.shape[1] # = n_views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        
        contrast_labels = labels.repeat(contrast_count).contiguous()
        anchor_labels = labels.repeat(anchor_count).contiguous()

        _, a_idx = torch.sort(anchor_labels)
        _, inv_idx = torch.sort(a_idx)

        # compute logits
        # anchor_dot_contrast and logits are sorted by label along dimension 0
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # num_labels = torch.unique(labels).max() + 1
        
        # calculating percentiles for positive pairs
        # label_percentile_dists = torch.zeros(num_labels).detach()

        # masks for values in the same class
        mask = torch.eq(anchor_labels.view(-1,1), contrast_labels.view(1,-1)).to(self.device)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask # selects elements of the same class and not along diagonal

        # aug_mask is tiled identity matrices, no need to remove main diagonal
        # because it gets zeroed out when multiplied by logits_mask
        aug_mask = torch.eye(batch_size).tile((anchor_count, contrast_count)).to(self.device)

        # offset version of anchor_dot_contrast that masks diagonal entries (self) and not in same class
        # and removes its own augmented views
        temp = ((anchor_dot_contrast + 2/self.temperature) * mask  * (1 - aug_mask))
        
        sorted_temp, _ = temp[a_idx].sort(dim = -1)

        quantiles = []
        start = 0

        for label_count in labels.unique(return_counts = True, sorted = True)[1]:
            # in the case that a label has only one sample, then that row becomes all 0
            quantiles.append(
                get_prob_quantile(sorted_temp[start:start + anchor_count * label_count, -contrast_count * (label_count - 1):].to(torch.float),
                1 - thresh,
                dim = -1)
                )
            start += anchor_count * label_count

        quantiles = torch.cat(quantiles).detach()[inv_idx]

        # quantiles contains the threshold for each row
        threshold_mask = temp > quantiles.view(-1, 1)

        mask = mask * torch.logical_or(threshold_mask, aug_mask)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss