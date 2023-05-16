import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

from pseudoR2 import *


class MetricEngine:
    def __init__(self, test_loader, model, device, use_r2 = True, use_pca = True, variance_threshold = 0.90):
        self.lr = LogisticRegression(fit_intercept=False, solver='lbfgs')
        self.pca = PCA(n_components=3)
        self.variance_threshold = variance_threshold

        self.use_r2 = use_r2
        self.use_pca = use_pca

        self.test_loader = test_loader
        self.model = model
        self.device = device

    def get_metrics(self):
        metrics = {}
        embeddings = []
        labels_list = []
        for batch_idx, (data, labels) in enumerate(self.test_loader):
            data, labels_batch = data.to(self.device), labels.to(self.device)
            embeddings_batch = self.model(data)

            embeddings.append(embeddings_batch)
            labels_list.append(labels_batch)

        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels_list, dim=0)
        metrics = self.compute_metrics(embeddings, labels)
        return metrics

    def compute_metrics(self, embeddings, labels):
        metrics = {}
        X = embeddings.detach().cpu().numpy() # (dataset_size, output_dim)
        y = labels.detach().cpu().numpy() # (dataset_size,)

        if self.use_r2:
            self.lr.fit(X, y)
            lr_coef = np.array(self.lr.coef_).transpose()
            y_pred = self.lr.predict_proba(X)[:, 1]

            metrics['efron_r2'] = efron_rsquare(y, y_pred)
            metrics['mcfadden_r2'] = mcfadden_rsquare(lr_coef, X, y)
            metrics['mz_r2'] = mz_rsquare(y_pred)
            metrics['count_r2'] = count_rsquare(y, y_pred)

        if self.use_pca:
            X_pca = self.pca.fit(X)
            explained_variance_ratio = self.pca.explained_variance_ratio_
            cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
            num_components = np.sum(cumulative_explained_variance_ratio < self.variance_threshold) + 1

            metrics['pca_dim'] = num_components

        return metrics