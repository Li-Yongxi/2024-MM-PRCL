"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import sys

import torch
import torch.nn as nn
from torch.nn import functional as F
import loss as l

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=1, contrast_mode='all',
                 base_temperature=1):
        """

        Returns:
            object: 
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def weighted_con_loss(self, view1, view2, labels, prob_v=None, prob_i=None, t=0.25, gamma=0.1, debias="none", tau_plus=0.1, sim_method="dist", weighted_softmax=False):
        view1_feature = F.normalize(view1, dim=1)
        view2_feature = F.normalize(view2, dim=1)
        # print(view1_feature.shape, view2_feature.shape)

        # cosine similarity: NxN
        sim_view12 = torch.matmul(view1_feature, view2_feature.T) / t
        sim_view11 = torch.matmul(view1_feature, view1_feature.T) / t
        sim_view22 = torch.matmul(view2_feature, view2_feature.T) / t

        if labels is None:

            label_sim = torch.eye(len(view1_feature), dtype=torch.float32).cuda()
            label_neg = 1 - label_sim
        else:
            labels = labels.float()
            labels = labels.contiguous().view(-1, 1).cuda()

            label_sim = torch.eq(labels, labels.T).float()
            label_neg = torch.ne(labels, labels.T).float()

        # label_sim = label_sim ** 0.5
        pro_inter = label_sim / label_sim.sum(1, keepdim=True).clamp(min=1e-6)
        label_sim_intra = (label_sim - torch.eye(label_sim.shape[0]).cuda()).clamp(min=0)
        pro_intra = label_sim_intra / label_sim_intra.sum(1, keepdim=True).clamp(min=1e-6)



        dist_mat = l.pdist_torch(view1_feature, view2_feature)
        weights_ap = dist_mat * label_sim
        weights_an = dist_mat * label_neg

        if weighted_softmax:
            dist_mat = l.pdist_torch(view1_feature, view2_feature)
            weights_ap = dist_mat * label_sim
            weights_an = dist_mat * label_neg
            weights_ap = l.softmax_weights(weights_ap, label_sim)
            weights_an = l.softmax_weights(weights_an, label_neg)
        if debias == "debias_weighted":

            N = len(view1) * 2 - 2
            # Ng = (-tau_plus * N * weights_ap + weights_an.sum(dim=-1)) / (1 - tau_plus + 1e-6)
            Ng = (-tau_plus * N * weights_ap + weights_an) / (1 - tau_plus + 1e-6)
            Ng = torch.clamp(Ng, min=N * torch.e ** (-1 / t))
            weights_all = weights_ap + Ng
            logits_view12 = sim_view12 * weights_ap - torch.log(torch.exp(1.06 * sim_view12 * weights_all ).sum(1, keepdim=True))
            logits_view21 = sim_view12.T * weights_ap - torch.log(torch.exp(1.06 * sim_view12.T * weights_all ).sum(1, keepdim=True))
            logits_view11 = sim_view11 * weights_ap - torch.log(torch.exp(1.06 * sim_view11 * weights_all ).sum(1, keepdim=True))
            logits_view22 = sim_view22 * weights_ap - torch.log(torch.exp(1.06 * sim_view22 * weights_all ).sum(1, keepdim=True))
        else:
            logits_view12 = sim_view12 - torch.log(
                torch.exp(1.06 * sim_view12).sum(1, keepdim=True))
            logits_view21 = sim_view12.T - torch.log(
                torch.exp(1.06 * sim_view12.T).sum(1, keepdim=True))
            logits_view11 = sim_view11 - torch.log(
                torch.exp(1.06 * sim_view11).sum(1, keepdim=True))
            logits_view22 = sim_view22 - torch.log(
                torch.exp(1.06 * sim_view22).sum(1, keepdim=True))

        if prob_v is None:
            mean_log_prob_pos_view12 = (pro_inter * logits_view12).sum(1)
            mean_log_prob_pos_view21 = (pro_inter * logits_view21).sum(1)
            mean_log_prob_pos_view11 = (pro_intra * logits_view11).sum(1)
            mean_log_prob_pos_view22 = (pro_intra * logits_view22).sum(1)
        else:
            prob_vi = torch.minimum(prob_v, prob_i)
            mean_log_prob_pos_view12 = prob_vi * (pro_inter * logits_view12).sum(1)
            mean_log_prob_pos_view21 = prob_vi * (pro_inter * logits_view21).sum(1)
            mean_log_prob_pos_view11 = prob_v * (pro_intra * logits_view11).sum(1)
            mean_log_prob_pos_view22 = prob_i * (pro_intra * logits_view22).sum(1)

        # supervised cross-modal contrastive loss
        loss = - mean_log_prob_pos_view12.mean() - mean_log_prob_pos_view21.mean() \
               - gamma * (mean_log_prob_pos_view11.mean() + mean_log_prob_pos_view22.mean())


        return loss