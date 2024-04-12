import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable
from utils import one_hot_embedding, relu_evidence
import pdb
from util.board_writter import BoardWriter


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, label_assign, targets, true_targets, prob, threshold=0.6
                , alpha=100, normalize_feature=False, epoch=None):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct, closest_negative.shape[0]




class TripletLoss_ADP(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self, alpha=1, gamma=1, square=0):
        super(TripletLoss_ADP, self).__init__()
        # self.ranking_loss = nn.SoftMarginLoss()
        self.ranking_loss = nn.SoftMarginLoss(reduction='none')
        self.alpha = alpha
        self.gamma = gamma
        self.square = square

    def forward(self, inputs, label_assign, targets, true_targets, prob, threshold=0.6, alpha=100, normalize_feature=False, epoch=None):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap * self.alpha, is_pos)
        weights_an = softmax_weights(-dist_an * self.alpha, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        # ranking_loss = nn.SoftMarginLoss(reduction = 'none')
        # loss1 = ranking_loss(closest_negative - furthest_positive, y)

        # squared difference
        self.square = 0
        if self.square == 0:
            y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
            loss = self.ranking_loss(self.gamma * (closest_negative - furthest_positive), y)
        else:
            diff_pow = torch.pow(furthest_positive - closest_negative, 2) * self.gamma
            diff_pow = torch.clamp_max(diff_pow, max=88)

            # Compute ranking hinge loss
            y1 = (furthest_positive > closest_negative).float()
            y2 = y1 - 1
            y = -(y1 + y2)

            loss = self.ranking_loss(diff_pow, y)
        loss = prob * loss
        loss = torch.mean(loss)

        # loss = self.ranking_loss(self.gamma*(closest_negative - furthest_positive), y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct, closest_negative.shape[0]

class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()

    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T = 3

        predict = F.log_softmax(pred / T, dim=1)
        target_data = F.softmax(label / T, dim=1)
        target_data = target_data + 10 ** (-7)
        target = Variable(target_data.data.cuda(), requires_grad=False)
        loss = T * T * ((target * (target.log() - predict)).sum(1).sum() / target.size()[0])
        return loss


def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis=1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis=1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx

def _batch_hard(mat_distance, mat_similarity, indice=False):
    # mat_similarity=reduce(lambda x, y: x * y, mat_similaritys)
    # mat_similarity=mat_similaritys[0]*mat_similaritys[1]*mat_similaritys[2]*mat_similaritys[3]
    # sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
    
    sorted_mat_distance, positive_indices = torch.sort(mat_distance + (1 - mat_similarity), dim=1, descending=True)
    hard_p = sorted_mat_distance[:, 0]
    hard_p_indice = positive_indices[:, 0]
    # sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
    sorted_mat_distance, negative_indices = torch.sort(mat_distance + (mat_similarity), dim=1, descending=False)
    hard_n = sorted_mat_distance[:, 0]
    hard_n_indice = negative_indices[:, 0]
    if (indice):
        return hard_p, hard_n, hard_p_indice, hard_n_indice
    return hard_p, hard_n

class RobustTripletLoss_final(nn.Module):
    def __init__(self, batch_size, margin):
        super(RobustTripletLoss_final, self).__init__()
        self.batch_size = batch_size
        self.margin = margin

    def forward(self, inputs, prediction, targets, true_targets, prob, threshold, epoch=None):
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the positive and negative
        is_pos = targets.expand(n, n).eq(targets.expand(n, n).t())
        is_neg = targets.expand(n, n).ne(targets.expand(n, n).t())
        is_confident = (prob >= threshold)
        dist_ap, dist_an = [], []
        cnt, loss = 0, 0
        loss_inverse = False
        # if epoch >=1:
        #     print(targets)
            # print(is_pos)
            # print(is_neg)
        for i in range(n):
            if is_confident[i]:
                pos_idx = (torch.nonzero(is_pos[i].long())).squeeze(1)
                neg_idx = (torch.nonzero(is_neg[i].long())).squeeze(1)
                # if epoch >= 0:
                #     print(pos_idx)
                random_pos_index = int(np.random.choice(pos_idx.cpu().numpy(), 1))
                cct = 0
                while random_pos_index == i:
                    random_pos_index = int(np.random.choice(pos_idx.cpu().numpy(), 1))
                    cct += 1
                    if cct > 20:
                        print(pos_idx)



                rank_neg_index = dist[i][neg_idx].argsort()
                hard_neg_index = rank_neg_index[0]
                hard_neg_index = neg_idx[hard_neg_index]

                dist_ap.append(dist[i][random_pos_index].unsqueeze(0))
                dist_an.append(dist[i][hard_neg_index].unsqueeze(0))

                if prob[random_pos_index] >= threshold and prob[hard_neg_index] >= threshold:
                    # TP-TN
                    pass

                elif prob[random_pos_index] >= threshold and prob[hard_neg_index] < threshold:
                    is_FN = (torch.argmax(prediction[hard_neg_index]) == targets[i])
                    # TP-FN
                    if is_FN:
                        tmp = rank_neg_index[1]
                        hard_neg_index_new = neg_idx[tmp]
                        j = 1
                        loop_cnt = 0
                        while prob[hard_neg_index_new] < threshold:
                            j += 1
                            tmp = rank_neg_index[j]
                            hard_neg_index_new = neg_idx[tmp]
                            loop_cnt += 1
                            if loop_cnt >= 10:
                                # print("------------warning, break the death loop---------------")
                                break
                        dist_ap[cnt] = (dist[i][random_pos_index].unsqueeze(0) +
                                        dist[i][hard_neg_index].unsqueeze(0)) / 2
                        dist_an[cnt] = dist[i][hard_neg_index_new].unsqueeze(0)
                    # TP-TN
                    else:
                        pass

                elif prob[random_pos_index] < threshold and prob[hard_neg_index] >= threshold:
                    # FP-TN
                    random_pos_index_new = int(np.random.choice(pos_idx.cpu().numpy(), 1))
                    loop_cnt = 0
                    while random_pos_index_new == i or prob[random_pos_index_new] < threshold:
                        random_pos_index_new = int(np.random.choice(pos_idx.cpu().numpy(), 1))
                        loop_cnt += 1
                        if loop_cnt >= 5:
                            # print("------------warning, break the death loop---------------")
                            break
                    dist_an[cnt] = (dist[i][random_pos_index].unsqueeze(0)
                                    + dist[i][hard_neg_index].unsqueeze(0)) / 2
                    dist_ap[cnt] = dist[i][random_pos_index_new].unsqueeze(0)

                elif prob[random_pos_index] < threshold and prob[hard_neg_index] < threshold:
                    is_FN = (torch.argmax(prediction[hard_neg_index]) == targets[i])
                    # FP-FN
                    if is_FN:
                        loss_inverse = True
                    # FP-TN
                    else:
                        random_pos_index_new = int(np.random.choice(pos_idx.cpu().numpy(), 1))
                        loop_cnt = 0
                        while random_pos_index_new == i or prob[random_pos_index_new] < threshold:
                            random_pos_index_new = int(np.random.choice(pos_idx.cpu().numpy(), 1))
                            loop_cnt += 1
                            if loop_cnt >= 5:
                                print("------------warning, break the death loop---------------")
                                break
                        dist_an[cnt] = (dist[i][random_pos_index].unsqueeze(0)
                                        + dist[i][hard_neg_index].unsqueeze(0)) / 2
                        dist_ap[cnt] = dist[i][random_pos_index_new].unsqueeze(0)

                if loss_inverse:
                    loss += torch.clamp(dist_an[cnt] - dist_ap[cnt] + self.margin, 0)
                else:
                    loss += torch.clamp(dist_ap[cnt] - dist_an[cnt] + self.margin, 0)

                cnt += 1
                loss_inverse = False
            else:
                continue

        # compute accuracy
        if cnt == 0:
            return torch.Tensor([0.]).to(inputs.device), 0, cnt
        else:
            dist_ap = torch.cat(dist_ap)
            dist_an = torch.cat(dist_an)
            correct = torch.ge(dist_an, dist_ap).sum().item()
            return loss / cnt, correct,
class UncTripletLoss(nn.Module):

    def __init__(self):
        super(UncTripletLoss, self).__init__()
        self.margin = 0.3

    def forward(self, inputs, label_assign, targets, true_targets, prob, threshold=0.6
                , alpha=100, normalize_feature=False, epoch=None):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        mat_dist = pdist_torch(inputs, inputs)

        assert mat_dist.size(0) == mat_dist.size(1)
        N = mat_dist.size(0)

        # mat_sims=[]
        # for label in labels:
        # 	mat_sims.append(label.expand(N, N).eq(label.expand(N, N).t()).float())
        # mat_sim=reduce(lambda x, y: x + y, mat_sims)
        mat_sim = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        dist_ap, dist_an, ap_idx, an_idx = _batch_hard(mat_dist, mat_sim, indice=True)
        assert dist_an.size(0) == dist_ap.size(0)
        triple_dist = torch.stack((dist_ap, dist_an), dim=1)
        triple_dist = F.log_softmax(triple_dist, dim=1)

        # mat_dist_ref = euclidean_dist(emb2, emb2)
        # dist_ap_ref = torch.gather(mat_dist_ref, 1, ap_idx.view(N,1).expand(N,N))[:,0]
        # dist_an_ref = torch.gather(mat_dist_ref, 1, an_idx.view(N,1).expand(N,N))[:,0]
        # triple_dist_ref = torch.stack((dist_ap_ref, dist_an_ref), dim=1)
        # triple_dist_ref = F.softmax(triple_dist_ref, dim=1).detach()

        # torch.gather
        uncer_ap_ref = torch.gather(prob, 0, ap_idx) + prob
        uncer_an_ref = torch.gather(prob, 0, an_idx) + prob
        uncer = torch.stack((uncer_ap_ref, uncer_an_ref), dim=1).detach() / 2.0
        print(uncer)
        print(triple_dist)
        exit(1)

        loss = (-uncer * triple_dist).mean(0).sum() / 10
        #(uncer * triple_dist)[:,0].mean(0).sum()-(uncer * triple_dist)[:,1].mean(0).sum() #- triple_dist[:,1].mean()
        # print(loss)
        # exit(0)

        return loss, torch.tensor(0), torch.tensor(0)

class Edl_loss(nn.Module):
    def __init__(self, device='cpu'):
        super(Edl_loss, self).__init__()
        self.device = device


    def kl_divergence(self, alpha, num_classes):
        ones = torch.ones([1, num_classes], dtype=torch.float32, device=self.device)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        first_term = (
                torch.lgamma(sum_alpha)
                - torch.lgamma(alpha).sum(dim=1, keepdim=True)
                + torch.lgamma(ones).sum(dim=1, keepdim=True)
                - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )
        second_term = (
            (alpha - ones)
                .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
                .sum(dim=1, keepdim=True)
        )
        kl = first_term + second_term
        return kl

    def loglikelihood_loss(self, y, alpha):
        device = self.device
        y = y.to(device)
        alpha = alpha.to(device)
        S = torch.sum(alpha, dim=1, keepdim=True)
        loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
        loglikelihood_var = torch.sum(
            alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
        )
        loglikelihood = loglikelihood_err + loglikelihood_var
        return loglikelihood

    def mse_loss(self, y, alpha, epoch_num, num_classes, annealing_step):
        device = self.device

        y = y.to(device)
        alpha = alpha.to(device)
        loglikelihood = self.loglikelihood_loss(y, alpha)

        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
        )

        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * self.kl_divergence(kl_alpha, num_classes)
        return loglikelihood + kl_div

    def edl_loss(self, func, y, alpha, epoch_num, num_classes, annealing_step):
        y = y.to(self.device)
        alpha = alpha.to(self.device)
        S = torch.sum(alpha, dim=1, keepdim=True)

        A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
        )

        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * self.kl_divergence(kl_alpha, num_classes)
        return A + kl_div

    def edl_loss_prob(self, func, y, alpha, epoch_num, num_classes, annealing_step, prob):
        y = y.to(self.device)
        alpha = alpha.to(self.device)
        S = torch.sum(alpha, dim=1, keepdim=True)

        A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)
        PA = prob * A
        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
        )
        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * self.kl_divergence(kl_alpha, num_classes)
        # print("PA", PA)
        # print("PA", PA)
        # print("KL_DIV:", kl_div)
        # print("kl_alpha", kl_alpha)
        # print("annealing_coef", annealing_coef)
        return PA #+ kl_div
    def edl_loss_nonkl(self, func, y, alpha, epoch_num, num_classes, annealing_step):
        y = y.to(self.device)
        alpha = alpha.to(self.device)
        S = torch.sum(alpha, dim=1, keepdim=True)

        A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)
        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
        )
        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * self.kl_divergence(kl_alpha, num_classes)

        return A
    def edl_digamma_loss(
        self, output, target, epoch_num, num_classes, annealing_step
    ):
        evidence = relu_evidence(output)
        alpha = evidence + 1
        loss = torch.mean(
            self.edl_loss(
                torch.digamma, target, alpha, epoch_num, num_classes, annealing_step))
        return loss


    def edl_digamma_loss_prob(
        self, output, target, epoch_num, num_classes, annealing_step, labels, prob
    ):
        # ce_loss = F.cross_entropy(output, labels, reduction="none")

        evidence = relu_evidence(output)
        prob = prob.reshape(-1, 1)
        # temp = evidence[torch.arange(
        #     evidence.size(0)), labels.long()] + ce_loss
        # evidence.scatter(1, labels.reshape(-1, 1).long(), temp.reshape(-1, 1))
        evidence = evidence
        target = target.to(self.device)

        alpha = evidence + 1
        loss = torch.mean(
            self.edl_loss_prob(
                torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, prob)
        )
        return loss

    def edl_digamma_loss_nonkl(
        self, output, target, epoch_num, num_classes, annealing_step, labels, prob
    ):
        # ce_loss = F.cross_entropy(output, labels, reduction="none")

        evidence = relu_evidence(output)
        prob = prob.reshape(-1, 1)
        # temp = evidence[torch.arange(
        #     evidence.size(0)), labels.long()] + ce_loss
        # evidence.scatter(1, labels.reshape(-1, 1).long(), temp.reshape(-1, 1))
        evidence = evidence
        target = target.to(self.device)

        alpha = evidence + 1
        loss = torch.mean(
            self.edl_loss_nonkl(
                torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, prob)
        )
        return loss
    def edl_mse_loss(self, output, target, epoch_num, num_classes, annealing_step):

        evidence = relu_evidence(output)
        alpha = evidence + 1
        loss = torch.mean(
            self.mse_loss(target, alpha, epoch_num, num_classes, annealing_step)
        )
        return loss

    def edl_log_loss(self, output, labels, target, epoch_num, num_classes, annealing_step):

        evidence = relu_evidence(output)
        target = target.to(self.device)
        classify = torch.nn.Softmax(dim=1)(output).argmax(1).to(self.device)
        noisy_index = (classify != labels)


        evidence = evidence * noisy_index.reshape(-1, 1)
        alpha = evidence + 1
        loss = torch.mean(
            self.edl_loss(
                torch.log, target, alpha, epoch_num, num_classes, annealing_step
            )
        )
        return loss
    def edl_log_loss_ori(self, output, labels, target, epoch_num, num_classes, annealing_step):

        evidence = relu_evidence(output)
        target = target.to(self.device)
        evidence = evidence
        alpha = evidence + 1
        loss = torch.mean(
            self.edl_loss(
                torch.log, target, alpha, epoch_num, num_classes, annealing_step
            )
        )
        return loss
    def edl_log_loss_prob(self, output, labels, target, epoch_num, num_classes, annealing_step, prob):

        # print(output.shape)
        # print(target.shape)
        # print(prob.shape)
        # print(prob)
        prob = prob.reshape(-1, 1)

        evidence = relu_evidence(output)
        target = target.to(self.device)

        evidence = evidence * prob
        
        target = target

        alpha = evidence + 1
        edl = self.edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step
        )
        # print(edl.shape)
        loss = torch.mean(
              edl
        )
        return loss

    def edl_log_loss_entropy(self, output, labels, target, epoch_num, num_classes, annealing_step, prob):

        # prob = prob.reshape(-1, 1)
        ce_loss = F.cross_entropy(output, labels, reduction="none")


        evidence = relu_evidence(output)

        temp = evidence[torch.arange(
            evidence.size(0)), labels.long()] + ce_loss
        evidence.scatter(1, labels.reshape(-1, 1).long(), temp.reshape(-1, 1))

        target = target.to(self.device)

        # target = target * prob
        alpha = evidence + 1
        edl = self.edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step
        )
        # print(edl.shape)
        loss = torch.mean(
            edl
        )
        return loss