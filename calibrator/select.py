import faiss
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.nn.models.label_prop import LabelPropagation


class select_calibrator():

    def __init__(self, criterion_CE, logs_dir, args):
        self.criterion_CE = criterion_CE
        self.logs_dir = logs_dir
        self.args = args
        if args.dataset == "sysu":
            self.n = 395
        elif args.dataset == "regdb":
            self.n = 206
        else:
            print("dataset is wrong")
            exit(1)

    def dual_eval_train(self, net1, net2, dataloader, evaltrainset, epoch, trainset):
        print("select dual train")

        with torch.no_grad():
            features_V1_A = -1. * torch.ones(len(evaltrainset.train_color_label), self.n)
            features_V2_A = -1. * torch.ones(len(evaltrainset.train_color_label), self.n)
            features_I_A = -1. * torch.ones(len(evaltrainset.train_thermal_label), self.n)
            feats_V1_A = -1. * torch.ones(len(evaltrainset.train_color_label), 2048)
            feats_V2_A = -1. * torch.ones(len(evaltrainset.train_color_label), 2048)
            feats_I_A = -1. * torch.ones(len(evaltrainset.train_thermal_label), 2048)

            features_V1_B = -1. * torch.ones(len(evaltrainset.train_color_label), self.n)
            features_V2_B = -1. * torch.ones(len(evaltrainset.train_color_label), self.n)
            features_I_B = -1. * torch.ones(len(evaltrainset.train_thermal_label), self.n)

            feats_V1_B = -1. * torch.ones(len(evaltrainset.train_color_label), 2048)
            feats_V2_B = -1. * torch.ones(len(evaltrainset.train_color_label), 2048)
            feats_I_B = -1. * torch.ones(len(evaltrainset.train_thermal_label), 2048)

            net1.train()
            net2.train()
            for batch_idx, (input10, input11, input2, label1, label2, index_V, index_I) in enumerate(dataloader):
                input1 = torch.cat((input10, input11,), 0)
                input1 = input1.cuda()
                input2 = input2.cuda()
                label1 = label1.cuda()
                label2 = label2.cuda()
                index_V = np.concatenate((index_V, index_V), 0)
                labels = torch.cat((label1, label1, label2), 0)
                feats_A, out0_A, = net1(input1, input2)
                feats_B, out0_B, = net2(input1, input2)

                out1_A = out0_A[0:dataloader.batch_size * 2]
                out2_A = out0_A[dataloader.batch_size * 2:dataloader.batch_size * 3]

                out1_B = out0_B[0:dataloader.batch_size * 2]
                out2_B = out0_B[dataloader.batch_size * 2:dataloader.batch_size * 3]

                for n1 in range(input2.size(0)):
                    features_V1_A[index_V[n1]] = out1_A[n1]
                    feats_V1_A[index_V[n1]] = feats_A[n1]
                    features_V2_A[index_V[n1 + dataloader.batch_size]] = out1_A[n1 + dataloader.batch_size]
                    feats_V2_A[index_V[n1 + dataloader.batch_size]] = feats_A[n1 + dataloader.batch_size]


                    features_V1_B[index_V[n1]] = out1_B[n1]
                    feats_V1_B[index_V[n1]] = feats_B[n1]
                    features_V2_B[index_V[n1 + dataloader.batch_size]] = out1_B[n1 + dataloader.batch_size]
                    feats_V2_B[index_V[n1 + dataloader.batch_size]] = feats_B[n1 + dataloader.batch_size]

                for n2 in range(input2.size(0)):
                    features_I_A[index_I[n2]] = out2_A[n2]
                    feats_I_A[index_I[n2]] = feats_A[n2]

                    features_I_B[index_I[n2]] = out2_B[n2]
                    feats_I_B[index_I[n2]] = feats_B[n2]


            features_V_A = torch.cat((features_V1_A, features_V2_A), 0)
            feats_V_A = torch.cat((feats_V1_A, feats_V2_A), 0)

            features_V_B = torch.cat((features_V1_B, features_V2_B), 0)
            feats_V_B = torch.cat((feats_V1_B, feats_V2_B), 0)
            labels_V_A = torch.cat(
                (torch.from_numpy(evaltrainset.train_color_label), torch.from_numpy(evaltrainset.train_color_label)), 0)
            labels_V_B = torch.cat(
                (torch.from_numpy(evaltrainset.train_color_label), torch.from_numpy(evaltrainset.train_color_label)), 0)
            labels_I_A = torch.from_numpy(evaltrainset.train_thermal_label)
            labels_I_B = torch.from_numpy(evaltrainset.train_thermal_label)

            prob_V_A, prob_I_A = self.prob_and_correct_lp(features_V_A, feats_V_A, labels_V_A, features_I_A, feats_I_A,
                                                          labels_I_A, evaltrainset, trainset, epoch)
            prob_V_B, prob_I_B = self.prob_and_correct_lp(features_V_B, feats_V_B, labels_V_B, features_I_B, feats_I_B,
                                                          labels_I_B, evaltrainset, trainset, epoch)

            del features_V1_A, features_V2_A, features_I_A, labels_I_A, features_V1_B, features_V2_B, features_I_B, labels_I_B
        return prob_V_A, prob_I_A, prob_V_B, prob_I_B

    def graph_construct(self, features_V, feats_V, features_I, feats_I, evaltrainset, trainset):
        features_V_half = features_V[:int(len(features_V) / 2)]
        feats_V_half = feats_V[:int(len(feats_V) / 2)]
        print("begin lp prob")
        faiss_index_V = faiss.IndexFlatL2(self.n)  # build the index
        faiss_index_I = faiss.IndexFlatL2(self.n)  # build the index
        print('build index')
        faiss_index_V.add(features_V_half.numpy())
        faiss_index_I.add(features_I.numpy())
        print("index done")

        dist_I, ind_I = faiss_index_I.search(features_I.numpy(), 8)
        dist_V, ind_V = faiss_index_V.search(features_V_half.numpy(), 8)

        fa_dist_I = torch.zeros([features_I.shape[0], features_I.shape[0]])
        centers_I = torch.zeros_like(features_I)
        centers_f_I = torch.zeros_like(feats_I)
        far_I = torch.zeros_like(feats_I)
        for i, ind in enumerate(ind_I):
            centers_I[i] = (features_I[ind].mean(0))
            centers_f_I[i] = (feats_I[ind].mean(0))
            far_I[i] = feats_I[ind[-1]]
            fa_dist_I[ind, i] += torch.FloatTensor(1 - dist_I[i] / dist_I.max())
            fa_dist_I[i, ind] += torch.FloatTensor(1 - dist_I[i] / dist_I.max())
            fa_dist_I[i][i] = 0
        trainset.centers_I = centers_I
        trainset.centers_f_I = centers_f_I
        trainset.far_I = far_I
        fa_dist_V = torch.zeros([features_V_half.shape[0], features_V_half.shape[0]])
        centers_V = torch.zeros_like(features_V_half)
        centers_f_V = torch.zeros_like(feats_V_half)

        far_V = torch.zeros_like(feats_V_half)
        for i, ind in enumerate(ind_V):
            centers_V[i] = (features_V_half[ind].mean(0))
            centers_f_V[i] = (feats_V_half[ind].mean(0))
            far_V[i] = feats_V_half[ind[-1]]
            fa_dist_V[ind, i] += torch.FloatTensor(1 - dist_V[i] / dist_V.max())
            fa_dist_V[i, ind] += torch.FloatTensor(1 - dist_V[i] / dist_V.max())
            fa_dist_V[i][i] = 0
        trainset.centers_V = centers_V
        trainset.centers_f_V = centers_f_V
        trainset.far_V = far_V
        sparse_adj_I = sp.coo_matrix(fa_dist_I)
        sparse_adj_V = sp.coo_matrix(fa_dist_V)

        return sparse_adj_V, sparse_adj_I

    def prob_and_correct_lp(self, features_V, feats_V, labels_V, features_I, feats_I, labels_I, evaltrainset, trainset,
                            epoch):
        sparse_adj_V, sparse_adj_I = self.graph_construct(features_V, feats_V, features_I, feats_I, evaltrainset,
                                                          trainset)

        indices_I = np.vstack((sparse_adj_I.row, sparse_adj_I.col))
        index_I = torch.LongTensor(indices_I)

        indices_V = np.vstack((sparse_adj_V.row, sparse_adj_V.col))
        index_V = torch.LongTensor(indices_V)

        lp_v = LabelPropagation(2, 0.9)
        lp_i = LabelPropagation(2, 0.9)

        with torch.no_grad():
            lp_I = lp_i(y=torch.from_numpy(evaltrainset.train_thermal_label).long(),
                        edge_index=index_I)

        new_label_I = torch.argmax(lp_I, dim=1)
        new_label_pos_I = (torch.max(lp_I, 1)[0] > self.args.p_threshold)
        prob_I = (new_label_I == torch.from_numpy(evaltrainset.train_thermal_label)) & new_label_pos_I

        print("train_thermal_label done")
        with torch.no_grad():
            lp_V = lp_v(torch.from_numpy(evaltrainset.train_color_label).long(),
                        edge_index=index_V)

        new_label_V = torch.argmax(lp_V, dim=1)
        new_label_pos_V = (torch.max(lp_V, 1)[0] > self.args.p_threshold)

        prob_V = (new_label_V == torch.from_numpy(evaltrainset.train_color_label)) & new_label_pos_V

        print("train_color_label done")

        return torch.cat((prob_V, prob_V), 0), prob_I
