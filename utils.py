import os
import numpy as np
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
import sys
import os.path as osp
import torch
import matplotlib.pyplot as plt
import random
import collections

def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of color image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label
    

def GenIdx( train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k,v in enumerate(train_color_label) if v==unique_label_color[i]]
        color_pos.append(tmp_pos)
        
    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k,v in enumerate(train_thermal_label) if v==unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)
    return color_pos, thermal_pos
    
def GenCamIdx(gall_img, gall_label, mode):
    if mode =='indoor':
        camIdx = [1,2]
    else:
        camIdx = [1,2,4,5]
    gall_cam = []
    for i in range(len(gall_img)):
        gall_cam.append(int(gall_img[i][-10]))
    
    sample_pos = []
    unique_label = np.unique(gall_label)
    for i in range(len(unique_label)):
        for j in range(len(camIdx)):
            id_pos = [k for k,v in enumerate(gall_label) if v==unique_label[i] and gall_cam[k]==camIdx[j]]
            if id_pos:
                sample_pos.append(id_pos)
    return sample_pos
    
def ExtractCam(gall_img):
    gall_cam = []
    for i in range(len(gall_img)):
        cam_id = int(gall_img[i][-10])
        # if cam_id ==3:
            # cam_id = 2
        gall_cam.append(cam_id)
    
    return np.array(gall_cam)


class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize, epoch):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)

        N = np.maximum(len(train_color_label), len(train_thermal_label))
        iter_num = max(int(N / (batchSize * num_pos)) + 1, 300)
        for j in range(iter_num):
            batch_idx = np.random.choice(uni_label, batchSize, replace=False)
            for i in range(batchSize):
                try:
                    while len(color_pos[batch_idx[i]]) < 4 or len(thermal_pos[batch_idx[i]]) < 4:
                        batch_idx[i] = np.random.choice(uni_label, 1, replace=False)
                        print("re-sampling")
                except:
                    print(len(uni_label))
                    print(batch_idx)
                    print(color_pos)
                    print(thermal_pos)
                sample_color = np.random.choice(color_pos[batch_idx[i]], num_pos)
                sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], num_pos)

                if j == 0 and i == 0:
                    index1 = sample_color
                    index2 = sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))
        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
 
def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """  
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
            
def set_seed(seed, cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)

def set_requires_grad(nets, requires_grad=False):
            """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
            Parameters:
                nets (network list)   -- a list of networks
                requires_grad (bool)  -- whether the networks require gradients or not
            """
            if not isinstance(nets, list):
                nets = [nets]
            for net in nets:
                if net is not None:
                    for param in net.parameters():
                        param.requires_grad = requires_grad


class AllSampler(Sampler):
    def __init__(self, dataset, train_color_label, train_thermal_label, shuffle=True):
        N1 = len(train_color_label)
        N2 = len(train_thermal_label)
        # N = np.maximum(len(train_color_label), len(train_thermal_label))
        if dataset == 'regdb':
            index1 = np.concatenate((np.arange(N1), np.arange(20)))
            index2 = np.concatenate((np.arange(N2), np.arange(20)))
        else:
            index1 = np.concatenate((np.arange(N1), np.arange(14)))
            index2 = np.concatenate((np.arange(N2), np.arange(N1 - N2 + 14)))

        if shuffle:
            np.random.shuffle(index1)
            np.random.shuffle(index2)

        self.index1 = index1
        self.index2 = index2
        self.N = len(index1)

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

@torch.no_grad()
def generate_cluster_features(features, labels):
    centers = collections.defaultdict(list)
    for i, label in enumerate(labels):
        centers[int(labels[i])].append(features[i])

    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]

    centers = torch.stack(centers, dim=0)
    return centers

def relu_evidence(y):
    return F.relu(y)


def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]


def class_sim_eval(features, labels):
    centers = generate_cluster_features(features, labels)
    centers = centers / torch.norm(centers, dim=-1, keepdim=True)
    feat = features / torch.norm(features, dim=-1, keepdim=True)
    class_sim = torch.mm(feat, centers.T)
    #     class_sim = F.cosine_similarity(features, centers, dim=-1)
    #     instance_sim = F.cosine_similarity(features, features, dim=-1)
    return class_sim

def instance_sim_eval(features1, features2):
    feat1 = features1 / torch.norm(features1, dim=-1, keepdim=True)
    feat2 = features2 / torch.norm(features2, dim=-1, keepdim=True)
    instance_sim = torch.mm(feat1, feat2.T)

    return instance_sim