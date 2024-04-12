from __future__ import print_function

import argparse
import os.path
import time

import fitlog
import torch

# fitlog.commit(__file__)

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net
from utils import *
from ChannelAug import ChannelAdapGray, ChannelRandomErasing
from util.board_writter import BoardWriter
import datetime
from supconstrast_losses import SupConLoss
from apex import amp
import wandb
from calibrator.select import select_calibrator
from datetime import timedelta
from wandb import AlertLevel


parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume-net1', default='', type=str,
                    help='resume net1 from checkpoint')
parser.add_argument('--resume-net2', default='', type=str,
                    help='resume net2 from checkpoint')
parser.add_argument('--model_path', default='./save_model/contrastive/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='robust', type=str,
                    metavar='m', help='method type: base or agw or robust')
parser.add_argument('--loss1', default='sid', type=str, help='loss type: id or soft id')
parser.add_argument('--loss2', default='robust_tri', type=str,
                    metavar='m', help='loss type: wrt or adp or robust_tri')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--epoch', default=50, type=int, help='random seed')
parser.add_argument('--gpu', default='3', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--savename', default='release_sysu_dart_nr20', type=str,
                    help='name of the saved model')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--augc', default=1, type=int,
                    metavar='aug', help='use channel aug or not')
parser.add_argument('--rande', default=0.5, type=float,
                    metavar='ra', help='use random erasing or not and the probability')
parser.add_argument('--kl', default=0, type=float,
                    metavar='kl', help='use kl loss and the weight')
parser.add_argument('--alpha', default=1, type=int,
                    metavar='alpha', help='magnification for the hard mining')
parser.add_argument('--gamma', default=1, type=int,
                    metavar='gamma', help='gamma for the hard mining')
parser.add_argument('--square', default=1, type=int,
                    metavar='square', help='gamma for the hard mining')
parser.add_argument('--noise-mode', default='sym', type=str, help='sym')
parser.add_argument('--noise-rate', default=0.2, type=float,
                    metavar='nr', help='noise_rate')
parser.add_argument('--data-path', default='../dataset/SYSU-MM01', type=str, help='path to dataset')
parser.add_argument('--p-threshold', default=0.1, type=float, help='clean probability threshold')
parser.add_argument('--warm-epoch', default=1, type=int, help='epochs for net warming up')
parser.add_argument('--print-prob', default=False, type=bool, help='print prob value')
parser.add_argument('--eval-method', default='gmm', type=str, help='noisy label process')
parser.add_argument('--comment', default="", type=str, help='comment message')
parser.add_argument('--loss3', default="", type=str, help='comment message')
parser.add_argument('--debias', default="none", type=str, help='contrastive debias')
parser.add_argument('--tau_plus', default=0.1, type=float, help='tau_plus')
parser.add_argument('--sim-method', default="dist", type=str, help='sim_method')
parser.add_argument('--weighted_softmax', default=False, type=bool, help='weighted_softmax')

args = parser.parse_args()

wandb.init(
    config=args,
    project="dart"
)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
set_seed(args.seed)

dataset = args.dataset
data_path = args.data_path

if dataset == 'sysu':
    test_mode = [1, 2]  # thermal to visible
elif dataset == 'regdb':
    test_mode = [2, 1]  # visible to thermal

timestamp = datetime.datetime.strftime(datetime.datetime.now(), '%m%d%H%M%S')
checkpoint_path = os.path.join(args.model_path, args.savename + "_" + timestamp + "/")
print("checkpoint_path ", checkpoint_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

tensorboard_dir = checkpoint_path.replace('save_model', 'tensorboard_dir')
logs_dir = checkpoint_path.replace('save_model', 'logs')
sys.stdout = Logger(osp.join(logs_dir, args.savename + '_log.txt'))
BoardWriter.setWriter(BoardWriter, tensorboard_dir)
fitlog_dir = "./fitlogs"
# fitlog.set_log_dir(fitlog_dir)
# fitlog.add_hyper(args)

print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train_list = [
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize]

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize])

if args.rande > 0:
    transform_train_list = transform_train_list + [ChannelRandomErasing(probability=args.rande)]

if args.augc == 1:
    transform_train_list = transform_train_list + [ChannelAdapGray(probability=0.5)]

transform_train = transforms.Compose(transform_train_list)

end = time.time()
if dataset == 'sysu':
    # evaltrain set
    evaltrainset = SYSUData(data_path,
                            transform=transform_test,
                            noise_rate=args.noise_rate,
                            noise_file='%s/%.1f_%s' % (args.data_path, args.noise_rate, args.noise_mode),
                            mode='evaltrain')
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(evaltrainset.train_color_label, evaltrainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    evaltrainset = RegDBData(data_path,
                             trial=args.trial,
                             transform=transform_test,
                             noise_rate=args.noise_rate,
                             noise_file='%s/%.1f_%s' % (args.data_path, args.noise_rate, args.noise_mode),
                             mode='evaltrain')
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(evaltrainset.train_color_label, evaltrainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(evaltrainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(evaltrainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(evaltrainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')

net1 = embed_net(n_class, no_local='on', gm_pool='on', arch=args.arch)
net2 = embed_net(n_class, no_local='on', gm_pool='on', arch=args.arch)
net1.to(device)
net2.to(device)

cudnn.benchmark = True

# define loss function
criterion_id = nn.CrossEntropyLoss()
criterion_CE = nn.CrossEntropyLoss(reduction='none')

supcon = SupConLoss()

ignored_params1 = list(map(id, net1.bottleneck.parameters())) \
                  + list(map(id, net1.classifier.parameters()))
base_params1 = filter(lambda p: id(p) not in ignored_params1, net1.parameters())
optimizer1 = optim.SGD([
    {'params': base_params1, 'lr': 0.1 * args.lr},
    {'params': net1.bottleneck.parameters(), 'lr': args.lr},
    {'params': net1.classifier.parameters(), 'lr': args.lr}],
    weight_decay=5e-4, momentum=0.9, nesterov=True)

ignored_params2 = list(map(id, net2.bottleneck.parameters())) \
                  + list(map(id, net2.classifier.parameters()))
base_params2 = filter(lambda p: id(p) not in ignored_params2, net2.parameters())
optimizer2 = optim.SGD([
    {'params': base_params2, 'lr': 0.1 * args.lr},
    {'params': net2.bottleneck.parameters(), 'lr': args.lr},
    {'params': net2.classifier.parameters(), 'lr': args.lr}],
    weight_decay=5e-4, momentum=0.9, nesterov=True)

if len(args.resume_net1) > 0:
    model_path1 = os.path.join(args.model_path, args.resume_net1)
    model_path2 = os.path.join(args.model_path, args.resume_net2)

    if os.path.isfile(model_path1):
        print('==> loading checkpoint {} and {}'.format(args.resume_net1, args.resume_net2))
        checkpoint1 = torch.load(model_path1)
        checkpoint2 = torch.load(model_path2)
        net1.load_state_dict(checkpoint1['net'])
        net2.load_state_dict(checkpoint2['net'])
        start_epoch = checkpoint1['epoch'] + 1
        optimizer1.load_state_dict(checkpoint1['optimizer'])
        optimizer2.load_state_dict(checkpoint2['optimizer'])
        print('==> loaded checkpoint {} (epoch {}, mAP {}), {} (epoch {}, mAP {})'
              .format(args.resume_net1, checkpoint1['epoch'], checkpoint1['mAP'],
                      args.resume_net2, checkpoint2['epoch'], checkpoint2['mAP']))
    else:
        print('==> no checkpoint found at {} and {}'.format(args.resume_net1, args.resume_net2))
net1, optimizer1 = amp.initialize(net1, optimizer1, opt_level="O1")
net2, optimizer2 = amp.initialize(net2, optimizer2, opt_level="O1")

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr


def warmup(epoch, net, optimizer, dataloader):
    current_lr = adjust_learning_rate(optimizer, 0)
    net.train()

    num_iter = (len(dataloader.dataset.cIndex) // dataloader.batch_size) + 1
    for batch_idx, (input10, input11, input2, label1, label2) in enumerate(dataloader):
        labels = torch.cat((label1, label1, label2), 0)
        input1 = torch.cat((input10, input11,), 0)

        input1 = input1.cuda()
        input2 = input2.cuda()
        labels = labels.cuda()

        _, out0, = net(input1, input2)
        loss_id = criterion_id(out0, labels)

        optimizer.zero_grad()
        with amp.scale_loss(loss_id, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f\t Current-lr: %.4f'
                  % (args.dataset, args.noise_rate, args.noise_mode, epoch, 80, batch_idx + 1,
                     num_iter, loss_id.item(), current_lr))


def train(epoch, net, optimizer, trainloader):
    current_lr = adjust_learning_rate(optimizer, epoch - args.warm_epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    id_correct = 0
    total = 0

    net.train()
    end = time.time()

    for batch_idx, (input10, input11, input2, label1, label2, true_label1, true_labels2, probV_1, probV_2,
                    probI, idx_v, idx_i) in enumerate(trainloader):

        label1 = label1.cuda()
        label2 = label2.cuda()
        labels = torch.cat((label1, label1, label2), 0)
        input1 = torch.cat((input10, input11,), 0)
        prob = torch.cat((probV_1, probV_2, probI), 0)
        input1 = input1.cuda()
        input2 = input2.cuda()
        labels = labels.cuda()
        prob = prob.cuda()
        data_time.update(time.time() - end)

        feat, out0 = net(input1, input2)

        if args.loss1 == 'sid':
            loss_id = criterion_CE(out0, labels)
            loss_id = prob * loss_id
            loss_id = loss_id.sum() / prob.size(0)
        else:
            loss_id = criterion_id(out0, labels)
        BoardWriter.boardWriter.add_scalar('loss_id', loss_id, epoch)
        if args.loss2 == 'weighted_crosscon':
            sup_loss1 = supcon.weighted_con_loss(feat[:trainloader.batch_size], feat[trainloader.batch_size * 2:],
                                             prob_v=prob[:trainloader.batch_size],
                                             prob_i=prob[trainloader.batch_size * 2:], labels=label1, debias="none",
                                                 tau_plus=args.tau_plus, sim_method=args.sim_method, weighted_softmax=args.weighted_softmax)
            sup_loss2 = supcon.weighted_con_loss(feat[trainloader.batch_size:trainloader.batch_size * 2],
                                             feat[trainloader.batch_size * 2:],
                                             prob_v=prob[trainloader.batch_size:trainloader.batch_size * 2],
                                             prob_i=prob[trainloader.batch_size * 2:], labels=label1, debias="none",
                                                 tau_plus=args.tau_plus, sim_method=args.sim_method, weighted_softmax=args.weighted_softmax)
            sup_loss3 = supcon.weighted_con_loss(feat[:trainloader.batch_size],
                                             feat[trainloader.batch_size: trainloader.batch_size * 2],
                                             prob_v=prob[:trainloader.batch_size],
                                             prob_i=prob[trainloader.batch_size: trainloader.batch_size * 2],
                                             labels=label1, debias="none",
                                                 tau_plus=args.tau_plus, sim_method=args.sim_method, weighted_softmax=args.weighted_softmax)
            BoardWriter.boardWriter.add_scalar('sup_loss1', sup_loss1, epoch)
            BoardWriter.boardWriter.add_scalar('sup_loss2', sup_loss2, epoch)
            BoardWriter.boardWriter.add_scalar('sup_loss3', sup_loss3, epoch)
            loss = loss_id + (sup_loss1 + sup_loss2 + sup_loss3) / 10
        elif args.loss2 == "none":
            loss = loss_id

        else:
            print("loss2 is wrong")
            exit(2)

        if  args.loss3 == 'graph_con_feat_progressive' and epoch > 2:
            center_f_V = trainloader.dataset.centers_f_V[idx_v].cuda()
            center_f_I = trainloader.dataset.centers_f_I[idx_i].cuda()

            cross_center_loss1 = supcon.weighted_con_loss(feat[:trainloader.batch_size], center_f_I
                                                          , prob_v=~prob[:trainloader.batch_size]
                                                          , prob_i=~prob[:trainloader.batch_size]
                                                          , labels=label1, debias=args.debias,
                                                 tau_plus=args.tau_plus, sim_method=args.sim_method, weighted_softmax=args.weighted_softmax)
            cross_center_loss2 = supcon.weighted_con_loss(feat[trainloader.batch_size:trainloader.batch_size * 2],
                                                          center_f_I
                                                          , prob_v=~prob[
                                                                    trainloader.batch_size:trainloader.batch_size * 2]
                                                          , prob_i=~prob[
                                                                    trainloader.batch_size:trainloader.batch_size * 2]
                                                          , labels=label1, debias=args.debias,
                                                 tau_plus=args.tau_plus, sim_method=args.sim_method, weighted_softmax=args.weighted_softmax)
            cross_center_loss3 = supcon.weighted_con_loss(feat[trainloader.batch_size * 2:], center_f_V
                                                          , prob_v=~prob[trainloader.batch_size * 2:]
                                                          , prob_i=~prob[trainloader.batch_size * 2:]
                                                          , labels=label1, debias=args.debias,
                                                 tau_plus=args.tau_plus, sim_method=args.sim_method, weighted_softmax=args.weighted_softmax)
            BoardWriter.boardWriter.add_scalar('cross_center_loss1', cross_center_loss1, epoch)
            BoardWriter.boardWriter.add_scalar('cross_center_loss2', cross_center_loss2, epoch)
            BoardWriter.boardWriter.add_scalar('cross_center_loss3', cross_center_loss3, epoch)
            loss = loss + (cross_center_loss1 + cross_center_loss2 + cross_center_loss3) / 10

        else:
            pass
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # loss.backward()
        optimizer.step()

        # update
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        total = total + labels.size(0)

        _, predicted = out0.max(1)
        id_correct = id_correct + predicted.eq(labels).sum().item()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.avg:.3f} '
                  'lr:{:.3f} '
                  'Loss: {train_loss.avg:.4f} '
                  'iLoss: {id_loss.avg:.4f} '
                  'TLoss: {tri_loss:.4f} '
                  'id-acc: {:.2f} '
                  'tri-acc: {:.2f} '.format(
                epoch, batch_idx, len(trainloader), current_lr,
                100. * id_correct / total, 100. * 1 / 1,
                batch_time=batch_time, train_loss=train_loss, id_loss=id_loss, tri_loss=1))


def test(net1, net2):
    # switch to evaluation mode
    net1.eval()
    net2.eval()
    ptr = 0
    gall_feat_att = np.zeros((ngall, 2048))

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = input.cuda()
            _, feat_att1 = net1(input, input, test_mode[0])
            _, feat_att2 = net2(input, input, test_mode[0])
            feat_att = (feat_att1 + feat_att2) / 2.
            gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num

    # switch to evaluation
    net1.eval()
    net2.eval()
    ptr = 0
    query_feat_att = np.zeros((nquery, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = input.cuda()
            _, feat_att1 = net1(input, input, test_mode[1])
            _, feat_att2 = net2(input, input, test_mode[1])
            feat_att = (feat_att1 + feat_att2) / 2.
            query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num

    # compute the similarity
    distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))

    # evaluation
    if dataset == 'regdb':
        cmc_att, mAP_att, mINP_att = eval_regdb(-distmat_att, query_label, gall_label)
    elif dataset == 'sysu':
        cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam)

    return cmc_att, mAP_att, mINP_att

def main():
    # training
    if args.dataset == 'sysu':
        warmupset = SYSUData(data_path, transform=transform_train, noise_rate=args.noise_rate, mode='warmup',
                             noise_file='%s/%.1f_%s' % (args.data_path, args.noise_rate, args.noise_mode))
        trainset = SYSUData(data_path, transform=transform_train, noise_rate=args.noise_rate, mode='train',
                            noise_file='%s/%.1f_%s' % (args.data_path, args.noise_rate, args.noise_mode))
    elif args.dataset == 'regdb':
        warmupset = RegDBData(data_path, trial=args.trial, transform=transform_train, noise_rate=args.noise_rate,
                              noise_file='%s/%.1f_%s' % (args.data_path, args.noise_rate, args.noise_mode),
                              mode='warmup')
        trainset = RegDBData(data_path, trial=args.trial, transform=transform_train, noise_rate=args.noise_rate,
                             noise_file='%s/%.1f_%s' % (args.data_path, args.noise_rate, args.noise_mode),
                             mode='train')

    print('==> Start Training...')
    best_mAP = 0

    for epoch in range(start_epoch, args.epoch):

        print('==> Preparing Data Loader...')
        loader_batch = args.batch_size * args.num_pos

        if epoch < args.warm_epoch:
            warm_sampler = AllSampler(args.dataset, warmupset.train_color_label, warmupset.train_thermal_label)

            warmupset.cIndex = warm_sampler.index1  # color index
            warmupset.tIndex = warm_sampler.index2  # thermal index

            warmup_trainloader = data.DataLoader(warmupset, batch_size=loader_batch, sampler=warm_sampler, \
                                                 num_workers=args.workers, drop_last=True)

            print('Warmup Net1')
            warmup(epoch, net1, optimizer1, warmup_trainloader)

            print('\nWarmup Net2')
            warmup(epoch, net2, optimizer2, warmup_trainloader)
            print('\n')
        else:
            # models eval
            eval_sampler = AllSampler(args.dataset, evaltrainset.train_color_label, evaltrainset.train_thermal_label)

            evaltrainset.cIndex = eval_sampler.index1  # color index
            evaltrainset.tIndex = eval_sampler.index2  # thermal index

            eval_loader = data.DataLoader(evaltrainset,
                                          batch_size=loader_batch,
                                          sampler=eval_sampler,
                                          num_workers=args.workers,
                                          drop_last=True)

            cal = select_calibrator(criterion_CE, logs_dir, args)


            prob_A_V, prob_A_I, prob_B_V, prob_B_I = cal.dual_eval_train(net1, net2, eval_loader, evaltrainset,
                                                                                 epoch, trainset)


            noisy_sample_rgb = ~prob_A_V
            noisy_sample_ir = ~prob_A_I

            noisy_sample_rgb_index1 = noisy_sample_rgb[:int(len(noisy_sample_rgb) / 2)]
            clean_sample_rgb_index1 = ~noisy_sample_rgb_index1
            noisy_sample_rgb_index2 = noisy_sample_rgb[int(len(noisy_sample_rgb) / 2):]
            clean_sample_rgb_index2 = ~noisy_sample_rgb_index2

            noisy_rgb_count1 = noisy_sample_rgb_index1.sum()
            clean_rgb_count1 = clean_sample_rgb_index1.sum()
            noisy_rgb_count2 = noisy_sample_rgb_index2.sum()
            clean_rgb_count2 = clean_sample_rgb_index2.sum()
            noisy_ir_count = noisy_sample_ir.sum()
            clean_ir_count = (~noisy_sample_ir).sum()

            TCL_rgb1 = (evaltrainset.true_train_color_label[noisy_sample_rgb_index1] == evaltrainset.train_color_label[
                noisy_sample_rgb_index1]).sum()
            FCL_rgb1 = (evaltrainset.true_train_color_label[clean_sample_rgb_index1] != evaltrainset.train_color_label[
                clean_sample_rgb_index1]).sum()
            TCL_rgb2 = (evaltrainset.true_train_color_label[noisy_sample_rgb_index2] == evaltrainset.train_color_label[
                noisy_sample_rgb_index2]).sum()
            FCL_rgb2 = (evaltrainset.true_train_color_label[clean_sample_rgb_index2] != evaltrainset.train_color_label[
                clean_sample_rgb_index2]).sum()
            TCL_ir = (evaltrainset.true_train_thermal_label[noisy_sample_ir] ==
                      evaltrainset.train_thermal_label[noisy_sample_ir]).sum()
            FCL_ir = (evaltrainset.true_train_thermal_label[~noisy_sample_ir] !=
                      evaltrainset.train_thermal_label[~noisy_sample_ir]).sum()
            print("TCL_rgb1", TCL_rgb1)
            print("FCL_rgb1", FCL_rgb1)
            print("TCL_rgb2", TCL_rgb2)
            print("FCL_rgb2", FCL_rgb2)
            print("TCL_ir", TCL_ir)
            print("FCL_ir", FCL_ir)
            print("noisy_rgb_count1", noisy_rgb_count1)
            print("clean_rgb_count1", clean_rgb_count1)
            print("noisy_rgb_count2", noisy_rgb_count2)
            print("clean_rgb_count2", clean_rgb_count2)
            print("noisy_ir_count", noisy_ir_count)
            print("clean_ir_count", clean_ir_count)


            print('Train Net1')
            #  Co-training
            trainset.probV_1, trainset.probV_2, trainset.probI = prob_B_V[0:int(len(prob_B_V) / 2)], \
                                                                 prob_B_V[int(len(prob_B_V) / 2):], \
                                                                 prob_B_I

            train_sampler = IdentitySampler(trainset.train_color_label, trainset.train_thermal_label,
                                            color_pos, thermal_pos, args.num_pos, args.batch_size, epoch)

            trainset.cIndex = train_sampler.index1  # color index
            trainset.tIndex = train_sampler.index2  # thermal index

            trainloader = data.DataLoader(
                dataset=trainset,
                batch_size=loader_batch,
                num_workers=args.workers,
                sampler=train_sampler,
                drop_last=True)

            # train net1
            train(epoch, net1, optimizer1, trainloader)

            print('\nTrain Net2')
            trainset.probV_1, trainset.probV_2, trainset.probI = prob_A_V[0:int(len(prob_A_V) / 2)], prob_A_V[int(
                len(prob_A_V) / 2):], prob_A_I

            train_sampler = IdentitySampler(trainset.train_color_label, trainset.train_thermal_label, \
                                            color_pos, thermal_pos, args.num_pos, args.batch_size, epoch)

            trainset.cIndex = train_sampler.index1  # color index
            trainset.tIndex = train_sampler.index2  # thermal index

            trainloader = data.DataLoader(
                dataset=trainset,
                batch_size=loader_batch,
                num_workers=args.workers,
                sampler=train_sampler,
                drop_last=True)

            # train net2
            train(epoch, net2, optimizer2, trainloader)
            print('\n')

        if (epoch % 3 == 0 and epoch != 0) or epoch == 99:
            print('Test Epoch: {}'.format(epoch))
            flag = False
            with torch.no_grad():
                # testing
                cmc_att, mAP_att, mINP_att = test(net1, net2)
                BoardWriter.boardWriter.add_scalar('mAP', mAP_att, epoch)
                # fitlog.add_metric({"mAP": mAP_att}, step=epoch)
                wandb.log({"mAP": mAP_att})
                if best_mAP < mAP_att:
                    flag = True
                    best_mAP = mAP_att
                    # fitlog.add_best_metric({"mAP": mAP_att})
                mAP_str = '{:.2f}'.format(mAP_att * 100)
                wandb.alert(
                    title='Best mAP',
                    text=f'epoch is {epoch}, '
                         f'dataset is {args.dataset}, '
                         f'noise_rate is {args.noise_rate}, '
                         f'lr is {args.lr}, '
                         f'best mAP is {mAP_str} ',
                    level=AlertLevel.INFO,
                    wait_duration=timedelta(minutes=5)
                )

                # save model
                state = {
                    'net': net1.state_dict(),
                    'cmc': cmc_att,
                    'mAP': mAP_att,
                    'mINP': mINP_att,
                    'epoch': epoch,
                    'optimizer': optimizer1.state_dict()
                }
                state2 = {
                    'net': net2.state_dict(),
                    'cmc': cmc_att,
                    'mAP': mAP_att,
                    'mINP': mINP_att,
                    'epoch': epoch,
                    'optimizer': optimizer2.state_dict()
                }
                if args.dataset == 'sysu':
                    torch.save(state, checkpoint_path + args.savename + "_" + str(epoch) + '_net1.t')
                    torch.save(state2, checkpoint_path + args.savename + "_" + str(epoch) + '_net2.t')
                    if flag:
                        torch.save(state, checkpoint_path + args.savename + "_" + "best" + '_net1.t')
                        torch.save(state2, checkpoint_path + args.savename + "_" + "best" + '_net2.t')

                else:
                    torch.save(state, checkpoint_path + args.savename + "_" + str(epoch) + '_trial{}'.format(args.trial) +
                               '_net1.t')
                    torch.save(state2, checkpoint_path + args.savename + "_" + str(epoch) + '_trial{}'.format(args.trial) +
                               '_net2.t')
                    if flag:
                        torch.save(state,
                                   checkpoint_path + args.savename + "_" + "best" + '_trial{}'.format(args.trial) +
                                   '_net1.t')
                        torch.save(state2,
                                   checkpoint_path + args.savename + "_" + "best" + '_trial{}'.format(args.trial) +
                                   '_net2.t')

            print('Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'
                  .format(cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
            
if __name__ == '__main__':
    main()