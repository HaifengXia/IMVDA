import os, sys, argparse, re
import random, time
import warnings
warnings.filterwarnings('ignore')
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import *
from utils.helpers import print_log
from models.model import resnet50, model_init
from models.unimodel import uniresnet50, unimodel_init
from models.classifier import feat_classifier
from utils.datasets import create_loaders
from utils.AligenLoss import AligenLoss, Attention


def create_network(num_classes, num_parallel, bn_threshold, gpu):
    MultiNet = resnet50(num_parallel, num_classes, bn_threshold)
    UniNet = uniresnet50(num_classes)
    classifier = feat_classifier(num_classes)
    att_net = Attention(256, 64)
    assert(torch.cuda.is_available())
    MultiNet.to(gpu[0])
    UniNet.to(gpu[0])
    classifier.to(gpu[0])
    att_net.to(gpu[0])
    MultiNet = torch.nn.DataParallel(MultiNet, gpu)
    UniNet = torch.nn.DataParallel(UniNet, gpu)
    classifier = torch.nn.DataParallel(classifier, gpu)
    att_net = torch.nn.DataParallel(att_net, gpu)

    return MultiNet, UniNet, classifier, att_net


def create_optimizers(param_group):
    """Create optimisers for MultiNet, UniNet and classifier"""
    optimizer = torch.optim.SGD(param_group)

    return optimizer

def lr_schedule(optimizer, iter_num, max_iter, gamma=10, power=0.75):

    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def L1_penalty(var):
    return torch.abs(var).sum()

def L2_penalty(var):
    return torch.square(var).sum()

def train(MultiNet, UniNet, classifier, att_net, source_loader, target_loader, test_loader, optimizer_s, optimizer_t,
          freeze_bn, slim_params, compact_params, lamda, args):

    MultiNet.train()
    UniNet.train()
    classifier.train()
    att_net.train()
    input_types = ['rgb', 'depth']

    if freeze_bn:
        for module in MultiNet.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    len_source = len(source_loader)
    len_target = len(target_loader)
    num_iters = 50 * max(len_source, len_target)
    for i in range(num_iters):
        if i % min(len_source, len_target) == 0:
            UniNet.eval()
            classifier.eval()
            acc = evaluation(test_loader, UniNet, classifier)
            pred_labels = obtain_label(test_loader, UniNet, classifier, args)
            pred_labels = torch.from_numpy(pred_labels).cuda().long()
            log_str = 'Iter:{}/{}; tgt_acc = {:.2f}%;'.format(i, num_iters, acc)
            res_file = '../results/res_' + args.tgt_name + '.txt'
            with open(res_file, 'a') as res:
                res.write(log_str+'\n')
            res.close()
            print(log_str + '\n')
        lr_schedule(optimizer_s, iter_num=i, max_iter=num_iters)
        lr_schedule(optimizer_t, iter_num=i, max_iter=num_iters)
        if i % len_source == 0:
            iter_source = iter(source_loader)
        if i % len_target == 0:
            iter_target = iter(target_loader)

        sample = iter_source.next()
        input_rgbd = [sample[key].cuda().float() for key in input_types]
        input_source = sample['rgb'].cuda().float()
        labels = sample['target'].cuda().long()
        input_target, _, idx = iter_target.next()
        input_target = input_target.cuda()

        # Compute outputs
        feat_rgb, feat_depth = MultiNet(input_rgbd)
        x_rgb, fusion_rgbd, out_rgbd = classifier(feat_rgb, feat_depth)

        feat_src = UniNet(input_source)
        x_src, fusion_src, out_src = classifier(feat_src)

        feat_tgt = UniNet(input_target)
        x_tgt, fusion_tgt, out_tgt = classifier(feat_tgt)

        #loss functions
        ### loss function for source model
        rgbd_cls_loss = nn.CrossEntropyLoss()(out_rgbd, labels)
        L2_norm = sum([L2_penalty(m).cuda() for m in compact_params])
        L1_norm = sum([L1_penalty(m).cuda() for m in slim_params])
        loss_source = rgbd_cls_loss + 0.2 * (L2_norm + L1_norm)

        optimizer_s.zero_grad()
        loss_source.backward()
        optimizer_s.step()

        ### loss function for target model
        src_cls_loss = nn.CrossEntropyLoss()(out_src, labels)
        rgb_rec_loss = torch.norm((x_src - x_rgb.detach()).abs(), 2, 1).sum() / float(args.train_bs)
        fuse_rec_loss = torch.norm((fusion_rgbd.detach() - fusion_src).abs(), 2, 1).sum() / float(args.train_bs)
        rec_loss = rgb_rec_loss + fuse_rec_loss
        lam = 2 / (1 + math.exp(-1 * 10 * int(i/min(len_source, len_target))/ 50)) - 1
        align_loss = AligenLoss(fusion_src, fusion_tgt, att_net)
        tgt_cls_loss = nn.CrossEntropyLoss()(out_tgt, pred_labels[idx])

        loss_target = src_cls_loss + 0.4 * rec_loss + lam * (align_loss + tgt_cls_loss)

        optimizer_t.zero_grad()
        loss_target.backward()
        optimizer_t.step()

from scipy.spatial.distance import cdist
def obtain_label(loader, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            # feas = netB(inputs)
            _, feas, outputs = netC(netB(inputs))
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    prob_t, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(5):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    return pred_label.astype('int')

def evaluation(loader, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            _, _, outputs = netC(netB(inputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    return accuracy * 100

def get_arguments():
    parser = argparse.ArgumentParser(description='Full Pipeline Training')
    parser.add_argument('--random_seed', type=int, default=2021,
                        help='random seed for initialization.')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0],
                        help='select gpu.')
    parser.add_argument('--num_classes', type=int, default=0,
                        help='number of classes.')
    parser.add_argument('--num_parallel', type=int, default=2,
                        help='number of modalities.')
    parser.add_argument('--bn_threshold', type=float, default=2e-2,
                        help='filter out lower bn')
    parser.add_argument('--print-network', action='store_true', default=False,
                        help='Whether print newtork paramemters.')
    parser.add_argument('--train_bs', type=int, default=32,
                        help='train batch size.')
    parser.add_argument('--test_bs', type=int, default=64,
                        help='test batch size.')
    parser.add_argument('--freeze-bn', type=bool, nargs='+', default=True,
                        help='Whether to keep batch norm statistics intact.')
    parser.add_argument('--lamda', type=float, default=2e-4,
                        help='control the sparse')
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    return parser.parse_args()


def main(rgbd_file, rgb_file, tgt_name, set_name):
    global args
    args = get_arguments()
    args.out_file = open('./results/res.txt', 'w')
    args.tgt_name = tgt_name
    if set_name == 'RGBD2OfficeHome':
        args.num_classes = 13
    elif set_name == 'RGBD2Office31':
        args.num_classes = 8
    elif set_name == 'B3DO2OfficeHome':
        args.num_classes = 14
    elif set_name == 'B3DO2Office31':
        args.num_classes = 27
    elif set_name == 'RGBD2Caltech':
        args.num_classes = 10
    else:
        print('--------------Wrong----------------')

    # Set random seeds
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # Generate Segmenter
    torch.cuda.set_device(args.gpu[0])
    MultiNet, UniNet, classifier, att_net = create_network(args.num_classes, args.num_parallel,
                                                  args.bn_threshold, args.gpu)

    param_group_s, param_group_t = [], []
    slim_params, compact_params = [], []

    args.lr = 1e-3
    for name, param in MultiNet.named_parameters():
        param_group_s += [{'params': param, 'lr': args.lr}]
        if args.print_network:
            print_log(' MultiNet. parameter: {}'.format(name))

        if param.requires_grad and name.endswith('weight') and 'bn2' in name:
            compact_params.append(param[:len(param) // 2])
            slim_params.append(param[len(param) // 2:])

    if args.print_network:
        print_log('')

    for name, param in UniNet.named_parameters():
        param_group_t += [{'params': param, 'lr': args.lr}]
        if args.print_network:
            print_log(' UniNet. parameter: {}'.format(name))
    if args.print_network:
        print_log('')

    for name, param in classifier.named_parameters():
        param_group_s += [{'params': param, 'lr': args.lr * 10}]
        param_group_t += [{'params': param, 'lr': args.lr * 10}]
        if args.print_network:
            print_log(' classifier. parameter: {}'.format(name))

    for name, param in att_net.named_parameters():
        param_group_t += [{'params': param, 'lr': args.lr * 10}]
        if args.print_network:
            print_log(' att_net. parameter: {}'.format(name))

    if args.print_network:
        print_log('')

    MultiNet = model_init(MultiNet, 2)
    UniNet = unimodel_init(UniNet)

    source_loader, target_loader, test_loader = create_loaders(rgbd_file, rgb_file,
                                                               args.train_bs, args.test_bs)

    # create optimizers
    optimizer_s = create_optimizers(param_group_s)
    optimizer_t = create_optimizers(param_group_t)

    #train models
    train(MultiNet, UniNet, classifier, att_net, source_loader, target_loader, test_loader, optimizer_s, optimizer_t,
          args.freeze_bn, slim_params, compact_params, args.lamda, args)


if __name__ == '__main__':
    root = '../text_root/'
    datasets = ['RGBD2OfficeHome']
    domains = {'RGBD2OfficeHome': ['Product']}
    for set in datasets:
        for domain in domains[set]:
            rgbd_file = root + set + '/' + 'rgbd.txt'
            rgb_file = root + set + '/' + domain + '.txt'
            main(rgbd_file, rgb_file, domain, set)