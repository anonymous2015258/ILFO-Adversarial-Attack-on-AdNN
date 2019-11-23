"""
Training file for training SkipNets for supervised pre-training stage
"""

from __future__ import print_function
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch import optim
from torch import autograd
import os
import shutil
import argparse
import time
import logging
import sys
import models
from data import *
import numpy
import random
from scipy.optimize import fsolve, minimize, fmin
from PIL import Image
from datetime import datetime
import traceback

logger = logging.getLogger('train_sp')
logger.setLevel(logging.INFO)
import math

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name])
                     )


def parse_args():
    # hyper-parameters are from ResNet paper
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR10 training with gating')
    parser.add_argument('cmd', choices=['train', 'test'])
    parser.add_argument('arch', metavar='ARCH',
                        default='cifar10_rnn_gate_110',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: cifar10_rnn_gate_110)')
    parser.add_argument('--gate-type', type=str, default='ff',
                        choices=['ff', 'rnn'], help='gate type')
    parser.add_argument('--dataset', '-d', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset type')
    parser.add_argument('--thres', default='0', type=int,
                        help='threshold')
    parser.add_argument('--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4 )')
    parser.add_argument('--iters', default=10, type=int,
                        help='number of total iterations (default: 64,000)')
    parser.add_argument('--start-iter', default=0, type=int,
                        help='manual iter number (useful on restarts)')
    parser.add_argument('--batch-size', default=2, type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--resume', default='resnet-110-rnn-sp-cifar10.pth.tar', type=str,
                        help='path to  latest checkpoint (default: None)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pretrained model')
    parser.add_argument('--step-ratio', default=0.1, type=float,
                        help='ratio for learning rate deduction')
    parser.add_argument('--warm-up', action='store_true',
                        help='for n = 18, the model needs to warm up for 400 '
                             'iterations')
    parser.add_argument('--save-folder', default='save_checkpoints',
                        type=str,
                        help='folder to save the checkpoints')
    parser.add_argument('--eval-every', default=1000, type=int,
                        help='evaluate model every (default: 1000) iterations')
    parser.add_argument('--verbose', action="store_true",
                        help='print layer skipping ratio at training')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    save_path = args.save_path = os.path.join(args.save_folder, args.arch)

    if not os.path.exists(args.save_path):
        os.makedirs(save_path)
    # os.makedirs(save_path, exist_ok=True)

    # config logger file
    args.logger_file = os.path.join(save_path, 'log_{}.txt'.format(args.cmd))
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)
    logging.info('start evaluating {} with checkpoints from {}'.format(
        args.arch, args.resume))
    test_model(args)






def validate( test_loader, model):

    # switch to evaluation mode
    model.eval()

    flop_list1 = [1146752 for i in range(18)]
    flop_list2 = [96526080] + [1306368 for i in range(18)]
    flop_list3 = [33984000] + [1625600 for i in range(18)]

    flop_list = flop_list1 + flop_list2 + flop_list3

    for i, (input, target) in enumerate(test_loader):
        try:



            logging.info('i:{i}'.format(i=i))
            input_var = Variable(input, volatile=True)
            py_list = input_var.tolist()
            input_var = torch.FloatTensor(py_list)


            for j in range(2):
                probs, probs2, stacked_tensor, stacked_tensor2 = model(input_var)

                if (j == 0):
                    pr = probs
                else:
                    pr = probs2
                init_flops=0
                for i in range(len(pr)):
                    if(pr[i][0]>=0.5):
                        init_flops+=flop_list[i]

                py_list = input_var.tolist()

                create_image(py_list, i, j, "orig", 0)
                #start attack
                modifier = torch.zeros(input_var.size()).float()
                modifier_var = autograd.Variable(modifier, requires_grad=True)
                optimizer = optim.Adam([modifier_var], lr=0.0005)
                target = torch.tensor([0.5] * 54)
                target_var = autograd.Variable(target, requires_grad=False)
                min_loss = sys.float_info.max
                adv_img_min = numpy.zeros((2, 32, 32, 3))
                min_output = torch.tensor([0.0] * 54)


                for step in range(1000):
                    # perform the attack
                    loss, distance, output, adv_img = optimize(
                        optimizer,
                        model,
                        input_var,
                        modifier_var,
                        target_var, j)
                    if (loss < min_loss):
                        min_loss = loss
                        adv_img_min = adv_img
                        min_output = output

                # end attack

                new_flops = 0
                for i in range(len(min_output)):
                    if (min_output[i][0] >= 0.5):
                        new_flops += flop_list[i]
                
                time.sleep(5)
                adv_img = adv_img_min
                p_list = adv_img[j]
                img = [[[0.0 for i1 in range(3)] for j1 in range(32)] for k1 in range(32)]
                mean = [0.4914, 0.4822, 0.4465]
                std = [0.2023, 0.1994, 0.2010]

                for i1 in range(32):
                    for j1 in range(32):
                        p1 = int((p_list[i1][j1][0] * std[0] + mean[0]) * 255)
                        p2 = int((p_list[i1][j1][1] * std[1] + mean[1]) * 255)
                        p3 = int((p_list[i1][j1][2] * std[2] + mean[2]) * 255)
                        l = [p1, p2, p3]

                        img[i1][j1] = l

                img_arr = numpy.asarray(img)
                new_im = Image.fromarray(img_arr.astype('uint8'), mode='RGB')
                inc_val = float(new_flops-init_flops)/float(init_flops)
                new_im.save(
                    "output/img_" + str(i) + "_" + str(j) + "_"  + "ILFO" + "_" + str(inc_val) + ".png")

        except Exception as e:
            print(e)
            print('print_exc():')
            traceback.print_exc(file=sys.stdout)
        continue








def test_model(args):
    # create model
    # print('models dict-->', models)
    model = models.__dict__[args.arch](args.pretrained)
    print('model in use-->', model)
    model = torch.nn.DataParallel(model)
    # model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_iter = checkpoint['iter']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (iter: {})'.format(
                args.resume, checkpoint['iter']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = False
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)
    # logging.info('val_loader `{}`'.format(type(val_loader)))

    criterion = nn.CrossEntropyLoss()  # .cuda()

    validate(test_loader, model)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        save_path = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(save_path,
                                               'model_best.pth.tar'))














def create_image(py_list, i, j,  type, val):
    p_list = py_list[j]
    print("***********")
    print(numpy.asarray(p_list).shape)
    img = [[[0.0 for i1 in range(3)] for j1 in range(32)] for k1 in range(32)]
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    for i1 in range(32):
        for j1 in range(32):
            p1 = int((p_list[0][i1][j1] * std[0] + mean[0]) * 255)
            p2 = int((p_list[1][i1][j1] * std[1] + mean[1]) * 255)
            p3 = int((p_list[2][i1][j1] * std[2] + mean[2]) * 255)
            l = [p1, p2, p3]

            img[i1][j1] = l

    img_arr = numpy.asarray(img)
    new_im = Image.fromarray(img_arr.astype('uint8'), mode='RGB')

    new_im.save("output/img_" + str(i) + "_" + str(j) +  "_" + type + "_" + str(val)  + ".png")








def loss_op(output, target, dist, scale_const):

    loss1 = torch.clamp(target - output, min=0.)
    loss1 = torch.sum(scale_const * loss1)
    loss2 = dist.sum()
    loss = loss1 + loss2
    return loss


def tanh_rescale(x, x_min=-2.22, x_max=2.5):
    return (torch.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min


def reduce_sum(x, keepdim=True):
    # silly PyTorch, when will you get proper reducing sums/means?
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)
    return x


def l2_dist(x, y, keepdim=True):
    d = (x - y) ** 2
    return reduce_sum(d, keepdim=keepdim)


def optimize(optimizer, model, input_var, modifier_var, target_var, j):
    # apply modifier and clamp resulting image to keep bounded from clip_min to clip_max
    x = torch.tensor(100000)
    scale_const_var = autograd.Variable(x, requires_grad=False)
    input_adv = tanh_rescale(modifier_var + input_var, -2.22, 2.5)
    a, b, output1, output2 = model(input_adv)
    if (j == 0):
        output = output1
    else:
        output = output2
    # distance to the original input data

    dist = l2_dist(input_adv, input_var, keepdim=False)

    loss = loss_op(output, target_var, dist, scale_const_var)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print(loss)
    loss_np = loss.item()
    dist_np = dist.data.cpu().numpy()
    output_np = output.data.cpu().numpy()
    input_adv_np = input_adv.data.permute(0, 2, 3, 1).cpu().numpy()  # back to BHWC for numpy consumption
    return loss_np, dist_np, output_np, input_adv_np


if __name__ == '__main__':
    main()
