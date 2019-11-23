"""
Training file for training SkipNets for supervised pre-training stage
"""

from __future__ import print_function

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
from scipy.optimize import fsolve,minimize,fmin
from PIL import Image
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
    os.makedirs(save_path)
    #os.makedirs(save_path, exist_ok=True)

    # config logger file
    args.logger_file = os.path.join(save_path, 'log_{}.txt'.format(args.cmd))
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    if args.cmd == 'train':
        logging.info('start training {}'.format(args.arch))
        run_training(args)

    elif args.cmd == 'test':
        logging.info('start evaluating {} with checkpoints from {}'.format(
            args.arch, args.resume))
        test_model(args)


def run_training(args):
    # create model
    model = models.__dict__[args.arch](args.pretrained)
    model = torch.nn.DataParallel(model)
    #model = torch.nn.DataParallel(model).cuda()

    best_prec1 = 0

    # optionally resume from a checkpoint
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

    cudnn.benchmark = True

    train_loader = prepare_train_data(dataset=args.dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.workers)
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()#.cuda()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                       model.parameters()),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    skip_ratios = ListAverageMeter()

    end = time.time()
    for i in range(args.start_iter, args.iters):
        model.train()
        adjust_learning_rate(args, optimizer, i)

        input, target = next(iter(train_loader))
        # measuring data loading time
        data_time.update(time.time() - end)

        target = target#.cuda(async=False)
        input_var = Variable(input)#.cuda()
        target_var = Variable(target)#.cuda()

        # compute output
        output, masks, logprobs = model(input_var)

        # collect skip ratio of each layer
        skips = [mask.data.le(0.5).float().mean() for mask in masks]
        if skip_ratios.len != len(skips):
            skip_ratios.set_len(len(skips))

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        skip_ratios.update(skips, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # repackage hidden units for RNN Gate
        if args.gate_type == 'rnn':
            model.module.control.repackage_hidden()

        batch_time.update(time.time() - end)
        end = time.time()

        # print log
        if i % args.print_freq == 0 or i == (args.iters - 1):
            logging.info("Iter: [{0}/{1}]\t"
                         "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                         "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                         "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                         "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t".format(
                            i,
                            args.iters,
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses,
                            top1=top1)
            )
            for idx in range(skip_ratios.len):
                logging.info(
                    "{} layer skipping = {:.3f}({:.3f})".format(
                        idx,
                        skip_ratios.val[idx],
                        skip_ratios.avg[idx],
                    )
                )

        # evaluate every 1000 steps
        if (i % args.eval_every == 0 and i > 0) or (i == (args.iters-1)):
            prec1 = validate(args, test_loader, model, criterion)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            checkpoint_path = os.path.join(args.save_path,
                                           'checkpoint_{:05d}.pth.tar'.format(
                                               i))
            save_checkpoint({
                'iter': i,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            },
                is_best, filename=checkpoint_path)
            shutil.copyfile(checkpoint_path, os.path.join(args.save_path,
                                                          'checkpoint_latest'
                                                          '.pth.tar'))


def validate(args, test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    skip_ratios = ListAverageMeter()

    # switch to evaluation mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target#.cuda(async=True)
        logging.info('i:{i}'.format(i=i))
        input_var = Variable(input, volatile=True)#.cuda()
        py_list = input_var.tolist()
        input_var=torch.FloatTensor(py_list)
        target_var = Variable(target, volatile=True)#.cuda()
        # compute output
        #output, masks, _ = model(input_var)

        for j in range(2):
            #res = func(d, input_var, model, j)///
            probs, probs2 ,stacked_tensor,stacked_tensor2= model(input_var)
            sum = 0
            t = 0

            if (j == 0):
                pr = probs
            else:
                pr = probs2
            sum,t1,t2,t3=create_sum(pr)
            print("sum ",sum)
            if(sum<10.5):
                py_list = input_var.tolist()
		org_val = t1+(t2*1.2)+(t3*1.3)
                create_image(py_list, i, j, sum,t1,t2,t3,"orig",0)

                in_guess = [[0.0 for ind in range(16)] for ind2 in range(102)]
                dis2 = [-2.4 for ind in range(16)]
                in_guess[0] = dis2
                dis2 = [2.5 for ind in range(16)]
                in_guess[1] = dis2
                # print(in_guess)
                for ind2 in range(100):
                    for ind in range(16):
                        in_guess[ind2 + 2][ind] = random.uniform(-2.4, 2.5)
                    # print(dis2)
                    # print("---------")
                    # in_guess[ind2+2]=dis2
                    # print(in_guess)
                    # print("#######")
                max = 0
                # print(in_guess)
                for ind2 in range(102):
                    d = in_guess[ind2]
                    # print(dis2)
                    print(j)
                    res,t1,t2,t3 = func(d, input_var, model, j)
                    # s=get_score(res)
                    if (res > max):
                        max = res
                        dis2_f = d
                dis2 = dis2_f
                bnds = [(-2.4, 2.5)] * 16
                cons = ({'type': 'ineq', 'fun': lambda dis2: func(dis2, input_var, model, j)[0] - 11})
                # {'type': 'ineq', 'fun': lambda dis2: func(dis2, input_var, model)[8] - 0.5}
                # {'type': 'ineq', 'fun': lambda dis2: func(dis2, input_var, model)[12] - 0.5}
                '''cons=[]
                for ind in range(1,len(init)):
                    cons.append({'type': 'ineq', 'fun': lambda dis2: func(dis2,input_var,model)[ind]-0.5})
                print(cons)'''
                # f2 = np.vectorize(fun)
                # constraints=cons
                res1 = minimize(fun, dis2, args=input_var, method='SLSQP', bounds=bnds, constraints=cons,
                                options={'maxiter': 10})
                # res = fmin(func2, dis, args=(input_var, model))
                print("res1 ", res1.x)
		isNaN=False
		for tmp in res1.x:
			isNaN = math.isnan(tmp)
			if isNaN is True:
				break;
		if isNaN is False:
			res,cnt1,cnt2,cnt3 = func(res1.x, input_var, model, j)
		        py_list = input_var.tolist()

		        py_list = dist(py_list, j, res1.x)
		        logger.info(res)
		        logger.info(res1.x)
			inc_val = ((cnt1+(cnt2*1.2)+(cnt3*1.3))/org_val)*100
		        create_image(py_list, i, j, res,cnt1,cnt2,cnt3,"borderline",inc_val)
		              

                in_guess = [0.0 for ind2 in range(102)]

                # print(in_guess)
                for ind2 in range(102):
                    in_guess[ind2] = random.uniform(-0.15, 0.15)
                    # print(dis2)
                    # print("---------")
                    # in_guess[ind2+2]=dis2
                    # print(in_guess)
                    # print("#######")
                max = 0
                # print(in_guess)
                dis2=0.0
                dis2_f=0.0
                for ind2 in range(102):
                    d = in_guess[ind2]
                    # print(dis2)
                    res,t1,t2,t3 = func2(d, input_var, model,j)
                    # s=get_score(res)
                    if (res > max):
                        max = res
                        dis2_f = d
                dis2 = dis2_f

                bnds = [(-0.15, 0.15)]
                cons = ({'type': 'ineq', 'fun': lambda dis2: func2(dis2[0], input_var, model, j)[0] - 11})
                # {'type': 'ineq', 'fun': lambda dis2: func(dis2, input_var, model)[8] - 0.5}
                # {'type': 'ineq', 'fun': lambda dis2: func(dis2, input_var, model)[12] - 0.5}
                '''cons=[]
                for ind in range(1,len(init)):
                    cons.append({'type': 'ineq', 'fun': lambda dis2: func(dis2,input_var,model)[ind]-0.5})
                print(cons)'''
                # f2 = np.vectorize(fun)
                # constraints=cons
                res1 = minimize(fun2, dis2,  method='SLSQP', bounds=bnds, constraints=cons,
                                options={'maxiter': 10})
                # res = fmin(func2, dis, args=(input_var, model))
                print("res1 ", res1.x)

		if math.isnan(res1.x[0]) is False:
			res,cnt1,cnt2,cnt3 = func2(res1.x[0], input_var, model, j)
		        py_list = input_var.tolist()
		        print('before static..', py_list)


		        py_list = dist2(py_list, j, res1.x[0])
		        print('after..',	py_list)
		        logger.info(res)
		        logger.info(res1.x)
			inc_val = ((cnt1+(cnt2*1.2)+(cnt3*1.3))/org_val)*100

		        create_image(py_list, i, j, res,cnt1,cnt2,cnt3,"static", inc_val)


                
                modifier = torch.zeros(input_var.size()).float()
                # Experiment with a non-zero starting point...
                # modifier = torch.normal(means=modifier, std=0.001)
                modifier_var = autograd.Variable(modifier, requires_grad=True)
                optimizer = optim.Adam([modifier_var], lr=0.0005)
                target = torch.tensor([0.5] * 54)
                target_var = autograd.Variable(target, requires_grad=False)
                min_loss = sys.float_info.max
                adv_img_min = numpy.zeros((2, 32, 32, 3))
                min_output = torch.tensor([0.0] * 54)
		py_list = input_var.tolist()
		
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
                print(distance)
                print(min_loss)
                print(min_output)
                sum,t1,t2,t3 = create_sum(min_output)
                print(sum)
                adv_img = adv_img_min

                #p_list = adv_img[0]
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
                # new_im.save("output/img_hello2.png")
		inc_val = ((t1+(t2*1.2)+(t3*1.3))/org_val)*100
                new_im.save("output/img_" + str(i) + "_" + str(j) + "_" + str(sum) + "_" + str(t1) + "_" + str(t2) + "_" + str(t3) + "_" + "cw" +"_"+str(inc_val)+".png")

        '''skips = [mask.data.le(0.5).float().mean() for mask in masks]
        if skip_ratios.len != len(skips):
            skip_ratios.set_len(len(skips))
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        top1.update(prec1, input.size(0))
        skip_ratios.update(skips, input.size(0))
        losses.update(loss.data, input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or (i == (len(test_loader) - 1)):
            logging.info(
                'Test: [{}/{}]\t'
                'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'.format(
                    i, len(test_loader), batch_time=batch_time,
                    loss=losses,
                    top1=top1
                )
            )
    logging.info(' * Prec@1 {top1.avg:.3f}, Loss {loss.avg:.3f}'.format(
        top1=top1, loss=losses))

    skip_summaries = []
    for idx in range(skip_ratios.len):
        # logging.info(
        #     "{} layer skipping = {:.3f}".format(
        #         idx,
        #         skip_ratios.avg[idx],
        #     )
        # )
        skip_summaries.append(1-skip_ratios.avg[idx])
    # compute `computational percentage`
    cp = ((sum(skip_summaries) + 1) / (len(skip_summaries) + 1)) * 100
    logging.info('*** Computation Percentage: {:.3f} %'.format(cp))

    return top1.avg'''


def test_model(args):
    # create model
    #print('models dict-->', models)
    model = models.__dict__[args.arch](args.pretrained)
    print('model in use-->', model)
    model = torch.nn.DataParallel(model)
    #model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume,map_location='cpu')
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
    #logging.info('val_loader `{}`'.format(type(val_loader)))

    criterion = nn.CrossEntropyLoss()#.cuda()

    validate(args, test_loader, model, criterion)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        save_path = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(save_path,
                                               'model_best.pth.tar'))


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


class ListAverageMeter(object):
    """Computes and stores the average and current values of a list"""
    def __init__(self):
        self.len = 10000  # set up the maximum length
        self.reset()

    def reset(self):
        self.val = [0] * self.len
        self.avg = [0] * self.len
        self.sum = [0] * self.len
        self.count = 0

    def set_len(self, n):
        self.len = n
        self.reset()

    def update(self, vals, n=1):
        assert len(vals) == self.len, 'length of vals not equal to self.len'
        self.val = vals
        for i in range(self.len):
            self.sum[i] += self.val[i] * n
        self.count += n
        for i in range(self.len):
            self.avg[i] = self.sum[i] / self.count


def adjust_learning_rate(args, optimizer, _iter):
    """ divide lr by 10 at 32k and 48k """
    if args.warm_up and (_iter < 400):
        lr = 0.01
    elif 32000 <= _iter < 48000:
        lr = args.lr * (args.step_ratio ** 1)
    elif _iter >= 48000:
        lr = args.lr * (args.step_ratio ** 2)
    else:
        lr = args.lr

    if _iter % args.eval_every == 0:
        logging.info('Iter [{}] learning rate = {}'.format(_iter, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def fun2(dis2):
    return abs(dis2)



def fun(dis2,input_var):
    sum=0
    #sum=abs(dis2)
    py_list = input_var.tolist()

    #print("sum1 ", input1)
    '''v = [0.0 for i in range(10)]
    #sum=abs(v)
    for i in range(10):
        v[i]=input1[i]-input2[i]'''


    for i in range(3):
        for j in range(32):
            rem=j%16
            for k in range(1):
                sum+=abs(py_list[1][i][j][k]-dis2[rem])
            for k in range(31,32):
                sum += abs(py_list[1][i][j][k]-dis2[rem])
    for i in range(3):
        for j in range(32):
            if ((j >= 0 and j < 1) or (j >= 31 and j < 32)):
                for k in range(32):
                    rem=k%16
                    sum += abs(py_list[1][i][j][k] - dis2[rem])

    '''for i in range(0,10,2):
        sum += abs(-1.6 - py_list[1][0][dis2[i]][dis2[i+1]])
        sum+=abs(-1.6-py_list[1][1][dis2[i]][dis2[i+1]])
        sum+=abs(-1.6-py_list[1][2][dis2[i]][dis2[i+1]])'''
    #sum+=abs(-1.6-py_list[1][0][dis2[0]][dis2[1]])+abs(-1.6-py_list[1][1][dis2[0]][dis2[1]])+abs(-1.6-py_list[1][2][dis2[0]][dis2[1]])

    print(sum)
    return sum

def func2(dis2,input_var,model,imgno):

    print(dis2)
    py_list = input_var.tolist()
    py_list=dist2(py_list, imgno, dis2)
    # list=list1[0]
    # v=v.reshape(224)
    # logging.info('v:{v}'.format(v=(numpy.asarray(py_list).shape)))
    # logging.info('v:{v}'.format(v=type(v)))
    '''for i in range(3):
        for j in range(32):
            for k in range(32):
                py_list[1][i][j][k] = py_list[1][i][j][k] + dis2
                if (py_list[1][i][j][k] < -2.4):
                    py_list[1][i][j][k] = -2.4
                elif (py_list[1][i][j][k] < 2.5):
                    py_list[1][i][j][k] = 2.5'''
    # list1[0]=list
    # print(type(list1))
    # print(torch.as_tensor(py_list))
    print(input_var.dtype)
    input_var = torch.as_tensor(py_list)
    probs, probs2,stacked_tensor,stacked_tensor2 = model(input_var)
    s = 0
    t = 0


    if (imgno == 0):
        pr = probs
    else:
        pr = probs2
    s,t1,t2,t3 = create_sum(pr)

    # print(probs)
    #py_list = input_var.tolist()
    logger.info('s:{s}'.format(s=s))
    #print(t)
    '''s = 0
    for ind in probs:
        if (ind < 0.5):
            s += 0.5-ind'''

    return s,t1,t2,t3
def func(dis2,input_var,model,imgno):
    '''py_list=input_var.tolist()
    #print("sum1 ",sum(dis2))
    for i in range(3):
        for j in range(224):
            for k in range(10):
                # logging.info('list[i][j][k]:{v}'.format(v=(v[i])))
                temp = ((10 * i) + k)
                #print(py_list[0][i][j][k])
                py_list[0][i][j][k]=dis2[temp]
                #print("--------------")
                #print(py_list[0][i][j][k])
    input_var = torch.as_tensor(py_list)'''
    #print(dis2)
    py_list = input_var.tolist()
    '''if(sum>0):

        p_list = py_list[imgno]
        img = [[[0.0 for i in range(3)] for j in range(224)] for k in range(224)]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for i in range(224):
            for j in range(224):
                p1 = int((p_list[0][i][j] * std[0] + mean[0]) * 255)
                p2 = int((p_list[1][i][j] * std[1] + mean[1]) * 255)
                p3 = int((p_list[2][i][j] * std[2] + mean[2]) * 255)
                l = [p1, p2, p3]

                img[i][j] = l

        img_arr = numpy.asarray(img)
        new_im = Image.fromarray(img_arr.astype('uint8'), mode='RGB')
        new_im.save("img_" + str(batch) + "_" + str(imgno) + "_" + str(sum) + ".png")'''
    '''for i in range(3):
        for j in range(224):
            if(j<80 or j>=160):
                rem = j % 16
                for k in range(3):
                    py_list[imgno][i][j][k] = dis2[rem]
                for k in range(221, 224):
                    py_list[imgno][i][j][k] = dis2[rem]
    for i in range(3):
        for j in range(224):
            if (j < 80 or j >= 160):
                if ((j >= 0 and j < 3) or (j >= 221 and j < 224)):
                    for k in range(224):
                        rem = k % 16
                        py_list[imgno][i][j][k] = dis2[rem]'''
    py_list=dist(py_list, imgno, dis2)
    #print("type ",dis2)
    '''for i in range(3):
        for j in range(224):
            for k in range(224):
                py_list[1][i][j][k] = py_list[1][i][j][k]+dis2
                if(py_list[1][i][j][k]<-1.8):
                    py_list[1][i][j][k] = -1.8
                elif(py_list[1][i][j][k]<2.2):
                    py_list[1][i][j][k] = 2.2'''


    '''print(dis2)

    for i in range(0,20,2):
        py_list[1][0][int(dis2[i])][int(dis2[i+1])]=2.2
        py_list[1][1][int(dis2[i])][int(dis2[i+1])] = 2.2
        py_list[1][2][int(dis2[i])][int(dis2[i+1])] = 2.2'''
    #list=list1[0]
    #v=v.reshape(224)
    #logging.info('v:{v}'.format(v=(numpy.asarray(py_list).shape)))
    #logging.info('v:{v}'.format(v=type(v)))
    start = 194
    end = 224
    '''for i in range(3):
        for j in range(224):
            if((j>=0 and j<30) or (j>=194 and j<224)):
                for k in range(start, end):
                    # logging.info('list[i][j][k]:{v}'.format(v=(v[i])))
                    py_list[1][i][j][k] = float(dis2[k - start])
                for k in range(30):
                    py_list[1][i][j][k] = float(dis2[k])
                if (j == 0 or j == 223):
                    for k in range(100, 130):
                        py_list[1][i][j][k] = float(dis2[k - 100])
                    for k in range(150, 180):
                        py_list[1][i][j][k] = float(dis2[k - 150])
                    for k in range(180, 210):
                        py_list[1][i][j][k] = float(dis2[k - 180])
                    for k in range(40, 70):
                        py_list[1][i][j][k] = float(dis2[k - 50])'''
                #py_list[0][i][j][k] = py_list[0][i][j][k] + float(v)
    #logging.info('v:{v}'.format(v=(numpy.asarray(py_list).shape)))
    #list1[0]=list
    #print(type(list1))
    #print(torch.as_tensor(py_list))
    '''p_list = py_list[1]
    img = [[[0.0 for i in range(3)] for j in range(224)] for k in range(224)]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(224):
        for j in range(224):
            p1 = int((p_list[0][i][j] * std[0] + mean[0]) * 255)
            p2 = int((p_list[1][i][j] * std[1] + mean[1]) * 255)
            p3 = int((p_list[2][i][j] * std[2] + mean[2]) * 255)
            l = [p1, p2, p3]

            img[i][j] = l

    img_arr = numpy.asarray(img)
    new_im = Image.fromarray(img_arr.astype('uint8'), mode='RGB')

    new_im.save("img2.png")'''

    input_var = torch.as_tensor(py_list)
    print(input_var.dtype)
    probs, probs2,stacked_tensor,stacked_tensor2 = model(input_var)
    s = 0
    t = 0

    if (imgno == 0):
        pr = probs
    else:
        pr = probs2
    s,t1,t2,t3=create_sum(pr)

    #print(probs)
    py_list = input_var.tolist()
    '''if (sum > 0):

        p_list = py_list[imgno]
        img = [[[0.0 for i in range(3)] for j in range(224)] for k in range(224)]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for i in range(224):
            for j in range(224):
                p1 = int((p_list[0][i][j] * std[0] + mean[0]) * 255)
                p2 = int((p_list[1][i][j] * std[1] + mean[1]) * 255)
                p3 = int((p_list[2][i][j] * std[2] + mean[2]) * 255)
                l = [p1, p2, p3]

                img[i][j] = l

        img_arr = numpy.asarray(img)
        new_im = Image.fromarray(img_arr.astype('uint8'), mode='RGB')
        new_im.save("img_" + str(batch) + "_" + str(imgno) + "_" + str(s) + ".png")'''
    logger.info('s:{s}'.format(s=s))
    #print(t)
    '''s = 0
    for ind in probs:
        if (ind < 0.5):
            s += 0.5-ind'''

    return s,t1,t2,t3
'''def get_bounds():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    min1=(0-0.485)/0.229
    min2=(0-0.456)/0.224
    min3=(0-0.406)/0.225
    max1 = (1 - 0.485) / 0.229
    max2 = (1 - 0.456) / 0.224
    max3 = (1 - 0.406) / 0.225
    return min1,max1,min2,max2,min3,max3'''
def create_image(py_list,i,j,sum,t1,t2,t3,type,val):

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
    new_im.save("output/img_" + str(i) + "_" + str(j) + "_" + str(sum) +"_"+str(t1)+"_"+str(t2)+"_"+str(t3)+"_"+type+"_"+str(val)+ ".png")
def dist(py_list,j,dis2):
    for ind1 in range(3):
        for ind2 in range(32):
            rem = ind2 % 16
            for ind3 in range(1):
                py_list[j][ind1][ind2][ind3] = dis2[rem]
            for ind3 in range(31,32):
                py_list[j][ind1][ind2][ind3] = dis2[rem]

    for ind1 in range(3):
        for ind2 in range(32):
            if ((ind2 >= 0 and ind2 < 1) or (ind2 >= 31 and ind2 < 32)):
                for ind3 in range(32):
                    rem = ind3 % 16
                    py_list[j][ind1][ind2][ind3] = dis2[rem]

    return py_list

def dist2(py_list,num,dis2):
    for i in range(3):
        for j in range(32):
            for k in range(32):
                py_list[num][i][j][k] = py_list[1][i][j][k] + dis2
                if (py_list[num][i][j][k] < -2.4):
                    py_list[num][i][j][k] = -2.4
                elif (py_list[num][i][j][k] > 2.5):
                    py_list[num][i][j][k] = 2.5
    return py_list
def create_sum(pr):
    t1=0
    t2=0
    t3=0
    sum=0
    #print(pr)
    for ind in range(len(pr)):
        if (pr[ind][0] > 0.5):

            if ind <= 18:
                t1+=1
                sum += 0.1
            elif ind <= 36:
                t2+=1
                sum += 0.2
            elif ind <= 54:
                t3+=1
                sum += 0.5

    return sum,t1,t2,t3
def get_score(res):
    i=res[15]
    j=res[8]
    sum=0
    for r in range(len(res)):
        if(r==24 or r==25 or r==18):
            if(res[r]<0.5):
               sum+=0.5-res[r]
    print("sum ",sum)
    return sum

def loss_op(output, target, dist, scale_const):
    # compute the probability of the label class versus the maximum other
    '''real = (target * output).sum(1)
    other = ((1. - target) * output - target * 10000.).max(1)[0]
    if self.targeted:
        # if targeted, optimize for making the other class most likely
        loss1 = torch.clamp(other - real + self.confidence, min=0.)  # equiv to max(..., 0.)
    else:
        # if non-targeted, optimize for making this class least likely.
        loss1 = torch.clamp(real - other + self.confidence, min=0.)  # equiv to max(..., 0.)'''
    ##print(target)
    # print(output)
    # print("***************")
    loss1 = torch.clamp(target - output, min=0.)
    loss1 = torch.sum(scale_const * loss1)

    loss2 = dist.sum()
    # print(loss1)
    # print("----------------------")
    # print(loss2)
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
    d = (x - y)**2
    return reduce_sum(d, keepdim=keepdim)
def optimize(optimizer, model, input_var, modifier_var, target_var, j):
    # apply modifier and clamp resulting image to keep bounded from clip_min to clip_max
    x = torch.tensor(100000)
    scale_const_var = autograd.Variable(x, requires_grad=False)
    '''if self.clamp_fn == 'tanh':
        input_adv = tanh_rescale(modifier_var + input_var, -1.8, 2.2)
    else:
        input_adv = torch.clamp(modifier_var + input_var, -1.8, 2.2)'''
    input_adv = tanh_rescale(modifier_var + input_var, -2.22, 2.5)
    a, b,output1, output2 = model(input_adv)
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
