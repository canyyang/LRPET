import sys
import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import collections
from PIL import Image
from PIL import ImageFile
from tqdm import tqdm
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from nni.compression.pytorch.utils.counter import count_flops_params
import copy
from collections import OrderedDict
from torchstat import stat
from ptflops import get_model_complexity_info

from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_histogram, plot_objective_2D, plot_objective
from skopt.utils import point_asdict
np.random.seed(123)
import matplotlib.pyplot as plt
from skopt import gp_minimize
import torchvision.models as models_ima

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,7"

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/148Dataset/Dataset/ImageNet', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=int,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--s', default='./LRPET/CNN/imagenet/save', type = str, 
                    help='Session')
parser.add_argument('--schedule', type=int, nargs='+', default=[30,60,90],
                        help='Decrease learning rate at these epochs.') #[60, 120,180]  , [75, 150, 180] [10, 20, 40] [20, 40, 60] 
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')       
parser.add_argument('--model_name', default='resnet34', type=str)
parser.add_argument('--depth', type=int, default=50, help='Model depth.')
parser.add_argument('--dectype', type=str, help='the type of decouple', choices=['channel','space'], default='channel')
parser.add_argument('--prune_ratio', type=float, default=0.7,
                    help="sigma be pruned")
parser.add_argument('--redu_fac', type=float, default=0.0,
                    help="sigma be pruned")
parser.add_argument('--num', default=1, type=int)
parser.add_argument('--prun_goal', type=float, default=0.62,
                    help="sigma be pruned")
parser.add_argument('--search_start', type=float, default=0.5,
                    help="sigma be pruned")

args = parser.parse_args()
args.model_name = 'resnet%s' % args.depth
print(args.model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

def change_test(val_loader, model, criterion, args, epoch ,num, prune_index_list):
    # device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    model_test = copy.deepcopy(model)
    base_model = torch.load('%s/pth_epo%d_prun%s_num%d_%s.pth' % (args.s, epoch, args.prune_ratio, num, args.model_name))['state_dict']
    model_test.load_state_dict(base_model)
    print('==> exchange to SVD')
    model_train = channel_decompose_guss(model_in=model_test, prune_index_list=prune_index_list)
    acc1, acc5 = validate(val_loader, model_train, criterion, args)
    dummy_input = torch.randn(1, 3, 224, 224)
    flops, params, results = count_flops_params(model_train, dummy_input)
    print("%s |%s |%s" % ('model_name',flops/1e6,params/1e6))

def _set_model_attr(field_name, att, obj):
    '''
    set a certain filed_name like 'xx.xx.xx' as the object att
    :param field_name: str, 'xx.xx.xx' to indicate the attribution tree
    :param att: input object to replace a certain field attribution
    :param obj: objective attribution
    '''

    field_list = field_name.split('.')
    a = att

    # to achieve the second last level of attr tree
    for field in field_list[:-1]:
        a = getattr(a, field)

    setattr(a, field_list[-1], obj)


def fcConvWeightReguViaSVB_redu(model, epoch, prune_index_list):
    layer_count = 0
    conv_layer_num = 0
    
    for name, m in model.named_modules():
        if isinstance(m,nn.Conv2d):
            bn_layer_num = 0
            conv_layer_num += 1
            for name_bn, n in model.named_modules():
                if isinstance(n,nn.BatchNorm2d):
                    bn_layer_num += 1
                    if bn_layer_num == conv_layer_num:
                        prun_index = prune_index_list[layer_count]
                        layer_count += 1
                        tmpbatch_bn = n.weight.data
                        running_var = n.running_var
                        eps = n.eps
                        running_std = torch.sqrt(torch.add(running_var, eps))
                        alpha_bn = tmpbatch_bn / running_std
                        alpha_bn = alpha_bn.reshape((alpha_bn.size(0), 1))
                        eps1 = eps_set
                        # alpha_bn_back = 1/alpha_bn
                        alpha_bn_back = (1 / (alpha_bn**2+eps1)) *alpha_bn
                        tmpbatchM = m.weight.data.view(m.weight.data.size(0), -1).clone()
                        tmpbatchM_merge = torch.mul(tmpbatchM, alpha_bn)
                        try:
                            tmpU, tmpS, tmpV = torch.svd(tmpbatchM_merge)
                        except:
                            tmpbatchM_merge = tmpbatchM_merge.cpu().numpy()
                            # tmpbatchM_merge = tmpbatchM_merge[np.logical_not(np.isnan(tmpbatchM_merge))]
                            tmpbatchM_merge = tmpbatchM_merge.reshape(m.weight.data.size(0), -1)
                            tmpbatchM_merge = tmpbatchM_merge.transpose((1,0))
                            tmpU, tmpS, tmpV = np.linalg.svd(tmpbatchM_merge, full_matrices=0)
                            tmpU = torch.from_numpy(tmpU).to(device)
                            tmpS = torch.from_numpy(tmpS).to(device)
                            tmpV = torch.from_numpy(tmpV).to(device)
                        if tmpS[0] < 1e-4:
                            continue


                        alpha = ( torch.norm(tmpS, p=2).pow(2) - torch.norm((tmpS[prun_index:] * args.redu_fac), p=2).pow(2) ) / torch.norm(tmpS[:prun_index], p=2).pow(2)
                        tmpS[:prun_index] = tmpS[:prun_index] * torch.sqrt(alpha)
                        tmpS[prun_index:] = tmpS[prun_index:] * args.redu_fac

                        tmpbatchMx = torch.mm(torch.mm(tmpU, torch.diag(tmpS.to(device))), tmpV.t()).contiguous()
                        tmpbatchMx = torch.mul(tmpbatchMx, alpha_bn_back)  
                        

                        m.weight.data.copy_(tmpbatchMx.view_as(m.weight.data))

    return model  
 


def channel_decompose_guss(model_in, prune_index_list):
    '''
    decouple a input pre-trained model under nuclear regularization
    with singular value decomposition
    a single NxCxHxW low-rank filter is decoupled
    into a NxRx1x1 kernel following a RxCxHxW kernel
    :param model_in: object of derivated class of nn.Module, the model is initialized with pre-trained weight
    :param look_up_table: list, containing module names to be decouple
    :param criterion: object, a filter to filter out small valued simgas, only valid when train is False
    :param train: bool, whether decompose during training, if true, function only compute corresponding
           gradient w.r.t each singular value and do not apply actual decouple
    :param lambda_: float, weight for regularization term, only valid when train is True
    :return: model_out: a new nn.Module object initialized with a decoupled model
    '''
    layer_count = 0
    for name, m in model_in.named_modules():
        prun_flag = False
        if isinstance(m,nn.Conv2d):
            prun_index = prune_index_list[layer_count]
            layer_count += 1
            # if name in look_up_table:
            param = m.weight.data
            dim = param.size()
            
            if m.bias is not None:             
                hasb = True
                b = m.bias.data
            else:
                hasb = False
            
            NC = param.view(dim[0], -1) # [N x CHW]

            # try:
            N, sigma, C = torch.svd(NC, some=True)
            C = C.t()
            # prune
            if sigma[0] < 1e-5:
                prun_flag = True



            N = N[:, :prun_index].contiguous()
            sigma = sigma[:prun_index]
            C = C[:prun_index, :]

            
            r = int(sigma.size(0))
            C = torch.mm(torch.diag(torch.sqrt(sigma)), C)
            N = torch.mm(N,torch.diag(torch.sqrt(sigma)))

            C = C.view(r,dim[1],dim[2], dim[3])
            N = N.view(dim[0], r, 1, 1)

            if prun_flag == True:
                new_m = nn.Sequential()
            else:
                new_m = nn.Sequential(
                    OrderedDict([
                        ('C', nn.Conv2d(dim[1], r, dim[2], m.stride, m.padding, bias=False)),
                        ('N', nn.Conv2d(r, dim[0], 1, 1, 0, bias=hasb))
                    ])
                )
    
            
                state_dict = new_m.state_dict()
                print(name+'.C.weight'+' <-- '+name+'.weight')
                state_dict['C.weight'].copy_(C)
                print(name + '.N.weight' + ' <-- ' + name + '.weight')

                state_dict['N.weight'].copy_(N)
                if hasb:
                    print(name+'.N.bias'+' <-- '+name+'.bias')
                    state_dict['N.bias'].copy_(b)

                new_m.load_state_dict(state_dict)
            _set_model_attr(name, att=model_in, obj=new_m)

    return model_in.to(device)

def ratio_print(model,prune_index_list):
    model_test = copy.deepcopy(model)
    
    dummy_input = torch.randn(1, 3, 224, 224)
    flops, params, results = count_flops_params(model_test, dummy_input)

    model_pruning = channel_decompose_guss(model_in=model_test, prune_index_list=prune_index_list)
    dummy_input = torch.randn(1, 3, 224, 224)
    flops_svd, params_svd, results = count_flops_params(model_pruning, dummy_input)
    print('pruning ratio param：', (1-(params_svd / params)))
    print('pruning ratio flops：', (1-(flops_svd / flops)))
    
best_acc1 = 0
best_acc_reduce = 0
num_classes = 1000
def main_worker(gpu, args):
    global best_acc1
    global best_acc_reduce
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    print("Use GPU: {} for training".format(args.gpu))
    torch.cuda.set_device(args.gpu)
    print(model)

    default_parameters = []
    for m in model.modules():
        if isinstance(m,nn.Conv2d):

            tmpbatchM = m.weight.data.view(m.weight.data.size(0), -1).t().clone()
            try:
                tmpU, tmpS, tmpV = torch.svd(tmpbatchM)
            except:
                tmpbatchM = tmpbatchM[np.logical_not(np.isnan(tmpbatchM))]
                tmpbatchM = tmpbatchM.view(m.weight.data.size(0), -1).t()
                tmpU, tmpS, tmpV = np.linalg.svd(tmpbatchM.cpu().numpy())
                tmpU = torch.from_numpy(tmpU).cuda()
                tmpS = torch.from_numpy(tmpS).cuda()
                tmpV = torch.from_numpy(tmpV).cuda()

            prun_index = int(tmpS.size(0) - tmpS.size(0)*args.prun_goal)

            default_parameters.append(prun_index)


    ## 1. Use fixed pruning rate to perform singular value pruning. Only need to set '--prun_goal', prune_index_list = default_parameters
    # prune_ratio_list=default_parameters
    '''
    ResNet-34     --prun_goal 0.58  -->  reduction of FLOPs 53.2%
    ResNet-50    --prun_goal 0.62  -->  reduction of FLOPs 53.5% 
                  --prun_goal 0.80  -->  reduction of FLOPs 75.8%
    '''
    prune_index_list = default_parameters
    # print(default_parameters)

    ## 2. Use manual design parameters to perform singular value pruning. Use the following prune_index_list directly
    '''
    ## ResNet-34  --  reduction of FLOPs 46.0% 
    # prune_ratio_list = [32, 32, 32, 32, 32, 32, 32, 65, 65, 32, 65, 65, 65, 65, 65, 65, 120, 120, 65, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 220, 220, 120, 220, 220, 220, 220]

    '''

    ## 3. Use searched parameters to perform singular value pruning. Use the following prune_index_list directly
    '''
    ## ResNet-50-S   --  reduction of FLOPs 66.1% 
    # prune_ratio_list = [23, 16, 24, 29, 30, 6, 18, 27, 17, 4, 26, 33, 38, 61, 78, 63, 55, 23, 48, 51, 45, 35, 36, 55, 118, 50, 42, 80, 14, 45, 44, 30, 33, 89, 100, 15, 36, 44, 58, 45, 99, 49, 115, 145, 213, 95, 155, 238, 129, 170, 52, 94, 191]
    '''

    ratio_print(model,prune_index_list)
    model = torch.nn.DataParallel(model).to(device)
    
    

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile('%s/pth%d_%d.tar'%(args.s,args.i%args.num,args.resume)):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']+1
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    base_model = torch.load('LRPET/CNN/imagenet/save/pth_epo25_prun0.7_num21_resnet50_copy.pth')['state_dict']
    
    model.load_state_dict(base_model)
    acc1,acc5= validate(val_loader, model, criterion, args)
    args.start_epoch = 26

    
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        adjust_learning_rate(optimizer, epoch, args)
        train(train_loader, model, criterion, optimizer, epoch, args, prune_index_list)
        acc1,acc5= validate(val_loader, model, criterion, args)
        is_best = acc1 > best_acc1
        is_best_reduce = acc1 > best_acc_reduce
        if is_best:
            best_acc1 = max(acc1, best_acc1)
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.prune_ratio,  filename='%s/pth_epo%d_prun%s_num%d_%s.pth'%(args.s, epoch, args.prune_ratio, args.num, args.model_name))
        
        if epoch == 119:
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.prune_ratio,  filename='%s/pth_epo%d_prun%s_num%d_%s.pth'%(args.s, epoch, args.prune_ratio, args.num, args.model_name))

            change_test(val_loader, model, criterion,args,  epoch , args.num, prune_index_list)
            model = channel_decompose_guss(model_in=model, prune_index_list=prune_index_list)
            acc1, acc5 = validate(val_loader, model, criterion, args)
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.prune_ratio,  filename='%s/prune_pth_epo%d_prun%s_num%d.pth'%(args.s, epoch, args.prune_ratio, args.num))

        data_time = time.time() - start_time
        print('time: %s', data_time)

        print(time.strftime("%Y-%m-%d--%H:%M:%S", time.localtime()))
        

def train(train_loader, model, criterion, optimizer, epoch, args, prune_index_list):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        # compute output
        output = model(images)
        loss = criterion(output, target)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)
        
        if i % 250 == 0 and i != 0:
            print('energy transfer')
            fcConvWeightReguViaSVB_redu(model, epoch,prune_index_list)

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            output = model(images)
            loss = criterion(output, target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)
            # break
        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
                 
    return top1.avg, top5.avg
    
def save_checkpoint(state, is_best, prune, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '%s/model_best_prun%s_num%d_%s.pth'%(args.s, prune, args.num, args.model_name))
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args):
    if epoch in args.schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.gamma

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    
    main_worker(args.gpu, args)



