import sys
import argparse
import os
import shutil
import time
import random
import math
import torch
import torch.nn as nn
from torch.nn.functional import batch_norm
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import copy
from collections import OrderedDict
from nni.compression.pytorch.utils.counter import count_flops_params
from models.resnet import *
from models.vgg_cifar import *
from models.googlenet import GoogLeNet
print(__doc__)
import numpy as np
from math import exp
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_histogram, plot_objective_2D, plot_objective
from skopt.utils import point_asdict
np.random.seed(123)
import matplotlib.pyplot as plt
from skopt import gp_minimize
import torchvision.models as models_ima





parser = argparse.ArgumentParser(description='PyTorch MNIST pruning from deep compression paper')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=400, metavar='N',
                    help='number of epochs to train (default: 100)')#164
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--weight_decay', type=float, default=0.0001, metavar='D',
                    help='weight decay (default: 0.0001)')
parser.add_argument('--schedule', type=int, nargs='+', default=[200,300],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')                      
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--save_path', type=str, default='./saves',
                    help='model file')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--depth', type=int, default=56, help='Model depth.')
parser.add_argument('--print_frequence', default=1000, type=int)
parser.add_argument('--s', default='LRPET/CNN/cifar10/save', type=str)
parser.add_argument('--model', default='resnet56', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--prune_ratio', type=float, default=0,
                    help="sigma be pruned")
parser.add_argument('--add_fac', type=float, default=1.3,
                    help="mplification factor for sigma")
parser.add_argument('--redu_fac', type=float, default=0.0,
                    help="reduce factor for sigma")
parser.add_argument('--num', default=0, type=int)
parser.add_argument('--cifar', default=10, type=int)
parser.add_argument('--prun_goal', type=float, default=0.80,
                    help="sigma be pruned")
parser.add_argument('--train', action='store_true', default=True)

parser.add_argument('--pretrained',
                    dest='pretrained',
                    help='whether use pretrained model',
                    default=False,
                    type=bool)
parser.add_argument('--checkpoint',
                    dest='checkpoint',
                    help='checkpoint dir',
                    default=None,
                    type=str)
parser.add_argument('--use_bn', action='store_false', default=True)

args = parser.parse_args()
print(args)
def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch(args.seed)
# seed_torch()
eps_set = 1e-5
print('PID:%d' % (os.getpid()))

# Data
print('==> Preparing dataset cifar-10')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataloader = datasets.CIFAR10
num_classes = 10

trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
train_loader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
test_loader = data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)

print(args.model)
# Select Device
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(args.seed)
else:
    print('Not using CUDA!!!')

if args.model == 'resnet20':
    model_full = resnet20(args.cifar).to(device)
elif args.model == 'resnet32':
    model_full = resnet32(args.cifar).to(device)
elif args.model == 'resnet56':
    model_full = resnet56(args.cifar).to(device)
elif args.model == 'resnet110':
    model_full = resnet110(args.cifar).to(device)
elif args.model == 'vgg16':
    model_full = vgg16(num_classes = args.cifar).to(device)
elif args.model == 'vgg11_bn':
    model_full = vgg11_bn(num_classes = args.cifar).to(device)
elif args.model == 'vgg16_bn':
    model_full = vgg16_bn(num_classes = args.cifar).to(device)
elif args.model == 'googlenet':
    model_full = GoogLeNet().to(device)
elif args.model == 'resnet50_ima':
    # print("=> creating model '{}'".format(args.arch))
    # model_full = models_ima.__dict__[args.arch]()
    model_full = models_ima.resnet50(pretrained=True)
    num_ftrs = model_full.fc.in_features
    model_full.fc = nn.Linear(num_ftrs, args.cifar)
    model_full = model_full.to(device)

state = {
    'model': model_full.state_dict(),
}
torch.save(state, '%s/init_model_%s.t7' % (args.s, args.model))

print(model_full)
criterion = nn.CrossEntropyLoss()
optimizer_full = optim.SGD(model_full.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)


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
 
def adjust_learning_rate(optimizer, epoch):
    if epoch in args.schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.gamma

def learning_rate_reset(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] == args.lr

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

def train_full(epoch, model, optimer):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    adjust_learning_rate(optimer, epoch)
    # print(optimer)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % args.print_frequence == args.print_frequence - 1 or args.print_frequence == train_loader.__len__() - 1:
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # lr_scheduler.step()
    test(epoch, model)
    return model
    


best_acc = 0
best_acc_reduce399 = 0
def test(epoch, model):
    
    global best_acc
    global best_acc_reduce399
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        state = {
            'model': model.state_dict(),
            'acc': acc
        }
        torch.save(state, '%s/best_%s_reduce_%s_num%s.t7' % (args.s, args.model, args.prune_ratio, args.num))

    if epoch >= 300 and epoch<=399:
        if acc > best_acc_reduce399:
            best_acc_reduce399 = acc
            state = {
                'model': model.state_dict(),
                'acc': acc
            }
            torch.save(state, '%s/399best_%s_reduce_%s_num%s.t7' % (args.s, args.model, args.prune_ratio, args.num))
            print('best_reduce399:%.3f' % (best_acc_reduce399))

    print('epoch:%d    accuracy:%.3f    best:%.3f' % (epoch, acc, best_acc))
    



# init for prun ratio
default_parameters = []
for m in model_full.modules():
    if isinstance(m,nn.Conv2d):
        tmpbatchM = m.weight.data.view(m.weight.data.size(0), -1).t().clone()
        try:
            tmpU, tmpS, tmpV = torch.svd(tmpbatchM)
        except:
            tmpbatchM = tmpbatchM[np.logical_not(np.isnan(tmpbatchM))]
            tmpbatchM = tmpbatchM.view(m.weight.data.size(0), -1).t()
            tmpU, tmpS, tmpV = np.linalg.svd(tmpbatchM.cpu().numpy())
            tmpU = torch.from_numpy(tmpU).to(device)
            tmpS = torch.from_numpy(tmpS).to(device)
            tmpV = torch.from_numpy(tmpV).to(device)

        prun_index = int(tmpS.size(0) - tmpS.size(0)*args.prun_goal)

        default_parameters.append(prun_index)
    


def change_test_guss(model, num, prune_index_list):
    model_test = copy.deepcopy(model)
    base_model = torch.load('%s/%sbest_%s_reduce_%s_num%s.t7' % (args.s, num , args.model, args.prune_ratio, args.num))['model']
    model_test.load_state_dict(base_model)
    print('==> exchange to SVD')
    model_train = channel_decompose_guss(model_in=model_test, prune_index_list=prune_index_list)
    test(0,model_train)  
    torch.save(model_train.state_dict(),args.save_path+"/pruning_Model.pth")

def ratio_print(model,prune_index_list):
    model_test = copy.deepcopy(model)
    
    dummy_input = torch.randn(1, 3, 32, 32)
    flops, params, results = count_flops_params(model_test, dummy_input)

    model_pruning = channel_decompose_guss(model_in=model_test, prune_index_list=prune_index_list)
    dummy_input = torch.randn(1, 3, 32, 32)
    flops_svd, params_svd, results = count_flops_params(model_pruning, dummy_input)
    print('pruning ratio param：', (1-(params_svd / params)))
    print('pruning ratio flops：', (1-(flops_svd / flops)))



if args.train:

    start_epoch = 0
    # ************************************ #
    # Note: Here we write the reduction of FLOPs before training. 
    #       In fact, some layers are invalid after training. 
    #       We will cut off these layers as a whole, so the actual reduction of FLOPs will be more.
    # ************************************ #

    ## 1. Use fixed pruning rate to perform singular value pruning. Only need to set '--prun_goal', prune_index_list = default_parameters
    # prune_ratio_list=default_parameters
    '''
    ResNet-56     --prun_goal 0.80  -->  reduction of FLOPs 79.1% (after training 79.1%)
                  --prun_goal 0.54  -->  reduction of FLOPs 51.0% (after training 51.0%)

                  --prun_goal 0.57 * 
                  --prun_goal 0.70 * 
                  --prun_goal 0.80 * 
    ResNet-110    --prun_goal 0.62  -->  reduction of FLOPs 58.2% (after training 67.6%)
                  --prun_goal 0.70  -->  reduction of FLOPs 69.3% (after training 77.5%)

                  --prun_goal 0.79  -->  0.785 *
                  --prun_goal 0.65  -->  0.629 * 
    GoogLeNet     --prun_goal 0.65  -->  reduction of FLOPs 57.5% (after training 57.5%)
                  --prun_goal 0.76  -->  reduction of FLOPs 70.9% (after training 70.9%)
                  --prun_goal 0.68  -->  reduction of FLOPs 0.61 * 
                  --prun_goal 0.60  -->  reduction of FLOPs 0.51 * 

    '''
    prune_index_list = default_parameters
    ## 3. Use searched parameters to perform singular value pruning. Use the following prune_index_list directly
    '''
    ## ResNet-56-S   --  reduction of FLOPs 78.0% (after training 80.8%)
    # prune_ratio_list = [6, 6, 3, 5, 2, 3, 4, 4, 2, 3, 6, 1, 2, 4, 4, 5, 2, 1, 2, 10, 6, 2, 6, 6, 7, 9, 1, 5, 4, 11, 6, 2, 2, 12, 12, 5, 4, 16, 6, 17, 23, 11, 3, 11, 17, 4, 12, 15, 16, 6, 7, 6, 24, 18, 15]
    '''
    model_train = model_full
    if args.model=='vgg16_bn':
        if prune_index_list[0] <18: # 27*2/3
            prune_index_list[0] = 18
        for i in range(1, len(prune_index_list)):
            if prune_index_list[i]<43:
                prune_index_list[i]=43

    print(prune_index_list)
    ratio_print(model_train,prune_index_list)
    dummy_input = torch.randn(1, 3, 32, 32)
    flops, params, results = count_flops_params(model_full, dummy_input)

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        model_train = train_full(epoch, model_train, optimizer_full)
        if args.use_bn:
            # if epoch%2==0 and epoch!=0:
            print('energy transfer use-bn')
            model_train = fcConvWeightReguViaSVB_redu(model_train, epoch, prune_index_list)
        data_time = time.time() - start_time
        print('time: %s', data_time)

        print(time.strftime("%Y-%m-%d--%H:%M:%S", time.localtime()))

        if epoch == 399:
            print('==> 399')
            change_test_guss(model_train, '399', prune_index_list)

    print('==> exchange to SVD')
    model_test = channel_decompose_guss(model_in=model_train, prune_index_list=prune_index_list)
    test(0,model_test)
    dummy_input = torch.randn(1, 3, 32, 32)
    flops_svd, params_svd, results = count_flops_params(model_test, dummy_input)
    print('pruning ratio param：', (1-(params_svd / params)))
    print('pruning ratio flops：', (1-(flops_svd / flops)))

