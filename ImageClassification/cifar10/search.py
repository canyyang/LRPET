import sys
import argparse
import os
from re import M
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
import numpy as np
import copy
from collections import OrderedDict
from nni.compression.pytorch.utils.counter import count_flops_params
from models.resnet import *
from models.vgg_cifar import *
from models.googlenet import GoogLeNet
import torchvision

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


model_names = sorted(name for name in models_ima.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models_ima.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch MNIST pruning from deep compression paper')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train (default: 100)')#164
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--decay', type=float, default=0.001, metavar='D',
                    help='weight decay (default: 0.001)')
parser.add_argument('--schedule', type=int, nargs='+', default=[20,20],
                        help='Decrease learning rate at these epochs.') 
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')                      
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log', type=str, default='log.txt',
                    help='log file name')
parser.add_argument('--save_path', type=str, default='./saves',
                    help='model file')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--depth', type=int, default=56, help='Model depth.')
parser.add_argument('--print_frequence', default=1000, type=int)
parser.add_argument('--s', default='SVD/CNN/cifar10/save_log', type=str)
parser.add_argument('--model', default='resnet56', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--prune_ratio', type=float, default=0.7,
                    help="sigma be pruned")
parser.add_argument('--add_fac', type=float, default=1.025,
                    help="sigma be pruned")
parser.add_argument('--redu_fac', type=float, default=0.0,
                    help="sigma be pruned")
parser.add_argument('--num', default=0, type=int)
parser.add_argument('--cifar', default=10, type=int)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--search_start', type=float, default=0.5,
                    help="sigma be pruned")
parser.add_argument('--lambda_search', type=float, default=1)

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

# args.model = 'resnet%s' % args.depth
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
elif args.model == 'densenet40':
    model_full = DenseNet40_4(num_classes = args.cifar).to(device)
elif args.model == 'googlenet':
    model_full = GoogLeNet().to(device)

elif args.model == 'resnet50_ima':
    print("=> creating model '{}'".format(args.arch))
    model_full = models_ima.__dict__[args.arch]()
    num_ftrs = model_full.fc.in_features  
    model_full.fc = nn.Linear(num_ftrs, args.cifar) # num classes 12(according to your classes)
    model_full = model_full.to(device)

model_train = model_full
state = {
    'model': model_full.state_dict(),
}
torch.save(state, '%s/init_model_%s.t7' % (args.s, args.model))

dummy_input = torch.randn(1, 3, 32, 32)
flops, params, results = count_flops_params(model_full, dummy_input)
print("%s |%s |%s" % ('model_name',flops/1e6,params/1e6))
# model = model_svd
print(model_full)
criterion = nn.CrossEntropyLoss()
optimizer_full = optim.SGD(model_full.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=False)


def test_fit(model):
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
    # mean_loss = test_loss / (batch_idx + 1)
    acc = correct / total
    return -acc

def fcConvWeightReguViaSVB_redu(model, epoch, prune_ratio_list):
    layer_count = 0
    for name, m in model.named_modules():
        if isinstance(m,nn.Conv2d):
            prune_ratio = prune_ratio_list[layer_count]
            layer_count += 1
            if m.stride==(1, 1) or m.stride==(2, 2):
                
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

                # pr1
                prun_index = int(tmpS.size(0) - tmpS.size(0)*prune_ratio)

                if epoch % 20 == 0 and epoch != 0:
                    print(name, tmpS)

                
                alpha = ( torch.norm(tmpS, p=2).pow(2) - torch.norm((tmpS[prun_index:] * args.redu_fac), p=2).pow(2) ) / torch.norm(tmpS[:prun_index], p=2).pow(2)
                tmpS[:prun_index] = tmpS[:prun_index] * torch.sqrt(alpha)
                tmpS[prun_index:] = tmpS[prun_index:] * args.redu_fac
                
                tmpbatchM = torch.mm(torch.mm(tmpU, torch.diag(tmpS.cuda())), tmpV.t()).t().contiguous()

                m.weight.data.copy_(tmpbatchM.view_as(m.weight.data))

    return model  #///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    
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

def channel_decompose_guss(model_in, prune_ratio_list):
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
    # layer_count_ln = 0
    for name, m in model_in.named_modules():
        prun_flag = False
        if isinstance(m,nn.Conv2d):
            prune_ratio  = prune_ratio_list[layer_count]
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
            prun_index = int(sigma.size(0) - sigma.size(0)*prune_ratio)
            # prune
            if sigma[0] < 1e-5:
                prun_flag = True



            N = N[:, :prun_index].contiguous()
            sigma = sigma[:prun_index]
            C = C[:prun_index, :]

            if m.stride == (1, 1) or m.stride==(2, 2):  # when decoupling, only conv with 1x1 stride is considered
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
def test(epoch, model):
    
    global best_acc
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
        torch.save(state, '%s/best_%s_reduce_%s.t7' % (args.s, args.model, args.prune_ratio))
    print('epoch:%d    accuracy:%.3f    best:%.3f' % (epoch, acc, best_acc))
    



count_layer = 0
dimensions = []
default_parameters = []
for m in model_full.modules():
    if isinstance(m,nn.Conv2d):
        count_layer += 1
        dimensions.append(Real(name='pruning_ratio_auto'+'_%s'%(count_layer), low=args.search_start, high=0.999, prior='log-uniform'))
        default_parameters.append(random.uniform(args.search_start,0.999))

def model_fitness(x):
    seed_torch(args.seed)
    learning_rate1 = x
    model = copy.deepcopy(model_full)
    base_model = torch.load('SVD/CNN/cifar10/save_log/init_model_%s.t7' % (args.model))['model']
    model.load_state_dict(base_model)
    optimizer_full = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    
    for epoch in range(0, 1):
    # for epoch in range(0, args.epochs):
        model = train_full(epoch, model,optimizer_full)
        print('energy transfer')
        model = fcConvWeightReguViaSVB_redu(model, epoch, learning_rate1)

    learning_rate_reset(optimizer_full)
    fitness = test_fit(model)

    model_test = copy.deepcopy(model)
    print('==> exchange to SVD')
    model_test = channel_decompose_guss(model_in=model_test, prune_ratio_list=learning_rate1)
    test_fitness = test_fit(model_test)
    print('model prune:' , test_fitness)
    fitness += test_fitness
    dummy_input = torch.randn(1, 3, 32, 32)
    flops, params, results = count_flops_params(model_train, dummy_input)
    flops_svd, params_svd, results_svd = count_flops_params(model_test, dummy_input)
    fitness += args.lambda_search * flops_svd / flops
    print('pruning ratio param：', (1-(params_svd / params)))
    print('pruning ratio flops：', (1-(flops_svd / flops)))
    print(fitness)
    return fitness

search_result = gp_minimize(func=model_fitness,
                            dimensions=dimensions,
                            acq_func="EI",
                            n_calls=2,
                            x0=default_parameters,
                            n_random_starts=0,
                            random_state=123
                            )

print(search_result.x)
print(search_result.fun)

for fitness, x in sorted(zip(search_result.func_vals, search_result.x_iters)):
    print(fitness, x)

count_layer = 0
prune_inedx_list = [] # search 
for m in model_full.modules():
    if isinstance(m,nn.Conv2d):
        prun_goal = search_result.x[count_layer]
        count_layer += 1
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

        prun_index = int(tmpS.size(0) - tmpS.size(0)*prun_goal)
        prune_inedx_list.append(prun_index)
print(prune_inedx_list)