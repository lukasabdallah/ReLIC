'''Train an encoder using Contrastive Learning.'''
import argparse
import os
import subprocess

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
# from torchlars import LARS
from tqdm import tqdm

from configs import get_datasets
from critic import LinearCritic
from evaluate import save_checkpoint, encode_train_set, train_clf, test
from models import *
from scheduler import CosineAnnealingWithLinearRampLR
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import torchvision

parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning.')
parser.add_argument('--base-lr', default=0.25, type=float, help='base learning rate, rescaled by batch_size/256')
parser.add_argument("--momentum", default=0.9, type=float, help='SGD momentum')
parser.add_argument('--resume', '-r', type=str, default='', help='resume from checkpoint with this filename')
parser.add_argument('--dataset', '-d', type=str, default='cifar100', help='dataset',
                    choices=['cifar10', 'cifar100', 'stl10', 'imagenet'])
parser.add_argument('--temperature', type=float, default=0.5, help='InfoNCE temperature')
parser.add_argument("--batch-size", type=int, default=128, help='Training batch size')
parser.add_argument("--num-epochs", type=int, default=500, help='Number of training epochs')
parser.add_argument("--cosine-anneal", type=bool, default=True, help="Use cosine annealing on the learning rate")
parser.add_argument("--arch", type=str, default='resnet50', help='Encoder architecture',
                    choices=['resnet18', 'resnet34', 'resnet50'])
parser.add_argument("--num-workers", type=int, default=1, help='Number of threads for data loaders')
parser.add_argument("--log_frequency", type=int, default=10, help='Logging frequency')
parser.add_argument("--test-freq", type=int, default=50, help='Frequency to fit a linear clf with L-BFGS for testing'
                                                              'Not appropriate for large datasets. Set 0 to avoid '
                                                              'classifier only training here.')
parser.add_argument("--filename", type=str, default='ckpt.pth', help='Output file name')
parser.add_argument("--mode", type=str, default='relic', choices=['relic', 'simclr'], help='ReLIC loss or SimCLR loss')
parser.add_argument("--alpha", type=float, default='40', help='Weighting of KL loss')
args = parser.parse_args()
args.lr = args.base_lr * (args.batch_size / 256)

args.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
args.git_diff = subprocess.check_output(['git', 'diff'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
clf = None
writer = SummaryWriter(log_dir=f"runs/{args.mode}")
print(args.mode)
print('==> Preparing data..')
trainset, testset, clftrainset, num_classes, stem = get_datasets(args.dataset)


def view_data(trainloader):
    figure = plt.figure(figsize=(8, 8))
    imgs, labels_idxs, _ = next(iter(trainloader))
    imgs = [torch.squeeze(i) for i in imgs]
    size = len(imgs[0])
    cols, rows = 2, size
    for add, view in enumerate(imgs):
        for i, (img, label_idx) in enumerate(zip(view, labels_idxs)):
            plot_pos = 2 * i + 1 + add
            figure.add_subplot(rows, cols, plot_pos)
            plt.title(trainset.classes[label_idx])
            plt.axis("off")
            for idx, channel in enumerate(img):
                channel = torch.add(channel, torch.abs(torch.min(channel)))
                img[idx] = torch.div(channel, torch.max(channel))

            plt.imshow(torch.movedim(img, 0, 2), norm=mpl.colors.Normalize())
    plt.show()


def get_sampler(mode, dataset):
    if mode == 'relic':
        sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(dataset), batch_size=args.batch_size, drop_last=True)
    else:
        sampler = None

    return sampler


shuffle = True
batch_size = args.batch_size
if args.mode == 'relic':
    shuffle = None
    batch_size = 1

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=shuffle,
                                          num_workers=args.num_workers, pin_memory=True,
                                          sampler=get_sampler(args.mode, trainset))
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=args.num_workers,
                                         pin_memory=True, sampler=get_sampler(args.mode, testset))
clftrainloader = torch.utils.data.DataLoader(clftrainset, batch_size=1000, shuffle=False, num_workers=args.num_workers,
                                             pin_memory=True)

# view_data(trainloader)

# Model
print('==> Building model..')
##############################################################
# Encoder
##############################################################
if args.arch == 'resnet18':
    net = ResNet18(stem=stem)
elif args.arch == 'resnet34':
    net = ResNet34(stem=stem)
elif args.arch == 'resnet50':
    net = ResNet50(stem=stem)
else:
    raise ValueError("Bad architecture specification")
net = net.to(device)

##############################################################
# Critic
##############################################################
critic = LinearCritic(net.representation_dim, temperature=args.temperature).to(device)

if device == 'cuda':
    repr_dim = net.representation_dim
    net = torch.nn.DataParallel(net)
    net.representation_dim = repr_dim
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    resume_from = os.path.join('./checkpoint', args.resume)
    checkpoint = torch.load(resume_from)
    net.load_state_dict(checkpoint['net'])
    critic.load_state_dict(checkpoint['critic'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
KLdiv = nn.KLDivLoss(reduction='batchmean', log_target=False)
base_optimizer = optim.SGD(list(net.parameters()) + list(critic.parameters()), lr=args.lr, weight_decay=1e-6,
                           momentum=args.momentum)

if args.cosine_anneal:
    scheduler = CosineAnnealingWithLinearRampLR(base_optimizer, args.num_epochs)
# encoder_optimizer = LARS(base_optimizer, trust_coef=1e-3)
encoder_optimizer = base_optimizer


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    critic.train()
    train_loss = 0
    n_iter = epoch * len(trainloader)
    # t = tqdm(enumerate(trainloader), desc='Loss: **** ', total=len(trainloader), bar_format='{desc}{bar}{r_bar}')

    t = enumerate(trainloader)
    for batch_idx, (inputs, _, _) in t:
        x1, x2 = inputs
        x1, x2 = x1.to(device), x2.to(device)
        encoder_optimizer.zero_grad()
        representation1, representation2 = net(x1), net(x2)
        raw_scores, pseudotargets, p_do1, p_do2 = critic(representation1, representation2)
        CEloss = criterion(raw_scores, pseudotargets)
        KL_div = KLdiv(p_do1, p_do2)
        if args.mode == 'relic':
            loss = CEloss + args.alpha * KL_div
        else:
            loss = CEloss
        loss.backward()
        encoder_optimizer.step()

        train_loss += loss.item()

        # t.set_description('Loss: %.3f ' % (train_loss / (batch_idx + 1)))

        if n_iter % args.log_frequency == 0:
            writer.add_scalar('CrossEntropyLoss', CEloss, global_step=n_iter)
            writer.add_scalar('KLDivLoss', KL_div, global_step=n_iter)
            writer.add_scalar('TotalLoss', loss, global_step=n_iter)
        n_iter += 1


for epoch in range(start_epoch, start_epoch + args.num_epochs):
    train(epoch)
    last_episode = (epoch == args.num_epochs - 1)
    if ((args.test_freq > 0) and (epoch % args.test_freq == (args.test_freq - 1))) or last_episode:
        X, y = encode_train_set(clftrainloader, device, net)
        clf = train_clf(X, y, net.representation_dim, num_classes, device, writer, epoch, reg_weight=1e-5)
        acc = test(testloader, device, net, clf, writer, epoch)
        writer.add_scalar('Classifier Total Test Accuracy', acc, global_step=epoch)
        if acc > best_acc:
            best_acc = acc
        save_checkpoint(net, clf, critic, epoch, args, os.path.basename(__file__))
    elif args.test_freq == 0:
        save_checkpoint(net, clf, critic, epoch, args, os.path.basename(__file__))
    if args.cosine_anneal:
        scheduler.step()
