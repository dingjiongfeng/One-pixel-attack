"""
train CNN using cifar10
"""
import os
import torch
from torch import optim
from torch.backends import cudnn
import torch.nn as nn
from torch.nn import parameter
from torch.optim import optimizer
from torch.serialization import save
import torchvision
from torchvision.transforms import transforms
from torchvision.models import *
from torch.utils.data import DataLoader
from IPython import embed
from torch.optim.lr_scheduler import StepLR
import argparse
from tqdm import tqdm
from utils.util import *
from models.model import init_model

parser = argparse.ArgumentParser(description="Train CNN using cifar10")
# Datasets
parser.add_argument(
    "--root", type=str, default='/home/chenkx/djf/data', help="root path to data directory"
)
parser.add_argument(
    "--model", type=str, default='vgg16', help="model to train"
)
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument(
    "--workers", default=4, type=int, help="number of data loading workers(default 4)"
)
parser.add_argument("--resume", action="store_true",
                    help="whether use checkpoint")
parser.add_argument("--pin-memory", action="store_true",
                    help="whether use pin memory")
parser.add_argument("--max-epoch", type=int, default=10,
                    help="max training epoch")
parser.add_argument("--start-epoch", type=int, default=0,
                    help="start training epoch")
parser.add_argument("--test_freq", type=int, default=2,
                    help="num of epoch per test")
parser.add_argument("--print_freq", type=int, default=30,
                    help="num of epoch per test")
parser.add_argument('--step-size', type=int, default=5,
                    help='lr scheduler steps')
parser.add_argument('--gpu-device', type=str,
                    default="0", help='gpu device ids')

args = parser.parse_args()

checkpoint_path = "./checkpoints"
use_cuda = torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
best_acc = -1
test_acc = 0
n_class = 10
args.resume = True


train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]
)
print('Preparing data!')
traindataset = torchvision.datasets.CIFAR10(
    root=args.root, download=False, train=True, transform=train_transform
)
testdataset = torchvision.datasets.CIFAR10(
    root=args.root, download=False, train=False, transform=test_transform
)

trainloader = DataLoader(
    dataset=traindataset,
    batch_size=32,
    shuffle=True,
    num_workers=args.workers,
    drop_last=True,
    pin_memory=args.pin_memory,
)
testloader = DataLoader(
    dataset=testdataset,
    batch_size=32,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=args.pin_memory,
)
print('Data prepared!')


def train(epoch, model, optimizer, criterion):
    model.train()
    train_loss = 0
    total = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(trainloader):  # image_nums / 32
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()  # optimizer.zero_step() ??
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predict = output.max(dim=1)  # 删除列维度 成为一行
        total += data.size(0)
        correct += (predict == target).cpu().sum()
        train_acc = correct / total

        if (batch_idx+1) % args.print_freq == 0:
            print(
                f'Epoch {epoch} [{batch_idx+1} / {len(trainloader)}] Loss: {train_loss:.3f} \t accuracy: {train_acc:.4f}')
            correct, total, train_loss = 0, 0, 0


def test(epoch, model, criterion, optimizer):
    global best_acc  # 若想在函数内部对函数外的变量进行操作，就需要在函数内部声明其为global。
    model.eval()
    test_loss = 0
    total = 0
    correct = 0

    with torch.no_grad():
        for i, (data, target) in tqdm(
            enumerate(testloader), desc="Testing!"
        ):  # image_nums / 32
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            test_loss += loss.item()
            _, predict = output.max(dim=1)
            total += data.size(0)
            correct += (predict == target).sum()
            test_acc = correct / total

            if test_acc > best_acc:
                best_acc = test_acc
                save_checkpoint(os.path.join(checkpoint_path, 'checkpoint_'+args.model+'.pth.tar'), epoch,
                                model.module if use_cuda else model, optimizer, test_acc)

            if (i+1) % args.print_freq == 0:
                print(
                    f'Epoch {epoch} [{i+1} / {len(trainloader)}] Loss: {test_loss:.3f} \t accuracy: {test_acc:.4f}')
                correct, total, test_loss = 0, 0, 0


def main():
    if use_cuda:
        # print('current using cuda device', args.gpu_device)
        # os.environ['CUDA_DEVICE_VISIBLE'] = args.gpu_device
        cudnn.benchmark = True  # benchmark基准 选择最合适的卷积算法，减少训练时间
        torch.cuda.manual_seed(666)
    print("initializing model")

    # 加载模型
    model = init_model(args.model)
    model = model.to(device)
    print(
        "model parameters size: {:.3f}M".format(
            sum(p.numel() for p in model.parameters()) / 1000000.0  # 参数量
        )
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001,
                          momentum=0.9, weight_decay=5e-4)
    lr_scheduler = StepLR(optimizer, args.step_size, gamma=0.5)
    start_epoch = args.start_epoch
    if args.resume:  # 如果需要加载模型
        start_epoch, acc = load_checkpoint(os.path.join(
            checkpoint_path, 'checkpoint_'+args.model+'.pth.tar'), model, optimizer)
        # print('best_acc:', acc.item())

    model = nn.DataParallel(model, device_ids=range(
        torch.cuda.device_count()))  # 数据并行 放在最后加数据并行，因为model已经是去掉DataParallel的了
    # model = nn.DataParallel(model)

    for epoch in range(start_epoch, args.max_epoch):  # (0, 60)
        train(epoch, model, optimizer, criterion)
        if epoch % args.test_freq == 0:
            test(epoch, model, criterion, optimizer)
        lr_scheduler.step()


if __name__ == "__main__":
    main()

    # model = resnet50(pretrained=True)
    # model = nn.DataParallel(model)
    # model = model.to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    # test(1, model, criterion, optimizer)

    # tmux new -s mysession 创建指定名称的会话
    # tmux a -t mysession 连接指定名称的会话
