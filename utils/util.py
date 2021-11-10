import torch
import os
from IPython import embed


def save_checkpoint(path, epoch, model, optimizer, accuracy):
    '''
    保存三个键值对，epoch，整个model，optimizer（lr）
    '''
    state_dict = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer,
        'acc': accuracy}
    torch.save(state_dict, path)


def load_checkpoint(path, model, optimizer=None, model_only=False):
    '''
    将path路径下的文件加载到model和optimizer， epoch赋给start_epoch
    return start_epoch 开始的epoch
    '''
    if not os.path.exists(path):
        print('Sorry, don\'t have checkpoint.pth file, continue training!')
        return 0, 0
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    if model_only:
        return
    optimizer = checkpoint['optimizer']
    start_epoch = checkpoint['epoch']
    acc = checkpoint['acc']
    return start_epoch, acc
    # return start_epoch


def load_acc(path):
    checkpoint = torch.load(path)
    acc = checkpoint['acc']
    return acc


# print(load_acc('../checkpoints/checkpoint_resnet18.pth.tar'))
