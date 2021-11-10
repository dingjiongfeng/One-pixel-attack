
import torch.nn as nn
from torchvision.models import vgg16, resnet50, resnet18
import torch
n_class = 10


def init_model(name):
    if name not in __model_names__.keys():
        raise KeyError(
            f'model {name} not supported, only {[*__model_names__.keys()]} supported')

    print(f'train {name} model!')
    return __model_names__[name]


def VGG16():
    model = vgg16(pretrained=False)
    in_feature = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_feature, n_class)
    nn.init.xavier_uniform_(model.classifier[6].weight)
    nn.init.constant_(model.classifier[6].bias, 0)
    return model


def Resnet50():
    model = resnet50(pretrained=False)
    in_feature = model.fc.in_features
    model.fc = nn.Linear(in_feature, n_class)
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.constant_(model.fc.bias, 0)
    return model


def Resnet18():
    model = resnet18(pretrained=False)
    in_feature = model.fc.in_features
    model.fc = nn.Linear(in_feature, n_class)
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.constant_(model.fc.bias, 0)
    return model


__model_names__ = {
    'vgg16': VGG16(),
    'resnet50': Resnet50(),
    'resnet18': Resnet18()
}
