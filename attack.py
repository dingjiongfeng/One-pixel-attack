"""
attack CNN by change some pixel or one pixel in image
"""
from matplotlib import image
import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import resnet50
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from IPython import embed
from utils.util import *
import matplotlib.pyplot as plt
from torch.backends import cudnn
import torch.nn.functional as F
import argparse
import numpy as np
from scipy.optimize import differential_evolution
from tqdm import tqdm
from models.model import init_model
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(
    description='attack net by changing some pixels')
parser.add_argument('--pixels', type=int, default=1,
                    help='the number of pixels can be changed')

parser.add_argument('--maxiter', type=int, default=100,
                    help='the maximum number of iteration in the DE algorithm')
parser.add_argument('--popsize', type=int, default=400,
                    help='the number of adversarial examples in each iteration')
parser.add_argument('--samples', type=int, default=150,
                    help='The number of image samples to attack')
parser.add_argument('--model', type=str, default='resnet18',
                    help='model to attack')

parser.add_argument('--targeted', action='store_true',
                    help='switch to test for targeted attacks')
parser.add_argument('--verbose', action='store_true',
                    help='Print out additional information every iteration')

args = parser.parse_args()

print('Model Prepareing!')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = init_model(args.model)
model = model.to(device)
checkpoint_path = "checkpoints/checkpoint_"+args.model+".pth.tar"
load_checkpoint(checkpoint_path, model, model_only=True)  # 获取到最终训练到的模型
print('Model Prepared!')
# 读取cifar10图片

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),  # transform不同，准确率直接降到很低，resize,normalize
    ]
)
dataset = torchvision.datasets.CIFAR10(
    root="../data", train=False, transform=transform)  # 要转化成tensor
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def show_images(images):
    images = images.numpy()
    images = images.transpose((1, 2, 0))
    print(images.shape)
    plt.imshow(images)
    plt.show()


def test(model, noise=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            if noise:  # 添加一个噪音
                # _, height, width = images.shape[1:]
                # noises = torch.zeros(1)
                # images[:, :, height // 2, width // 2] = noises  # [4,32,32]
                noises = torch.randn(images.shape)*0.005
                images = images + noises

            images, labels = images.to(device), labels.to(device)
            output = model(images)

            _, predict = output.max(dim=1)
            correct += predict.eq(labels).cpu().sum()
            total += images.size(0)
            print(correct / total)

    return correct / total

# origin_acc = test(model, False)
# print(f'original acc is: {origin_acc}') #original acc is: 0.939
# trans_acc = test(model, True)
# print(f'transformed acc is: {trans_acc}') #transformed acc is: 0.939


def example():
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(dataloader))
        images = images.to(device)
        output = model(images)  # (4, n_classes)
        _, predict = output.max(dim=0)  # (4)

        print("预测类别", "\t", "真实类别")
        for (p, l) in zip(predict.data, labels):
            print(classes[p], "\t", classes[l])

        print('Show original image!')
        _images = torchvision.utils.make_grid(images, nrow=4)
        show_images(_images.cpu())

        # 全部添加微弱噪音
        batch_size, height, width = images.shape[1:]
        noise = torch.randn((batch_size, height, width)).to(device) * 0.1
        t1_images = images + noise  # [4,32,32]

        _images = torchvision.utils.make_grid(t1_images, nrow=4)
        show_images(_images.cpu())

        output = model(t1_images)  # (4, n_classes)
        _, predict = output.max(dim=0)  # (4)

        print("预测类别", "\t", "真实类别")
        for (p, l) in zip(predict.data, labels):
            print(classes[p], "\t", classes[l])

        # 添加一个噪音
        batch_size, height, width = images.shape[1:]
        noise = torch.zeros(1)
        t2_images = images
        t2_images[:, :, height // 2, width // 2] = noise  # [4,32,32]

        _images = torchvision.utils.make_grid(t2_images, nrow=4)
        show_images(_images.cpu())

        output = model(t2_images)  # (4, n_classes)
        _, predict = output.max(dim=0)  # (4)

        print("预测类别", "\t", "真实类别")
        for (p, l) in zip(predict.data, labels):
            print(classes[p], "\t", classes[l])


def attack_all(model, dataloader, pixels, targeted, maxiter, popsize, verbose):
    correct = 0
    success = 0
    model.eval()
    # 1 images size=[1,3,224,224]
    for batch_idx, (image, target) in tqdm(enumerate(dataloader)):
        print(batch_idx)
        image = image.to(device)
        prior_probs = F.softmax(model(image))  # @1

        _, indices = torch.max(prior_probs, dim=1)  # size[1]

        # embed()
        # for i in range(target.size(0)):
        #     if target[i] != indices[i].cpu():  # 表示与标签不相等
        #         continue
        if target[0] != indices.data.cpu()[0]:  # 取出模型能正确分类出的图片
            continue

        correct += 1
        target = target.numpy()

        targets = [None] if not targeted else range(10)
        # 无目标攻击 对十种类别进行targeted attack

        for target_class in targets:
            if targeted and target_class == target[0]:
                continue
            flag, x = attack(image, target, model, target_class, pixels=pixels,
                             maxiter=maxiter, popsize=popsize, verbose=verbose)  # flag 是否成功 x 结果
            success += flag
            if targeted:
                success_rate = float(success) / (9*correct)
            else:
                success_rate = float(success) / correct
            if flag == 1:
                print(
                    f"success rate: {success_rate:.4f} ({success}/{correct})", end='')
                for i in range(pixels):
                    print(
                        f"[(x,y)= ({x[5*i]},{x[5*i+1]}) (R,G,B)=({x[5*i+2]},{x[5*i+3]},{x[5*i+4]})]", end='')
                print()

        if correct == args.samples:
            break
    return success_rate


def attack(image, label, model, target, pixels=1, maxiter=75, popsize=400, verbose=True):
    '''
    image [1*3*w*h] tensor
    label: real target or None
    model: cnn model
    target: the targeted class
    pixels: the number of perturbed pixels
    maxiter: the maximum number of iteration in the DE algorithm
    popsize: the number of adversarial examples in each iteration potential answer!
    '''
    targeted_attack = target is not None
    # target_class = target if targeted_attack else label  # target None(不考虑)
    target_class = target if targeted_attack else label
    bounds = [(0, 32), (0, 32), (0, 255), (0, 255), (0, 255)] * pixels

    popmul = max(1, popsize / len(bounds))

    inits = np.zeros([int(popmul*len(bounds)), len(bounds)])  # 400*5 一定要是整数
    for init in inits:
        for i in range(pixels):  # [x,y,R,G,B]
            init[i*5+0] = np.random.random()*32
            init[i*5+1] = np.random.random()*32
            init[i*5+2] = np.random.normal(128, 127)
            init[i*5+3] = np.random.normal(128, 127)
            init[i*5+4] = np.random.normal(128, 127)

    def predict_fn(xs): return predict_classes(
        xs, image, target_class, model, target is None)

    def callback_fn(x, convergence): return attack_success(
        x, image, target_class, model, targeted_attack, verbose)

    attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul,
                                           recombination=1, atol=-1, callback=callback_fn, polish=False, init=inits)

    print('attack_result: ', attack_result)

    attack_image = perturb_image(attack_result.x, image)

    attack_var = attack_image.to(device)
    predicted_probs = F.softmax(model(attack_var)).data.cpu().numpy()[0]  # @1

    predicted_class = np.argmax(predicted_probs)

    print('predicted_class:', predicted_class, 'predicted_probs:', predicted_probs[predicted_class],
          ' label:', label.item())

    if (not targeted_attack and predicted_class != label) or (targeted_attack and predicted_class == target_class):
        return 1, attack_result.x.astype(int)
    return 0, [None]


def attack_success(x, image, target_class, model, targeted_attack, verbose):
    '''
    x
    target_class: if targeted attack targeted class
                  if not targeted    real class
    '''
    attack_image = perturb_image(x, image.clone())
    input = attack_image.to(device)
    confidence = F.softmax(model(input)).data.cpu().numpy()[0]  # @n
    predicted_class = np.argmax(confidence)

    if verbose:
        print(
            f'Confidence: {confidence[target_class].item():.4f}')
    if (targeted_attack and predicted_class == target_class) or (not targeted_attack and predict_classes != target_class):
        return True


def perturb_image(xs, img):
    '''
    xs: x坐标 行 size[1, 5] 随机值
    image: one batch tensor
    '''
    if xs.ndim < 2:
        xs = np.array([xs])

    batch = len(xs)
    imgs = img.repeat(batch, 1, 1, 1)
    xs = xs.astype(int)
    count = 0

    for x in xs:
        pixels = np.split(x, len(x)/5)
        for pixel in pixels:
            x_pos, y_pos, r, g, b = pixel
            # 像训练一样归一化
            imgs[count, 0, x_pos, y_pos] = (r/255.0 - 0.485) / 0.229
            imgs[count, 1, x_pos, y_pos] = (g/255.0 - 0.456) / 0.224
            imgs[count, 2, x_pos, y_pos] = (b/255.0 - 0.406) / 0.225
        count += 1
    return imgs


def predict_classes(xs, image, target_class, model, minimize=True):
    '''
    xs: array that perturb imgs
    image: the [1*3*w*h] tensor
    target_class: the targeted class
    minimize: whether is to minimize prediction, 
     untargeted: minimize
     targeted maximize
    '''
    imgs_perturbed = perturb_image(xs, image.clone())
    input = imgs_perturbed.to(device)
    predictions = F.softmax(model(input)).data.cpu().numpy()[
        :, target_class]  # @n

    return predictions if minimize else 1 - predictions


def main():
    if torch.cuda.is_available():
        cudnn.benchmark = True

    print('starting attack')

    result = attack_all(model, dataloader, pixels=args.pixels, targeted=args.targeted,
                        maxiter=args.maxiter, popsize=args.popsize, verbose=args.verbose)

    print(f'Final success rate: {result:.4f}')
    #

    '''
    
    Resnet50 test accuracy: 0.84  vgg16 0.76   resnet18 0.72
    1 pixel success rate:  0.2759     0.26        0.1800
    3 pixels success rate 0.45     0.3533        0.24
    '''


if __name__ == '__main__':
    main()
    # xs = np.array([1])
    # image = np.random.rand(1, 3, 32, 32)
    # image = perturb_image(xs, image)
    # print(image)

    # example()
    # noise_acc = test(model, True)
    # print('noise acc', noise_acc)
