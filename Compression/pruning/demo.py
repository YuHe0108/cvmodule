import os
import time
import torch
import numpy as np
import torch.nn as nn
import torchsummary

import gen_model
from layer_utils import to_cpu
import torch.nn.functional as F
from site_config import parse_model_config
from torch.autograd import Variable
import hrank

from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class Args:
    def __init__(self):
        self.data_dir = '..\\data\\cifar10'
        self.train_batch_size = 64
        self.eval_batch_size = 256


class Data:
    def __init__(self, args):
        # pin_memory = False
        # if args.gpu is not None:
        pin_memory = True

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

        train_set = CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
        self.loader_train = DataLoader(
            train_set, batch_size=args.train_batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory
        )
        test_set = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
        self.loader_test = DataLoader(
            test_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)


def test():
    global best_accu

    model.eval()
    test_data = data.loader_test
    test_accu = 0
    mean_accu = 0
    t = time.time()
    with torch.no_grad():
        for step, (img, target) in enumerate(test_data):
            img, target = Variable(img), Variable(target)
            output = model(img)
            pred = torch.max(output, 1)[1]
            test_accu += (pred == target).sum() / len(target)
            mean_accu = test_accu / (step + 1)

    print('accu: {:.2f}'.format(mean_accu), time.time() - t)
    if mean_accu > best_accu:
        best_accu = mean_accu
        torch.save({'state_dict': model.state_dict()}, 'cifar10.pt')
    return


def train(pruning=False):
    for epoch in range(5):
        model.train()
        train_data = data.loader_train

        train_accu = []
        for step, (img, target) in enumerate(train_data):
            img, target = Variable(img), Variable(target)
            output = model(img)
            pred = torch.max(output, 1)[1]
            train_accu.append((pred == target).sum() / len(target))

            if pruning:  # 施加裁剪, 每一个step，都要施加裁剪，保证了每次计算都是裁剪后的模型
                mask_model.grad_mask(cov_id)

            loss = loss_fun(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('epoch: {}, step: {:.3f}, accu: {:.2f}'.format(epoch, step / len(train_data), np.mean(train_accu)))
        test()
    return


def get_feature_hook(self, input, output):
    """钩子函数：用于统计模型特征图的相关信息 """
    global feature_result
    global total

    a = output.shape[0]  # batch_size
    b = output.shape[1]  # channel
    # 每个 batch、channel特征图的 rank
    c = torch.tensor([torch.matrix_rank(output[i, j, :, :]).item() for i in range(a) for j in range(b)])

    c = c.view(a, -1).float()  # [batch_size, -1]
    c = c.sum(0)  # 按照 batch 维度进行求和, c.size = channel的数量

    # 对于同一层同一个卷积层的filter
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total  # 每个channel的 rank 是多个 step 的平均值


if __name__ == '__main__':
    args_ = Args()
    data = Data(args_)

    cfg = parse_model_config('model.cfg')
    model = gen_model.GenModel(cfg)
    loss_fun = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    best_accu = 0
    # model.load_model_weights('mnist.pt')
    # test()
    train()
    """
    model.load_model_weights('mnist.pt')
    test()

    # 根据卷积层生成秩
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)
    rank_save_dir = os.path.join('rank_conv', 'mnist')
    if not os.path.exists(rank_save_dir):
        os.makedirs(rank_save_dir)

    print(model.module_list)
    cnt = 0
    for i, (module_def, module) in enumerate(zip(model.module_defs, model.module_list)):
        if module_def['type'] == 'convolutional':
            for j, layer in enumerate(module):
                if isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU):
                    handler = module[j - 1].register_forward_hook(get_feature_hook)
                    test()
                    handler.remove()
                    np.save(os.path.join(rank_save_dir, 'rank_conv%d' % (cnt + 1) + '.npy'), feature_result.numpy())
                    cnt += 1
                    feature_result = torch.tensor(0.)
                    total = torch.tensor(0.)
    """
    # test()
    # weight = torch.load('mnist.pt')
    # for layer in model:
    #     print(layer)
    # print(weight['state_dict'])
    # train()
    """生成mask
    compress_rate = [0.8, 0.95, 0.8]  # 减掉 30 %
    mask_model = hrank.GeneratorMask(model, compress_rate, job_dir='mnist_mask')
    for cov_id in range(2):
        print("cov-id: %d ====> Resuming from pruned_checkpoint..." % cov_id)
        mask_model.layer_mask(cov_id + 1, resume='', param_per_cov=3, arch='mnist')  # 计算需要裁剪哪些
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        best_acc = 0.
        train()  # 施加裁剪
        test()
    torch.save({'state_dict': model.state_dict()}, 'mnist_pruned.pt')
    """
