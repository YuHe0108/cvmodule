import os
import torch
import pickle
import argparse
import numpy as np
from torch import nn
from collections import OrderedDict

# ------------------------------  generator feature rank per layer ----------------------------

parser = argparse.ArgumentParser(description='Rank extraction')
parser.add_argument('--data_dir', type=str, default=r'D:\LasoFiles\Github\DL\data', help='dataset path')
parser.add_argument('--dataset', type=str, default='cifar10', choices=('cifar10', 'imagenet'), help='dataset')
parser.add_argument('--job_dir', type=str, default='result\\tmp',
                    help='The directory where the summaries will be stored.')
parser.add_argument('--arch', type=str, default='resnet_56',
                    choices=('resnet_50', 'vgg_16_bn', 'resnet_56', 'resnet_110', 'densenet_40', 'googlenet'),
                    help='The architecture to prune')
parser.add_argument('--resume', type=str, default=None, help='load the model from the specified checkpoint')
parser.add_argument('--limit', type=int, default=5, help='The num of batch to get rank.')
parser.add_argument('--train_batch_size', type=int, default=128, help='Batch size for training.')
parser.add_argument('--eval_batch_size', type=int, default=100, help='Batch size for validation.')
parser.add_argument('--start_idx', type=int, default=0, help='The index of conv to start extract rank.')
parser.add_argument('--gpu', type=str, default='0', help='Select gpu to use')
parser.add_argument('--adjust_ckpt', action='store_true', default=None, help='adjust ckpt from pruned checkpoint')
parser.add_argument('--compress_rate', type=str, default=0, help='compress rate of each conv')

args = parser.parse_args()

print("实验配置：", args)


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


def get_feature_hook_googlenet(self, input, output):
    global feature_result
    global total
    a = output.shape[0]
    b = output.shape[1]
    c = torch.tensor([torch.matrix_rank(output[i, j, :, :]).item() for i in range(a) for j in range(b - 12, b)])

    c = c.view(a, -1).float()
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total


def test(train_loader):
    net.eval()
    test_loss = 0
    correct = 0
    total_cnt = 0
    limit = args.limit

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx >= limit:  # use the first 6 batches to estimate the rank.
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)  # 只进行正向计算

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total_cnt += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (test_loss / (batch_idx + 1), 100. * correct / total_cnt, batch_idx, len(train_loader)))
    return


if __name__ == '__main__':
    train_dataloader_ = None  # 数据集
    net = None  # 模型

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compress_rate = [0.1] + [0.60] * 35 + [0.0] * 2 + [0.6] * 6 + [0.4] * 3 + [0.1] + [0.4] + [0.1] + [0.4] + [0.1] + [
        0.4] + [0.1] + [0.4]  # 模型每层的压缩百分比

    if args.resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume, map_location='cuda:' + args.gpu)  # 加载模型
        new_state_dict = OrderedDict()
        if args.adjust_ckpt:
            for k, v in checkpoint.items():
                new_state_dict[k.replace('module.', '')] = v
        else:
            for k, v in checkpoint['state_dict'].items():
                new_state_dict[k.replace('module.', '')] = v
        net.load_state_dict(new_state_dict)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()  # 损失函数

    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    # 对于 resnet-56, 前面几层是 conv + bn + relu, 获取relu层，对conv+bn输出的特征图进行筛选
    cov_layer = eval('net.relu')  # 模型的 relu 激活函数
    handler = cov_layer.register_forward_hook(get_feature_hook)
    test(train_dataloader_)  # 每次移除一个filter的时候，都会重新测试一遍
    handler.remove()  # 移除 hook 函数

    # feature_result: 记录了每一个通道上所有 batch 在 rank 的平均值
    rank_save_dir = os.path.join('rank_conv', f'{args.arch}_limit{args.limit}')
    if not os.path.isdir(rank_save_dir):
        os.mkdir(rank_save_dir)
    np.save(os.path.join(rank_save_dir, 'rank_conv%d' % 1 + '.npy'), feature_result.numpy())

    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)  # 一共有多少个 batch * step 数据

    # ResNet56 per block
    cnt = 1  # 需要遍历每一层卷积
    for i in range(3):
        block = eval('net.layer%d' % (i + 1))
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            # 在每次执行test的时候，由于加入了 hook, 因此每个step都会计算一次特征图的rank，但是最后结果是平均值
            test(train_dataloader_)
            handler.remove()

            np.save(os.path.join(rank_save_dir, 'rank_conv%d' % (cnt + 1) + '.npy'), feature_result.numpy())

            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            test(train_dataloader_)
            handler.remove()
            np.save('rank_conv\\' + args.arch + '_limit%d' % args.limit + '\\rank_conv%d' % (cnt + 1) + '.npy',
                    feature_result.numpy())
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)
