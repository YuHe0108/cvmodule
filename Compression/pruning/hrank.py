import os
import pickle
import torch
import numpy as np
import torchsummary
import copy

# 自定义文件
from site_config import parse_model_config
import gen_model

"""
通过计算逐层计算每张特征图的rank(秩)，按照压缩率，将秩比较少的特征图进行裁剪, 
因为根据先验知识，rank小的特征图所携带的信息更少。
参考论文：https://arxiv.org/abs/2002.10179
"""


class GeneratorMask:
    """逐层生成 filter 的mask， mask[idx] == 0 的位置，所以该特征图不重要"""

    def __init__(self, model=None, compress_rate=None, job_dir='', device='cpu'):
        """
        model:          需要裁剪的模型
        compress_rate:  模型每层的压缩率
        job_dir:        mask 保存的位置
        device：        cpu 或者 cuda
        注意：生成 mask 之前需要先计算每层卷积输出特征图的秩
        """
        if compress_rate is None:
            compress_rate = [0.50]
        self.compress_rate = compress_rate
        self.model = model
        self.mask = {}  # 每一层需要被删除的filter
        self.job_dir = job_dir  # 存放 mask 的 dir 路径
        self.device = device
        self.param_per_cov = None
        self.prune_one_size = False  # 是否裁剪卷积核为 1x1 的filter

    def layer_mask(self, cov_id, shortcut_idx, resume=None, param_per_cov=3, arch="resnet_56"):
        """
        conv_id:            减掉第几个卷积层的参数
        resume:             第一次更新mask的时候，为None, 后续会在前次的基础上增加
        param_per_cov:      每一层有多少参数
        shortcut_idx:      当前的卷积层和哪层卷积输出相加
        """
        params = self.model.parameters()
        prefix = os.path.join('rank_conv', arch, 'rank_conv')  # 根据rank_conv和compress_rate删减filter
        subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume = self.job_dir + '\\mask'

        self.param_per_cov = param_per_cov  # 有多少参数需要共用一个mask
        for index, item in enumerate(params):  # 逐层遍历模型的参数
            if index == cov_id * param_per_cov:  # 逐层计算mask, 当前属于下一层卷积，需要跳出
                break

            if index == (cov_id - 1) * param_per_cov:  # 当前层是卷积层
                f, c, w, h = item.size()
                if self.prune_one_size and w == h == 1:  # 卷积核的尺寸为 1x1
                    pass
                if shortcut_idx == -1:
                    rank = np.load(prefix + str(cov_id) + subfix)
                    pruned_num = int(self.compress_rate[cov_id - 1] * f)
                    ind = np.argsort(rank)[pruned_num:]  # preserved filter id
                    mask = torch.zeros(f, 1, 1, 1).to(self.device)
                    for i in range(len(ind)):
                        mask[ind[i], 0, 0, 0] = 1.
                else:
                    # 残差结构需要保证mask一致
                    mask = self.mask[shortcut_idx * param_per_cov]
                self.mask[index] = mask  # convolutional weight
                item.data = item.data * self.mask[index]
            elif (cov_id - 1) * param_per_cov < index < cov_id * param_per_cov:
                # 当前层的权重为 BN 的 bias 和 weight，选择和前一层Conv相同的裁剪mask
                self.mask[index] = torch.squeeze(mask)
                item.data = item.data * self.mask[index].to(self.device)

        with open(resume, "wb") as f:  # 保存 mask 至本地
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov:
                break
            cur_mask = self.mask[index]
            item.data = item.data * cur_mask.to(self.device)  # 与权重mask相乘,减掉的权重置0,维度保持不变


def generator_pruned_model(pruned_ckpt, config_file, mask_dir):
    """
    根据 mask 和 模型的权重 生成裁剪后的权重信息
    并生成裁剪后模型的相关信息
    """
    module_defs = parse_model_config(config_file)  # 模型每层的定义
    pretrain_weight = torch.load(pruned_ckpt, map_location='cpu')['state_dict']
    with open(mask_dir, 'rb') as f:  # 每个卷积核是否要选择的mask
        mask = pickle.load(f)

    # 修剪权重
    idx = 0
    conv_filters = []  # 裁剪后每层卷积层的filter数量
    new_weight = copy.deepcopy(pretrain_weight)
    per_layer_params = 6  # 多少个单元共用一个 mask
    select_idx = torch.tensor([i for i in range(int(module_defs[0]['channels']))], dtype=torch.int)
    for i, (k, v) in enumerate(pretrain_weight.items()):
        if (i + 1) % per_layer_params == 1:  # 下一层卷积层的由于前一层输入发生了改变，因此也要裁剪
            v = torch.index_select(v.data, 1, select_idx)
        if idx >= len(mask):
            new_weight[k] = v
            break
        if (i + 1) % per_layer_params == 0:
            conv_filters.append(int(torch.sum(mask[idx]).item()))
            idx += 3  # mask为每三个一个单元, conv、bn-weight、bn-bias
            continue
        select_idx = torch.tensor([i for i in range(len(mask[idx])) if mask[idx][i] == 1],
                                  dtype=torch.int)
        v = torch.index_select(v.data, 0, select_idx)
        new_weight[k] = v

    torch.save({'state_dict': new_weight}, 'pruned_ckpt.pt')  # 保存已经剪枝后模型的参数
    filter_idx = 0  # 根据索引判定每层卷积 层，剪枝后的filter数量，重新生成配置文件
    with open('pruned_arc.cfg', 'w') as f:
        for module_def in module_defs:
            f.write(f"[{module_def['type']}]\n")
            for key, value in module_def.items():
                if key != 'type':
                    if module_def['type'] == 'convolutional' and key == 'filters':
                        if filter_idx >= len(conv_filters):
                            f.write(f"{key}={value}\n")
                        else:
                            f.write(f"{key}={conv_filters[filter_idx]}\n")
                            filter_idx += 1
                    else:
                        f.write(f"{key}={value}\n")
            f.write("\n")
    return


if __name__ == '__main__':
    generator_pruned_model('cifar10_pruned.pt', 'model.cfg',
                           r'D:\LasoFiles\test_git\cvmodule\Compression\pruning\cifar10_mask\mask')
