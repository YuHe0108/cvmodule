import os
import pickle
import torch
import numpy as np

"""
通过计算逐层计算每张特征图的rank(秩)，按照压缩率，将秩比较少的特征图进行裁剪, 
因为根据先验知识，rank小的特征图所携带的信息更少。
参考论文：https://arxiv.org/abs/2002.10179
"""


class GeneratorMask:
    """逐层生成 filter 的mask， mask[idx] == 0 的位置，所以该特征图不重要"""

    def __init__(self, model=None, compress_rate=None, job_dir='', device=None):
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

    def layer_mask(self, cov_id, resume=None, param_per_cov=3, arch="resnet_56"):
        """
        conv_id:    减掉第几个卷积层的参数
        resume:     第一次更新mask的时候，为None, 后续会在前次的基础上增加
        """
        params = self.model.parameters()
        prefix = "rank_conv\\" + arch + "\\rank_conv"  # 根据rank_conv和compress_rate删减filter
        subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume = self.job_dir + '\\mask'

        self.param_per_cov = param_per_cov  # 有多少个卷积层
        for index, item in enumerate(params):  # 逐层遍历模型的参数
            if index == cov_id * param_per_cov:
                break
            if index == (cov_id - 1) * param_per_cov:
                f, c, w, h = item.size()
                rank = np.load(prefix + str(cov_id) + subfix)
                pruned_num = int(self.compress_rate[cov_id - 1] * f)
                ind = np.argsort(rank)[pruned_num:]  # preserved filter id

                zeros = torch.zeros(f, 1, 1, 1).to(self.device)
                for i in range(len(ind)):
                    zeros[ind[i], 0, 0, 0] = 1.
                self.mask[index] = zeros  # convolutional weight
                item.data = item.data * self.mask[index]
            elif (cov_id - 1) * param_per_cov < index < cov_id * param_per_cov:
                self.mask[index] = torch.squeeze(zeros)
                item.data = item.data * self.mask[index].to(self.device)

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)  # 保存 mask 至本地

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        remain_weights_file = self.job_dir + '\\remain_weights'  # 保存裁剪后的权重
        if not os.path.exists(remain_weights_file):
            remain_weights = {}
        else:
            with open(remain_weights_file, 'rb') as f:
                remain_weights = pickle.load(f)

        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov:
                break
            cur_mask = self.mask[index]
            # 与权重mask相乘,减掉的权重置0,维度保持不变
            item.data = item.data * cur_mask.to(self.device)
            # 根据mask，裁剪掉对应的权重, 维度发生了改变
            select_idx = torch.tensor([i for i in range(len(cur_mask)) if cur_mask[i] == 1], dtype=torch.int)
            res = torch.index_select(item.data, 0, select_idx)
            remain_weights[index] = res

        with open(remain_weights_file, "wb") as f:  # 将裁剪后的重要的权重，保存在本地
            pickle.dump(remain_weights, f)
