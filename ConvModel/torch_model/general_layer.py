import torch as tc
import torch.nn as nn
import torch.nn.functional as F

"""
基于 pytorch 封装常用的层 
"""


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format='channels_last'):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(tc.ones(normalized_shape))  # 可训练参数
        self.bias = nn.Parameter(tc.zeros(normalized_shape))
        self.eps = eps

        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == 'channels_last':
            # pytorch 中实现 layer norm 是从数据的最后一维计算均值和方差
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / tc.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

