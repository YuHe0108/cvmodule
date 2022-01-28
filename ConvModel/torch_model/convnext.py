import sys
import torch as tc
import torch.nn as nn
import torchsummary as tcm
import torchvision as tcv
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

sys.path.append('ConvModel\\torch_model')

import general_layer


# Reference address: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    两种实现方式效果像痛经， 但是使用第二种方式，在pytorch会更快

    Args:
        dim (int): Number of input channels.
        drop_prob (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_prob=0., layer_scale_init_value=1e-6):
        super(Block, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # 深度可分离卷积
        self.norm = general_layer.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # point wise conv, 但是采用全连接层实现
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * tc.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_prob) if drop_prob > 0. else nn.Identity()

    def forward(self, x):
        skip_x = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = skip_x + self.drop_path(x)
        return x


class ConvNext(nn.Module):
    r""" ConvNeXt
            A PyTorch impl of : `A ConvNet for the 2020s`  -
              https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_channel (int): 图像的channel数量，RGB=3，GrayScale=1
        num_classes (int):  类别的总数
        depths (tuple(int)):    Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (tuple(int):       Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_channel=3, num_classes=1000, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768),
                 drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1.):
        super(ConvNext, self).__init__()

        # 模型的起始层, 对特征图进行了四倍的下采样 + LayerNorm
        stem = nn.Sequential(
            nn.Conv2d(in_channel, dims[0], kernel_size=4, stride=4),
            general_layer.LayerNorm(dims[0], eps=1e-6, data_format='channels_first')
        )

        # stem和中间层用于降低特征图的尺寸
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(stem)
        # 三个用于降采样的单元
        for i in range(3):
            downsample_layer = nn.Sequential(
                general_layer.LayerNorm(dims[i], eps=1e-6, data_format='channels_first'),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)

        # 四个stage，每个 stage 由 depths[i] 个block组合而成
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in tc.linspace(0, drop_path_rate, sum(depths))]  # drop_path 丢弃率
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_prob=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # 最后的输出层
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # 输出前的LayerNorm
        self.head = nn.Linear(dims[-1], num_classes)

        # 参数初始化的方式，参数权重以及bias的取值
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)  # 先经过下采样层
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)


def load_weight(model, device, ckpt_path):
    """
    加载 模型的权重信息，并去除头部用于分类的权重
    """
    weights_dict = tc.load(ckpt_path, map_location=device)["model"]
    for k in list(weights_dict.keys()):  # 删除head最后用于分类的权重
        if "head" in k:
            del weights_dict[k]
    # 模型层的名称如果与weight_dict中的键值对不能一一对应，strict=False，就会将没有对应的进行随机初始化
    model.load_state_dict(weights_dict, strict=False)
    return model


def convnext_tiny(device, ckpt_path=None, **kwargs):
    model = ConvNext(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if ckpt_path is not None:
        model = load_weight(model, device, ckpt_path)
    return model


def convnext_small(device, ckpt_path=None, **kwargs):
    model = ConvNext(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if ckpt_path is not None:
        model = load_weight(model, device, ckpt_path)
    return model


def convnext_base(device, ckpt_path=None, **kwargs):
    model = ConvNext(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if ckpt_path is not None:
        model = load_weight(model, device, ckpt_path)
    return model


def convnext_large(device, ckpt_path=None, **kwargs):
    model = ConvNext(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if ckpt_path is not None:
        model = load_weight(model, device, ckpt_path)
    return model


def convnext_xlarge(device, ckpt_path=None, **kwargs):
    model = ConvNext(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if ckpt_path is not None:
        model = load_weight(model, device, ckpt_path)
    return model


if __name__ == '__main__':
    # block = Block(16)
    # tcm.summary(block, (16, 32, 32))
    convnext_model = ConvNext(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], num_classes=21841)
    tcm.summary(convnext_model, (3, 320, 320))
    # weight = tc.load(r'D:\weights\convnext_base_22k_224.pth', map_location='cpu')["model"]
    # convnext_model.load_state_dict(weight, strict=False)
