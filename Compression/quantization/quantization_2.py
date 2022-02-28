import os
import torch
from torch import nn
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import OrderedDict


# from base_train import validation
# from config import checkpoint, device


class VGG_prunable(nn.Module):
    def __init__(self, cfg):
        super(VGG_prunable, self).__init__()
        self.features = self._make_layers(cfg)
        self.classifier = nn.Linear(cfg[-2], 10)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def VGG_11_prune(cfg=None):
    if cfg is None:
        cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    return VGG_prunable(cfg)


# 量化权重
def signed_quantize(x, bits, bias=None):
    # x: 权重，bits: 量化位数，bias: 偏置
    min_val, max_val = x.min(), x.max()
    n = 2.0 ** (bits - 1)
    scale = max(abs(min_val), abs(max_val)) / n
    qx = torch.floor(x / scale)
    if bias is not None:
        qb = torch.floor(bias / scale)
        return qx, qb
    else:
        return qx


# 对模型整体进行量化
def scale_quant_model(model, bits):
    net = deepcopy(model)
    params_quant = OrderedDict()
    params_save = OrderedDict()

    for k, v in model.state_dict().items():
        if 'classifier' not in k and 'num_batches' not in k and 'running' not in k:
            if 'weight' in k:
                weight = v
                bias_name = k.replace('weight', 'bias')
                try:
                    bias = model.state_dict()[bias_name]
                    w, b = signed_quantize(weight, bits, bias)
                    params_quant[k] = w
                    params_quant[bias_name] = b
                    if 8 < bits <= 16:
                        params_save[k] = w.short()  # int16 位
                        params_save[bias_name] = b.short()
                    elif 1 < bits <= 8:
                        params_save[k] = w.char()
                        params_save[bias_name] = b.char()
                    elif bits == 1:
                        params_save[k] = w.bool()
                        params_save[bias_name] = b.bool()
                except:
                    w = signed_quantize(weight, bits)  # w = signed_quantize(w, bits)
                    params_quant[k] = w
                    params_save[k] = w.char()
        else:
            params_quant[k] = v
            params_save[k] = v
    net.load_state_dict(params_quant)
    return net, params_save


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pruned = False
    if pruned:
        channels = [17, 'M', 77, 'M', 165, 182, 'M', 338, 337, 'M', 360, 373, 'M']
        net = VGG_11_prune(channels).to(device)
        net.load_state_dict(
            torch.load(
                os.path.join(checkpoint, 'best_retrain_model.pth'))['compressed_net'])
    else:
        net = VGG_11_prune().to(device)
        net.load_state_dict(
            torch.load(
                os.path.join(checkpoint, 'best_model.pth'), map_location=torch.device('cpu')
            )['net']
        )

    validation(net, torch.nn.CrossEntropyLoss())
    accuracy_list = []
    bit_list = [16, 12, 8, 6, 4, 3, 2, 1]
    for bit in bit_list:
        print('{} bit'.format(bit))
        scale_quantized_model, params = scale_quant_model(net, bit)
        print('validation: ', end='\t')
        accuracy, _ = validation(scale_quantized_model, torch.nn.CrossEntropyLoss())
        accuracy_list.append(accuracy)
        torch.save(params,
                   os.path.join(checkpoint, 'pruned_{}_{}_bits.pth'.format(pruned, bit)))

    plt.plot(bit_list, accuracy_list)
    plt.savefig('img/quantize_pruned:{}.jpg'.format(pruned))
    plt.show()
