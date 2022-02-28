from torchvision import datasets, transforms
from sklearn.cluster import KMeans
import torch.nn.functional as F
from bitarray import bitarray
from copy import deepcopy
import torch.utils.data
import torch as tc
import numpy as np
import torch.nn as nn
import pickle
import math


def as_bit_tensor(tensor, num_bits=2):
    shape = tensor.shape
    array = tensor.reshape(-1).numpy()
    bool_array = [np.zeros_like(array, dtype=np.bool) for _ in range(num_bits)]
    for i, bit in enumerate(reversed(range(num_bits))):
        idx = array >= (2 ** bit)
        bool_array[i][idx] = True
        array = array - (2 ** bit) * idx
    bit_arrays = [bitarray(bool_ten.tolist()) for bool_ten in bool_array]
    return {'s': shape, 'd': bit_arrays}


def as_int_tensor(bit_tensor):
    tensor = tc.zeros(bit_tensor['s'], dtype=tc.int)
    for i, bit in enumerate(reversed(bit_tensor['d'])):
        print(tc.tensor(bit.tolist(), dtype=tc.bool).reshape(bit_tensor['s']), (2 ** i))
        tensor += (2 ** i) * tc.tensor(bit.tolist(), dtype=tc.bool).reshape(bit_tensor['s'])
    return tensor


# CNN
def quantize_tensor(x, num_bits=8):
    q_min = 0.  # 量化的最小值
    q_max = (1 << num_bits) - 1  # 最大值
    min_val, max_val = x.min(), x.max()
    scale = (max_val - min_val) / (q_max - q_min)
    ini_zero_point = q_min - min_val / scale
    if ini_zero_point < q_min:
        zero_point = q_min
    elif ini_zero_point > q_max:
        zero_point = q_max
    else:
        zero_point = ini_zero_point
    zero_point = int(zero_point)
    q_x = zero_point + x / scale
    q_x.clamp(q_min, q_max).round_()
    q_x = q_x.round().byte()
    return q_x, scale, zero_point


def dequantize_tensor(q_x, scale, zero_point):
    return scale * (q_x.float() - zero_point)


class QuantLinear(nn.Linear):
    def __init__(self):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.quant_flag = False
        self.scale = None
        self.zero_point = None

    def liner_quant(self, quantize_bit=8):
        self.weight.data, self.scale, self.zero_point = quantize_tensor(self.weight.data, num_bits=quantize_bit)
        self.quant_flag = True

    def forward(self, inputs):
        if self.quant_flag:
            weight = dequantize_tensor(self.weight, self.scale, self.zero_point)
            return F.linear(inputs, weight, self.bias)
        else:
            return F.linear(inputs, self.weight, self.bias)

#


if __name__ == '__main__':
    a = tc.rand(3, 4)
    print(a)
    print(quantize_tensor(a))
    print(dequantize_tensor(*quantize_tensor(a)))
