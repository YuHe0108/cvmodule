import torch as tc
import torchsummary
import torchvision as tv
from torch.nn.utils import prune
import torch.nn.functional as F
from torch import nn

"""模型的剪枝"""

device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# lenet = LeNet().to(device)
# module = lenet.conv1
# for v in module.named_parameters():
#     print(v[0], v[1].size())
#
# prune.random_unstructured(module, name='weight', amount=0.3)
# for v in module.named_parameters():
#     print(v[0], v[1].size())
# for v in module.named_buffers():
#     print(v[1])
#
# print(module.weight)
#
# prune.l1_unstructured(module, name='bias', amount=3)
# print(module.bias)
# print(list(module.named_buffers()))
# print(module._forward_hooks)
# 对模型进行剪枝操作, 分别在weight和bias上剪枝
# module = lenet.conv1
# prune.random_unstructured(module, name="weight", amount=0.3)
# prune.l1_unstructured(module, name="bias", amount=3)
#
# # 再将剪枝后的模型的状态字典打印出来
# print(lenet.state_dict().keys())
# print(list(module.named_parameters()))
#
# # 打印剪枝后的模型mask buffers参数
# print('*'*50)
# print(list(module.named_buffers()))
#
# prune.remove(module, 'weight')
# print('*'*50)
# print(list(module.named_parameters()))
# 第二种: 多参数模块的剪枝(Pruning multiple parameters).
model = LeNet().to(device=device)
for k, v in model.named_parameters():
    print(k, v.dtype)
# 打印初始模型的所有状态字典
# print(model.state_dict().keys())
# print('*' * 50)
#
# # 打印初始模型的mask buffers张量字典名称
# print(dict(model.named_buffers()).keys())
# print('*' * 50)
#
# for name, module in model.named_modules():
#     if isinstance(module, nn.Conv2d):
#         prune.l1_unstructured(module, name='weight', amount=0.3)
#     elif isinstance(module, nn.Linear):
#         prune.ln_structured(module, name='weight', amount=0.3, n=2, dim=0)
#
# # 打印多参数模块剪枝后模型的所有状态字典名称
# print(model.state_dict().keys())
# print('*' * 50)
# print(dict(model.named_buffers()).keys())  # 打印多参数模块剪枝后的mask buffers张量字典名称

# 构建参数集合, 决定哪些层, 哪些参数集合参与剪枝
# parameters_to_prune = (
#     (model.conv1, 'weight'),
#     (model.conv2, 'weight'),
#     (model.fc1, 'weight'),
#     (model.fc2, 'weight'),
#     (model.fc3, 'weight'))
#
# # 调用prune中的全局剪枝函数global_unstructured执行剪枝操作, 此处针对整体模型中的20%参数量进行剪枝
# prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.2)
# # 最后打印剪枝后的模型的状态字典
# print(model.state_dict().keys())
# print(
#     "Sparsity in conv1.weight: {:.2f}%".format(
#         100. * float(tc.sum(model.conv1.weight == 0))
#         / float(model.conv1.weight.nelement())
#     ))
#
# print(
#     "Sparsity in conv2.weight: {:.2f}%".format(
#         100. * float(tc.sum(model.conv2.weight == 0))
#         / float(model.conv2.weight.nelement())
#     ))
#
# print(
#     "Sparsity in fc1.weight: {:.2f}%".format(
#         100. * float(tc.sum(model.fc1.weight == 0))
#         / float(model.fc1.weight.nelement())
#     ))
#
# print(
#     "Sparsity in fc2.weight: {:.2f}%".format(
#         100. * float(tc.sum(model.fc2.weight == 0))
#         / float(model.fc2.weight.nelement())
#     ))
#
# print(
#     "Sparsity in fc3.weight: {:.2f}%".format(
#         100. * float(tc.sum(model.fc3.weight == 0))
#         / float(model.fc3.weight.nelement())
#     ))
#
# print(
#     "Global sparsity: {:.2f}%".format(
#         100. * float(tc.sum(model.conv1.weight == 0)
#                      + tc.sum(model.conv2.weight == 0)
#                      + tc.sum(model.fc1.weight == 0)
#                      + tc.sum(model.fc2.weight == 0)
#                      + tc.sum(model.fc3.weight == 0))
#         / float(model.conv1.weight.nelement()
#                 + model.conv2.weight.nelement()
#                 + model.fc1.weight.nelement()
#                 + model.fc2.weight.nelement()
#                 + model.fc3.weight.nelement())
#     ))
