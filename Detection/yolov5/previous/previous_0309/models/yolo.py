from torch import nn
import argparse
import torch
import yaml

# 自定义文件
from models.experimental import *


class Detect(nn.Module):
    def __init__(self, nc=80, anchors=()):  # detection layer
        super(Detect, self).__init__()
        self.stride = None  # strides computed during build
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # anchors的长度等于输出特征图的数量
        self.na = len(anchors[0]) // 2  # 每个特征图中的像素点有多少 anchor
        self.grid = [torch.zeros(1)] * self.nl  # 如果输出为3个特征图: grid = [0, 0, 0]
        # a.size = (3, 3, 2), 小尺寸特征图对应的anchor: a[0, :, :], 中尺寸的对应的anchor: a[1, :, :]
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl, na, 2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl, 1, na,1,1,2)
        self.export = False  # onnx export: 是否转成 onnx 的格式

    def forward(self, x):
        """此时 x 为输出的三种尺度的特征图 """
        x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):  # 逐个计算每种尺寸特征图的输出
            # 其中 85 组成结构为: 4(边框信息) + 1(置信度) + 80(属于哪个类别)
            bs, _, ny, nx = x[i].shape  # x(bs, 255, 20, 20) to x(bs, 3, 20, 20, 85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # 推理阶段: inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()  # 输出映射到 0 ~ 1
                # 疑问?：* stride: 下采样的倍数, 相乘之后映射回原图大小
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        """生成网格, 网格的大小为 (nx, ny)"""
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, model_cfg='yolov5s.yaml', ch=3, nc=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        # 首先需要加在模型的配置文件
        if type(model_cfg) is dict:
            self.md = model_cfg  # model dict
        else:  # is *.yaml
            with open(model_cfg) as f:
                self.md = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        if nc:  # 定义模型：需要分几类（不包括背景）
            self.md['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(self.md, ch=[ch])  # 根据模型的配置文件生成模型: model, savelist, ch_out
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        m.stride = torch.tensor([64 / x.shape[-2] for x in self.forward(torch.zeros(1, ch, 64, 64))])  # forward
        m.anchors /= m.stride.view(-1, 1, 1)
        self.stride = m.stride

        # Init weights, biases
        torch_utils.initialize_weights(self)
        self._initialize_biases()  # only run once
        torch_utils.model_info(self)
        print('')

    def forward(self, x, augment=False, profile=False):
        if augment:  # 输入增强
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []
            for i, xi in enumerate((x,
                                    torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale
                                    torch_utils.scale_img(x, s[1]),  # scale
                                    )):
                # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
                y.append(self.forward_once(xi)[0])

            y[1][..., :4] /= s[0]  # scale
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= s[1]  # scale
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:  # 针对 model 中的每一层，逐个计算
            if m.f != -1:  # 如果当前网络的输入非前一层的输出，if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                import thop
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # FLOPS
                t = torch_utils.time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((torch_utils.time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # 保存了每一层模型的输出

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for f, s in zip(m.f, m.stride):  # from
            mi = self.model[f % m.i]
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            with torch.no_grad():  # 如果不加, 会将加法操作计入模型的结构图中
                b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for f in sorted([x % m.i for x in m.f]):  # from
            b = self.model[f].bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%g Conv2d.bias:' + '%10.3g' * 6) % (f, *b[:5].mean(1).tolist(), b[5:].mean()))

    def _print_weights(self):
        for m in self.model.modules():
            if type(m) is Bottleneck:
                print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):
        """
        是否使用 conv + bn 混合的方式推理模型
        fuse model Conv2d() + BatchNorm2d() layers
        """
        print('Fusing layers...')
        for m in self.model.modules():
            if type(m) is Conv:
                m.conv = torch_utils.fuse_conv_and_bn(m.conv, m.bn)  # update conv
                m.bn = None  # remove batch-norm
                m.forward = m.fuseforward  # update forward
        torch_utils.model_info(self)


def parse_model(md, ch):
    """根据配置文件以及输出维度，返回模型：model_dict, input_channels(3)
    其中模型的配置文件每行一共有四个参数：
    例如：
        [[-1, 4], 1, Concat, [1]],
        第一个参数：[-1, 4] 表示当前层的输入
        第二个参数：[1] 表示将当前层重复多少次
        第三个参数：Concat 表示当前层执行的是什么操作
        第四个参数：[1] 表示 Concat 所需的参数配置
    """
    print('\n%3s%15s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = md['anchors'], md['nc'], md['depth_multiple'], md['width_multiple']
    na = (len(anchors[0]) // 2)  # 输出特征图每个像素点 anchor 的数量
    no = na * (nc + 5)  # 每个anchor的输出维度，number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(md['backbone'] + md['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # 通过eval, 实例化每层 eval strings,
        for j, a in enumerate(args):  # 将参数由字符串转换为整数或其他形式
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # 卷积块重复多少次: depth gain
        if m in [nn.Conv2d, Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, ConvPlus, BottleneckCSP]:
            c1, c2 = ch[f], args[0]  # ch[f] 表示输入的特征图维度, args[0]: 表示当前层输出的特征图维度
            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2  # make_divisible: 保证特征图的维度可以被 8 整除
            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2
            args = [c1, c2, *args[1:]]  # in-channels, out-channels, 网络层其他配置参数
            if m is BottleneckCSP:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            f = f or list(reversed([(-1 if j == i else j - 1) for j, x in enumerate(ch) if x == no]))
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        num_params = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, num_params  # attach index, 'from' index, type, number params

        print('%3s%15s%3s%10.0f  %-40s%-30s' % (i, f, n, num_params, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)  # 每一个 yolo-v5 的组成单元层
        ch.append(c2)  # 每结果一个层的输出特征维度
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = glob.glob('./**/' + opt.cfg, recursive=True)[0]  # find file
    device = torch_utils.select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    y = model(img, profile=True)
    print([y[0].shape] + [x.shape for x in y[1]])

    # ONNX export
    # model.model[-1].export = True
    # torch.onnx.export(model, img, f.replace('.yaml', '.onnx'), verbose=True, opset_version=11)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
