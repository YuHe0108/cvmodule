import torch
import numpy as np
import torch.nn as nn
import torchsummary

from layer_utils import to_cpu
import torch.nn.functional as F
from site_config import parse_model_config


def creat_activation(activation_name):
    if activation_name == "leaky":
        return nn.LeakyReLU(0.1, inplace=True)
    elif activation_name == "relu":
        return nn.ReLU(inplace=True)
    elif activation_name == 'linear_ac':
        return LinearActivation()
    else:
        raise NotImplementedError


def create_modules(module_defs):
    """
    根据模块的定义返回 单元层
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]  # 输入图像的通道数
    module_list = nn.ModuleList()  # 一定要用ModuleList()才能被torch识别为module并进行管理，不能用list！
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(f"conv_{module_i}",
                               nn.Conv2d(in_channels=output_filters[-1], out_channels=filters, kernel_size=kernel_size,
                                         stride=int(module_def["stride"]), padding=pad, bias=not bn))
            if bn:
                modules.add_module(f"batch_norm_{module_i}",
                                   nn.BatchNorm2d(filters))  # BN(momentum=0.9, eps=1e-5)
            modules.add_module(f'{module_def["activation"]}_{module_i}', creat_activation(module_def['activation']))
        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)
        elif module_def["type"] == "flatten":  # 展平操作
            modules.add_module(f"flatten_{module_i}", nn.Flatten())
            filters = int(module_def["units"])
        elif module_def["type"] == "linear":  # 全连接层
            filters = int(module_def['units'])
            drop_rate = float(module_def['dropout'])
            modules.add_module(f"linear_{module_i}", nn.Linear(output_filters[-1], filters))  # 线性层
            modules.add_module(f"dropout_{module_i}", nn.Dropout(drop_rate))  # dropout 层
            modules.add_module(f'{module_def["activation"]}_{module_i}', creat_activation(module_def['activation']))
        elif module_def["type"] == "upsample":
            upsample = UpSample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)
        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())  # 使用emptyLayer占位
        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())
        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_size = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        else:
            raise NotImplementedError
        module_list.append(modules)  # Register module list and number of output filters
        output_filters.append(filters)  # filter保存了每个卷积层的输出 filter 的信息
    return hyperparams, module_list


class UpSample(nn.Module):
    """ nn.UpSample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(UpSample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):  # 只是为了占位，以便处理route层和shortcut层
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class LinearActivation(nn.Module):  # 线性激活层
    def __init__(self):
        super(LinearActivation, self).__init__()

    def forward(self, x):
        return x


class Add(nn.Module):
    def __init__(self, layer_feature):
        super(Add, self).__init__()
        self.layer_feature = layer_feature

    def forward(self, x):
        return x + self.layer_feature


class GenModel(nn.Module):
    def __init__(self, config_path, img_size=416):
        super().__init__()
        if isinstance(config_path, str):
            self.module_defs = parse_model_config(config_path)  # 根据配置文件，解析模型信息
        elif isinstance(config_path, list):
            self.module_defs = config_path
        else:
            raise NotImplementedError

        self.seen = 0
        self.img_size = img_size
        self.hyperparams, self.module_list = create_modules(self.module_defs)

    def initialize(self):
        """模型的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        layer_outputs = []  # 每一层的输出结果进行保存
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool", 'flatten', 'linear']:
                x = module(x)
            elif module_def["type"] == "route":  # 特征图拼接: inception 结构
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":  # shortcut: res-block结构
                layer_i = int(module_def["from"])
                x = layer_outputs[layer_i] + layer_outputs[-1]
                if module_def["activation"] == "leaky":
                    x = F.leaky_relu(x, negative_slope=0.1, inplace=True)
                elif module_def["activation"] == "relu":
                    x = F.relu(x, inplace=True)
                elif module_def["activation"] == "linear":
                    x = x
            else:
                raise NotImplementedError
            layer_outputs.append(x)  # 将每个块的output都保存起来
        return layer_outputs[-1]

    def load_model_weights(self, weights_path):
        """ 加载模型的权重信息 """
        weights = torch.load(weights_path, map_location='cpu')['state_dict']
        keys = list(weights.keys())[::-1]
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            for layer in module:
                if isinstance(layer, nn.Conv2d):
                    layer.weight.data.copy_(weights[keys.pop()])
                    if layer.bias:
                        layer.bias.data.copy_(weights[keys.pop()])
                elif isinstance(layer, nn.Linear):
                    layer.weight.data.copy_(weights[keys.pop()])
                    layer.bias.data.copy_(weights[keys.pop()])
                elif isinstance(layer, nn.BatchNorm2d):
                    layer.weight.data.copy_(weights[keys.pop()])
                    layer.bias.data.copy_(weights[keys.pop()])
                    layer.running_mean.copy_(weights[keys.pop()])
                    layer.running_var.copy_(weights[keys.pop()])
                    layer.num_batches_tracked.copy_(weights[keys.pop()])


#  example: YOLO layer 和 darknet 模型的构造方法
class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view(
            (1, self.num_anchors, 1, 1))  # anchor_w的范围是[0, grid_size](416下),浮点型数值
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2)  # num_samples, num_anchors, grid_size, grid_size, self.num_classes + 5
                .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:  # 不用每次都计算，只有在输入图片大小第一次发生变化时计算
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)  # 生成形状与prediction[..., :4]相同的张量
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w  # anchor_w的范围是[0,grid_size](416下)，浮点型变量
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),  # num_samples, num_anchors*grid_size*grid_size, 85
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            # TODO：这里没有针对wh的损失进行加权处理
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()

        if isinstance(config_path, str):
            self.module_defs = parse_model_config(config_path)
        elif isinstance(config_path, list):
            self.module_defs = config_path

        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0] for layer in self.module_list if
                            hasattr(layer[0], "metrics")]  # layer是个nn.Sequential()
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        img_dim = x.shape[2]  # 取决于输入图片的大小，因为是正方形输入，所以只考虑height
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":  # [82, 94, 106] for yolov3
                x, layer_loss = module[0](x, targets, img_dim)  # module是nn.Sequential()，所以要取[0]
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)  # 将每个块的output都保存起来
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))  # 只保存yolo层的output
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                    # Load conv. weights
                    num_w = conv_layer.weight.numel()
                    conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                    conv_layer.weight.data.copy_(conv_w)
                    ptr += num_w
                else:
                    # 对于yolov3.weights,不带bn的卷积层就是YOLO前的卷积层
                    if "yolov3.weights" in weights_path:
                        num_b = 255
                        ptr += num_b
                        num_w = int(self.module_defs[i - 1]["filters"]) * 255
                        ptr += num_w
                    else:
                        # Load conv. bias
                        num_b = conv_layer.bias.numel()
                        conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                        conv_layer.bias.data.copy_(conv_b)
                        ptr += num_b
                        # Load conv. weights
                        num_w = conv_layer.weight.numel()
                        conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                        conv_layer.weight.data.copy_(conv_w)
                        ptr += num_w
        # 确保指针到达权重的最后一个位置
        assert ptr == len(weights)

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()


class M(nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.module_list = nn.ModuleList()
        self.module_list.add_module("conv_1", nn.Conv2d(3, 10, 3, 1))
        self.module_list.add_module("bn_1", nn.BatchNorm2d(10))
        self.module_list.add_module("conv_2", nn.Conv2d(3, 10, 3, 1))
        self.module_list.add_module("relu", nn.ReLU())

    def forward(self, x):
        for layer in self.module_list:
            x = layer(x)
        return x


if __name__ == '__main__':
    cfg = parse_model_config('model.cfg')
    model = GenModel(cfg)
    inputs = torch.normal(1, 1, size=(2, 3, 32, 32))
    print(model(inputs))
    torchsummary.summary(model, (3, 32, 32))
    print(model)
