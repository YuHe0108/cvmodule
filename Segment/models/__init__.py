"""
包含了常见的基于深度学习的图像分割模型: FCN、UNet、DeepLab-v3、SegNet、PSPNet等,
所有模型的输出没有经过激活函数。
"""
from . import deeplab, fcn, hr_net, psp_net, refine_net, seg_net, unet, unet_bn, deeplabv3_plus


def get_seg_model(model_name, input_shape, num_classes, dims=32, **kwargs):
    model_name = str(model_name).lower().replace(' ', '')
    if model_name == 'unet':
        seg_model = unet.unet_model(input_shape, num_classes, dims, **kwargs)
    elif model_name == 'unet_bn':
        seg_model = unet_bn.unet_model(input_shape, num_classes, dims, **kwargs)
    elif model_name == 'fcn8s':
        seg_model = fcn.fcn_8_vgg(input_shape, num_classes)
    elif model_name == 'fcn16s':
        seg_model = fcn.fcn8_model(input_shape, num_classes)
    elif model_name == 'fcn32s':
        seg_model = fcn.fcn_32_vgg(input_shape, num_classes)
    elif model_name == 'segnet':
        seg_model = seg_net.segnet_model(input_shape, dims, num_classes)
    elif model_name == 'deeplabv3+':
        seg_model = deeplabv3_plus.deeplab_v3_plus(input_shape, num_classes, **kwargs)
    elif model_name == 'pspnet':
        seg_model = psp_net.pspnet(input_shape, num_classes, dims=dims ** kwargs)
    elif model_name == 'refinenet':
        seg_model = refine_net.build_refine_net(input_shape, num_classes, frontend_trainable=False)
    else:
        raise 'no model name {}'.format(model_name)
    return seg_model
