import torch.nn


def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()
    return module_defs


def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    # options['gpus'] = '0,1,2,3'
    # options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


def generator_config(module_defs, cfg_file, mask=None):
    """
    model 的组成需要通过 module_list 生成
    根据模型生成对应的配置文件, 在剪枝的时候，mask是一个0\1矩阵或向量，表明哪些channel需要减掉
    """
    mask_filters = []  # 每个filters是否可用
    if mask:
        module_defs['filters'] = 10
    """
    module_defs = []
    layer_list = list(model.named_children())[0][1]
    for seq in layer_list:
        module_def = {}
        for name, layer in seq.named_children():
            if 'conv' in name:
                module_def['batch_normalize'] = 0
                module_def['activation'] = 'linear'
                if 'transpose' in name:  # 转置卷积
                    module_def['type'] = 'convtranspose'
                else:
                    module_def['type'] = 'conv'
                module_def['filters'] = layer.out_channels
                module_def['pad'] = layer.padding[0]
                module_def['size'] = layer.kernel_size[0]
                module_def['dilation'] = layer.dilation
                module_def['groups'] = layer.groups
            elif 'upsample' in name:
                module_def['type'] = 'upsample'
                module_def['stride'] = layer.scale_factor
            elif 'batch_norm' in name:
                module_def['batch_normalize'] = 1
            elif 'shortcut' in name:
                module_def['type'] = 'shortcut'
                module_def['from'] = 'shortcut'
                module_def['activation'] = 'linear'
            else:
                pass
        module_defs.append(module_def)
    """
    with open(cfg_file, 'w') as f:
        for module_def in module_defs:
            f.write(f"[{module_def['type']}]\n")
            for key, value in module_def.items():
                if key != 'type':
                    f.write(f"{key}={value}\n")
            f.write("\n")
    return cfg_file



