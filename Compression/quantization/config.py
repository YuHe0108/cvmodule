import torch

init_epoch_lr = [(10, 0.01), (20, 0.001), (20, 0.0001)]
sparisity_list = [50, 60, 70, 80, 90]

finetune_epoch_lr = [
    [(3, 0.01), (3, 0.001), (3, 0.0001)],
    [(6, 0.01), (6, 0.001), (6, 0.0001)],
    [(9, 0.01), (9, 0.001), (9, 0.0001)],
    [(12, 0.01), (12, 0.001), (12, 0.0001)],
    [(20, 0.01), (20, 0.001), (20, 0.0001)]
]

batch_size = 128
checkpoint = 'checkpoint'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
