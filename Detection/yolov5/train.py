import argparse
import torch
import glob

from components import torch_utils
from models import yolo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    opt = parser.parse_args()

    opt.cfg = glob.glob('./**/' + opt.cfg, recursive=True)[0]  # find file
    device = torch_utils.select_device(opt.device)
    # Create model
    model = yolo.Model(opt.cfg).to(device)
    model.train()

    # Profile
    img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 416, 416).to(device)
    y = model(img, profile=True)
    for z in y:
        print(z.shape)
    print([y[0].shape] + [x.shape for x in y[1]])