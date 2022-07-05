import torch
import argparse
import cv2 as cv
import numpy as np
import torchvision
from PIL import Image
from torchvision import transforms

from utils import img_utils
from Detection.DETR.models import backbone, detr, transformer

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true', default=True,
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false', default=False,
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', default=True)
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()

backbone = backbone.build_backbone(args)
transformer = transformer.build_transformer(args)

model = detr.DETR(
    backbone,
    transformer,
    num_classes=91,
    num_queries=args.num_queries,
    aux_loss=args.aux_loss,
)

weight = torch.load(r'Detection\DETR\detr-r50-e632da11.pth')
model.load_state_dict(weight['model'])

img = cv.imread(r'D:\Vortex\YOLO\YOLOv6-main\data\images\image3.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = img_utils.letterbox(img, (800, 800))
# inputs = img_utils.image_process(img, '01')
inputs = Image.fromarray(img)

transform = transforms.Compose([
    transforms.Resize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
inputs = transform(inputs)
inputs = torch.unsqueeze(inputs, 0)

model.eval().to(DEVICE)
with torch.no_grad():
    inputs = torch.tensor(inputs).to(DEVICE)
    outputs = model(inputs)

h, w = img.shape[:2]
pred_logits = outputs['pred_logits'][0]
pred_boxes = outputs['pred_boxes'][0]

# 相当于预测了100个框，选择每个预测框 预测所有类别的最大值, 不包含背景类别
max_output = pred_logits[:, :91].softmax(-1).max(-1)
top_k = max_output.values.topk(5)  # 前五个最大值

pred_logits = pred_logits[top_k.indices].to('cpu')
pred_boxes = pred_boxes[top_k.indices].to('cpu')

print(pred_boxes.shape, pred_logits.shape)

for logits, box in zip(pred_logits, pred_boxes):
    cls = logits.argmax()
    if cls >= 91:
        continue
    print(cls, box, logits.softmax(-1)[cls])
    box = box * torch.Tensor([h, w, h, w])
    x, y, w, h = box.detach().numpy().astype(np.int)
    x1, y1 = x - w // 2, y - h // 2
    x2, y2 = x + w // 2, y + h // 2
    img = cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()
