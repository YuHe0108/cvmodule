# import torch
# import cv2 as cv
#
# #
# model = torch.load(r'D:\Vortex\Project_2\sdyd_box\data\models\zhxjc\1.0.0\zhxjc_yolov5s_small.pt', map_location='cpu')[
#     'model']
# img = cv.imread(r'D:\Vortex\Project_4_RKNN\reference\yolov5\bus.jpg')
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#
import torch
from Detection.yolov5.models.experimental import attempt_load

model = attempt_load(r'D:\Vortex\SVN\遗留物\xj3_left_detect\v1.0-left-detection.pt', map_location=torch.device('cpu'))
m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
print(m.anchor_grid)
