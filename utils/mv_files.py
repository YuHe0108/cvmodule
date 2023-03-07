import pathlib
import shutil
import os

# cnt = 777
#
# for a, b, c in os.walk(r'/mnt/YuHe/data/SDYD/left/遗留物（所有）20220919/遗留物验证集20220628/夜晚/业务遗留物'):
#     print(a, b, c)
#     if len(b) == 0 and len(c) > 0:
#         for path in c:
#             img_path = os.path.join(a, path)
#             print(img_path)
#             shutil.copy(img_path, os.path.join(r'/mnt/YuHe/data/SDYD/left/遗留物（所有）20220919/total/1', f'{cnt}.jpg'))
#             cnt += 1

# root_dir = '/mnt/YuHe/data/SDYD/tools/history/20221216/raw'
# save_dir = '/mnt/YuHe/data/SDYD/tools/history/20221216/ori_imgs'
# img_suffixes = ['.jpg', '.png']
# for path in pathlib.Path(root_dir).iterdir():
#     if path.suffix not in img_suffixes:
#         continue
#     shutil.move(str(path), save_dir)
