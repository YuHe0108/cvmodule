import pathlib
import hashlib
import shutil
import cv2
import os


def remove_same_file(root, patience=-1):
    not_same_count = 0
    files = []
    md5_vals = []
    for path in pathlib.Path(root).iterdir():
        path = str(path)
        if str(True) not in path and str(False) not in path:
            continue
        files.append(path)
        with open(path, 'rb') as f:
            data = f.read()
        md5_val = hashlib.md5(data).hexdigest()
        md5_vals.append(md5_val)

    for i, f1 in enumerate(files):
        if i % 2 != 0 or not os.path.exists(f1):
            continue
        for j, f2 in enumerate(files):
            if j % 2 != 0 or i == j:
                continue
            if md5_vals[i] == md5_vals[j] and os.path.exists(files[j]) and os.path.exists(files[j + 1]):
                os.remove(files[j])
                os.remove(files[j + 1])
    return


if __name__ == "__main__":
    # remove_same_file(r"C:\Users\yuhe\Desktop\images\outer_violation_images")
    pass
