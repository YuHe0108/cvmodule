# 人脸识别
import matplotlib.pyplot as plt
import face_recognition
import cv2 as cv
import pathlib
import time
import os

image = face_recognition.load_image_file(r"D:\Github\DL\face_recognition-master\tests\test_images\obama.jpg")
feature_1 = face_recognition.face_encodings(image)[0]

features = []
for d, sub_d, file_name in os.walk(r'D:\Github\DL\face_recognition-master\tests\test_images'):
    for name in file_name:
        file_path = os.path.join(d, name)
        image_2 = face_recognition.load_image_file(file_path)
        feature_2 = face_recognition.face_encodings(image_2)[0]
        features.append(feature_2)

print(len(features))
t = time.time()
results = face_recognition.compare_faces(features, feature_1)
print(time.time() - t)
print(results)
