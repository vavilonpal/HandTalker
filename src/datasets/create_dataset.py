# Module for working witch OS functions
import os
# OpenCV (Open Source Computer Vision Library),
import cv2


DATA_DIR = '/data'

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_,img_path))
        # Convert video color fro bgr to rgb
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
