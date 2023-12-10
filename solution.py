from PIL import Image
import torch
import os
from os import listdir
from os.path import isfile, join
from ultralytics import YOLO

model = YOLO('weights.pt')
folder_path = "result"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

folder_path = "result"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

folder_name = "photos"
mypath = fr"{folder_name}"
final_path = r"result"
files = listdir(mypath)

for file in files:
    filepath = mypath + "\\" + file
    img = Image.open(filepath)
    results = model(img)
    labels_2 = results[0].boxes.xyxy.tolist()
    with open(final_path + "\\" + file[0:len(file) - 3] + "txt", 'w') as resfile:
        if len(labels_2) == 0:
            continue
        for lp in labels_2:
            if (len(lp) == 0):
                continue
            try:
                xmin_pred, ymin_pred, xmax_pred, ymax_pred = lp[0], lp[1], lp[2], lp[3]
                lx = xmax_pred - xmin_pred
                ly = ymax_pred - ymin_pred
                mid = (xmin_pred + lx / 2, ymin_pred + ly / 2)
                resfile.write(", ".join(["0", str(mid[0]), str(mid[1]), str(lx), str(ly)]) + "\n")
            except Exception:
                continue
