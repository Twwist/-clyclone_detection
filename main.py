from PIL import Image
import torch
import cv2
import pandas

model = torch.hub.load('ultralytics/yolov5', 'custom', path='Веса модели хорошей.pt', force_reload=True)


from os import listdir
from os.path import isfile, join

folder_name = "photos"
mypath = fr"{folder_name}"
final_path = r"result"
files = listdir(mypath)

for file in files:
    filepath = mypath + "\\" + file
    img = Image.open(filepath)
    results = model(img)
    labels = results.pandas().xyxy
    with open(final_path + "\\" + file[0:len(file) - 3] + "txt", 'w') as resfile:
        print(labels)
        if len(labels) == 0:
            continue
        for i in range(labels[0].shape[0]):

            if len(labels[0]) > 0:
                lx = labels[0]["xmax"][i] - labels[0]["xmin"][i]
                ly = labels[0]["ymax"][i] - labels[0]["ymin"][i]
                mid = (labels[0]["xmin"][i] + lx / 2, labels[0]["ymin"][i] + ly / 2)
                resfile.write(", ".join(["Class_" + str(labels[0]["class"][i]), str(mid[0]), str(mid[1]), str(lx), str(ly)]) + '\n')
