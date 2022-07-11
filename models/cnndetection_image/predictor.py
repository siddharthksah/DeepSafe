import os
import sys
import torch
import torch.nn
import argparse
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from networks.resnet import resnet50


# if not os.path.exists('tempDir'):
#     os.makedirs('tempDir')

model_path = "models/cnndetection_image/weights/blur_jpg_prob0.5.pth"
#print(model_path)
crop = None
use_cpu = True
# print(os.getcwd)
#for item in os.listdir("./tempDir/"):
    #file = item
    #print(item)

def predictor_cnn():
    for item in os.listdir("./temp/"):
        #print(item)
        if item == "delete.jpg":
            file = item
        #print(item)
    model = resnet50(num_classes=1)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    if(not use_cpu):
        model.cuda()
        model.eval()

    # Transform
    trans_init = []
    if(crop is not None):
        trans_init = [transforms.CenterCrop(crop),]
        # print('Cropping to [%i]'%crop)

        # print('Not cropping')
    trans = transforms.Compose(trans_init + [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    #print(file)
    img = trans(Image.open("./temp/" + file).convert('RGB'))

    with torch.no_grad():
        in_tens = img.unsqueeze(0)
        if(not use_cpu):
            in_tens = in_tens.cuda()
        prob = model(in_tens).sigmoid().item()

    # print('probability of being synthetic: {:.2f}%'.format(prob * 100))
    #print(format(prob, ".2f"))

    with open("models/cnndetection_image/result.txt", "w") as text_file:
        text_file.write(str(prob))

    #deleting the temp folder

    #import shutil

    #shutil.rmtree('tempDir', ignore_errors=True)

    return (format(prob, ".2f"))

predictor_cnn()

