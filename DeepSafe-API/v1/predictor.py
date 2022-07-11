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
import shutil


# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('-f','--file', default='examples_realfakedir')
# parser.add_argument('-m','--model_path', type=str, default='weights/blur_jpg_prob0.5.pth')
# parser.add_argument('-c','--crop', type=int, default=None, help='by default, do not crop. specify crop size')
# parser.add_argument('--use_cpu', action='store_true', help='uses gpu by default, turn on to use cpu')

# opt = parser.parse_args()



def predictor_CNN():
    model_path = "./weights/blur_jpg_prob0.5.pth"
    crop = None
    use_cpu = True
    #print(os.getcwd)
    for item in os.listdir("./tempDir/"):
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
    else:
        pass
        # print('Not cropping')
    trans = transforms.Compose(trans_init + [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = trans(Image.open("./tempDir/" + file).convert('RGB'))

    with torch.no_grad():
        in_tens = img.unsqueeze(0)
        if(not use_cpu):
            in_tens = in_tens.cuda()
        prob = model(in_tens).sigmoid().item()

    # print('probability of being synthetic: {:.2f}%'.format(prob * 100))
    print(format(1-prob, ".2f"))

    #deleting the temp folder

    # import shutil

    # shutil.rmtree('tempDir', ignore_errors=True)

    return (format(1-prob, ".2f"))

# predictor_CNN()