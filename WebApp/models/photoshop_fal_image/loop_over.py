import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import time
import sys
import torch
from PIL import Image
import torchvision.transforms as transforms

from networks.drn_seg import DRNSub
from utils.tools import *
from utils.visualize import *


def load_classifier(model_path, gpu_id):
    if torch.cuda.is_available() and gpu_id != -1:
        device = 'cuda:{}'.format(gpu_id)
    else:
        device = 'cpu'
    model = DRNSub(1)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.to(device)
    model.device = device
    model.eval()
    return model


tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])
def classify_fake(model, img_path, no_crop=False,
                  model_file='utils/dlib_face_detector/mmod_human_face_detector.dat'):
    # Data preprocessing
    im_w, im_h = Image.open(img_path).size
    if no_crop:
        face = Image.open(img_path).convert('RGB')
    else:
        faces = face_detection(img_path, verbose=False, model_file=model_file)
        if len(faces) == 0:
            print("no face detected by dlib, exiting")
            sys.exit()
        face, box = faces[0]
    face = resize_shorter_side(face, 400)[0]
    face_tens = tf(face).to(model.device)

    # Prediction
    with torch.no_grad():
        prob = model(face_tens.unsqueeze(0))[0].sigmoid().cpu().item()
    return prob

def predictor_photoshop(path):
    gpu_id = 0
    model_path = "models/photoshop_fal_image/weights/global.pth"
    model = load_classifier(model_path, gpu_id)
    prob = classify_fake(model, path, True)
    #print("Probibility being modified by Photoshop FAL: {:.2f}%".format(prob*100))
    #print(prob)
    return prob

#predictor_photoshop("examples/modified.jpg")


directory = "./temp/"

directory = directory + "/"

file_names = []
deepfake = []
time_array = []

#for files in os.listdir(directory):
#start_time = time.time()
#img = cv2.imread(directory + files)
probab = predictor_photoshop(directory + "delete.jpg")
#processing_time = time.time() - start_time
with open("models/photoshop_fal_image/result.txt", "w") as text_file:
    #text_file.write(str(directory + files))
    #text_file.write("\t")
    text_file.write(str(probab))
    #text_file.write("\t")
    #text_file.write(str(processing_time))
    #text_file.write("\n")