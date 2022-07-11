import sys, os
import cv2
from facenet_pytorch import MTCNN
import torch.nn.functional as F
import torch
import torch.nn as nn
import glob
import torch.optim as optim
import numpy as np
from time import perf_counter
from torchvision import transforms
import pandas as pd
import json
import face_recognition
import random
from concurrent.futures import ThreadPoolExecutor

# sys.path.insert(1,'helpers')
# sys.path.insert(1,'model')
# sys.path.insert(1,'weight')

from model.cvit import CViT
from helpers.helpers_read_video_1 import VideoReader
from helpers.helpers_face_extract_1 import FaceExtractor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

from helpers.blazeface import BlazeFace
facedet = BlazeFace().to(device)
facedet.load_weights("models/cvit_video/helpers/blazeface.pth")
facedet.load_anchors("models/cvit_video/helpers/anchors.npy")
_ = facedet.train(False)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

normalize_transform = transforms.Compose([
        transforms.Normalize(mean, std)]
)

tresh=50
sample='temp/'

ran = random.randint(0,400)
ran_min = abs(ran-1)

filenames = sorted([x for x in os.listdir(sample) if x[-4:] == ".mp4"]) #[ran_min, ran] -  select video randomly
mtcnn = MTCNN(select_largest=False, keep_all=True, post_process=False, device=device)

#load cvit model
model = CViT(image_size=224, patch_size=7, num_classes=2, channels=512,
             dim=1024, depth=6, heads=8, mlp_dim=2048)
model.to(device)

#checkpoint = torch.load('weight/deepfake_cvit_gpu_inference_ep_50.pth') # for GPU
checkpoint = torch.load('models/cvit_video/weight/deepfake_cvit_gpu_inference_ep_50.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
_ = model.eval()

def predict_on_video(dfdc_filenames, num_workers):
    def process_file(i):
        filename = dfdc_filenames[i]
        #print(filename)
        decCViT = predict(os.path.join(sample, filename), tresh, mtcnn)
        return decCViT

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        predictions = ex.map(process_file, range(len(dfdc_filenames)))
    return list(predictions)

# MTCNN face extctaction
def face_mtcnn(frame, face_tensor_mtcnn):
    mtcnn_frame = mtcnn.detect(frame)
    temp_face = np.zeros((5, 224, 224, 3), dtype=np.uint8)
    count=0
    if count<5 and (mtcnn_frame[0] is not None):
        for face_mt in mtcnn_frame[0]:
            x1, y1, width_, height_ = face_mt
            face_mt= frame[int(y1):int(height_), int(x1):int(width_)]
            if face_mt.size>0 and (count<5):
                resized_image_mtcnn = cv2.resize(face_mt, (224, 224), interpolation=cv2.INTER_AREA) 
                resized_image_mtcnn = cv2.cvtColor(resized_image_mtcnn, cv2.COLOR_RGB2BGR)
                temp_face[count]=resized_image_mtcnn
                count+=1
    if count == 0:
        return [],0
    return temp_face[:count], count

# face_recognition face extctaction
def face_face_rec(frame, face_tensor_face_rec):
    
    face_locations = face_recognition.face_locations(frame)
    temp_face = np.zeros((5, 224, 224, 3), dtype=np.uint8)
    count=0
    for face_location in face_locations:
        if count<5:
            top, right, bottom, left = face_location
            face_image = frame[top:bottom, left:right]
            face_image = cv2.resize(face_image, (224, 224), interpolation=cv2.INTER_AREA)
            face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            #face_image = Image.fromarray(face_image)
            temp_face[count]=face_image
            count+=1
    if count == 0:
        return [],0
    return temp_face[:count], count

# blazeface face extctaction
def face_blaze(video_path, filename, face_tensor_blaze):

    frames_per_video = 45  
    video_reader = VideoReader()
    video_read_fn = lambda x: video_reader.read_random_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn, facedet)
    
    faces = face_extractor.process_video(video_path)
    # Only look at one face per frame.
    #face_extractor.keep_only_best_face(faces)
  
    count_blaze=0
    
    temp_blaze = np.zeros((45, 224, 224, 3), dtype=np.uint8)
    for frame_data in faces:
        for face in frame_data["faces"]:
                if count_blaze<44:
                    resized_facefrm = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
                    resized_facefrm = cv2.cvtColor(resized_facefrm, cv2.COLOR_RGB2BGR)
                    temp_blaze[count_blaze]=resized_facefrm
                    count_blaze+=1
    if count_blaze==0:
        return [],0
    return temp_blaze, count_blaze

y_pred=0
def predict(filename, tresh, mtcnn):
    store_faces=[]
    
    face_tensor_mtcnn = np.zeros((30, 224, 224, 3), dtype=np.uint8)
    face_tensor_blaze = np.zeros((30, 224, 224, 3), dtype=np.uint8)
    face_tensor_face_rec = np.zeros((30, 224, 224, 3), dtype=np.uint8)
    
    curr_start_time = perf_counter()
    cap = cv2.VideoCapture(filename)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame_number = 0
    frame_count=int(length*0.1)
    frame_jump = 5 #int(frame_count/5)
    start_frame_number = 0

    loop = 0
    count_mtcn=0
    count_blaze=0
    count_face_rec = 0
    
    while cap.isOpened() and loop<frame_count:
        loop+=1
        success, frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
        
        if success:
            '''face_mtcnn_temp, temp_count_mtcn = face_mtcnn(frame, face_tensor_mtcnn)
            
            if len(face_mtcnn_temp) and temp_count_mtcn>0:
                kontrol = count_mtcn+temp_count_mtcn
                for f in face_mtcnn_temp:
                    if count_mtcn<=kontrol and (count_mtcn<29):
                        face_tensor_mtcnn[count_mtcn] = f
                        count_mtcn+=1'''
            
            
            face_rec,count = face_face_rec(frame, face_tensor_face_rec)
            
            if len(face_rec) and count>0:
                kontrol = count_face_rec+count
                for f in face_rec:
                    if count_face_rec<=kontrol and (count_face_rec<29):
                        face_tensor_face_rec[count_face_rec] = f
                        count_face_rec+=1

            start_frame_number+=frame_jump
    
    #face_tensor_blaze, count_blaze = face_blaze(filename, count_blaze, face_tensor_blaze)

    store_rec= face_tensor_face_rec[:count_face_rec]
    #store_blaze = face_tensor_blaze[:count_blaze]
    #store_mtcnn = face_tensor_mtcnn[:count_mtcn]
    
    dfdc_tensor=store_rec
    #dfdc_tensor=[*store_rec,*store_blaze,*store_mtcnn] #testing CViT using all three dl libraries - expect lower accuracy.
    
    dfdc_tensor = torch.tensor(dfdc_tensor, device=device).float()

    # Preprocess the images.
    dfdc_tensor = dfdc_tensor.permute((0, 3, 1, 2))

    for i in range(len(dfdc_tensor)):
        dfdc_tensor[i] = normalize_transform(dfdc_tensor[i] / 255.)
    
    # the tranformer accepts batch of <=32.
    if not len(non_empty(dfdc_tensor, df_len=-1, lower_bound=-1, upper_bound=-1, flag=False)):
        return torch.tensor(0.5).item()
        
    dfdc_tensor = dfdc_tensor.contiguous()
    df_len = len(dfdc_tensor)
    
    with torch.no_grad(): 
        
        thrtw =32
        if df_len<33:
            thrtw =df_len  
        y_predCViT = model(dfdc_tensor[0:thrtw])
        
        if df_len>32:
            dft = non_empty(dfdc_tensor, df_len, lower_bound=32, upper_bound=64, flag=True)
            if len(dft):
                y_predCViT = pred_tensor(y_predCViT, model(dft))
        if df_len>64:
            dft = non_empty(dfdc_tensor, df_len, lower_bound=64, upper_bound=90, flag=True)
            if len(dft):
                y_predCViT = pred_tensor(y_predCViT, model(dft))
        
        decCViT = pre_process_prediction(pred_sig(y_predCViT))
        #print('CViT', filename, "Prediction:",decCViT.item())
        return decCViT.item()

def non_empty(dfdc_tensor, df_len, lower_bound, upper_bound, flag):
    
    thrtw=df_len
    if df_len>=upper_bound:
        thrtw=upper_bound
        
    if flag==True:
        return dfdc_tensor[lower_bound:thrtw]
    elif flag==False:
        return dfdc_tensor
        
    return []
    
def pred_sig(dfdc_tensor):
    return torch.sigmoid(dfdc_tensor.squeeze())

def pred_tensor(dfdc_tensor, pre_tensor):
    return torch.cat((dfdc_tensor,pre_tensor),0)

def pre_process_prediction(y_pred):
    f=[]
    r=[]
    if len(y_pred)>2:
        for i, j in y_pred:
            f.append(i)
            r.append(j)
        f_c = sum(f)/len(f)
        r_c= sum(r)/len(r)
        if f_c>r_c:
            return f_c
        else:
            r_c = abs(1-r_c)
            return r_c
    else:
        return torch.tensor(0.5)
    
start_time = perf_counter()
predictions = predict_on_video(filenames, num_workers=4)
#print(predictions)
end_time = perf_counter()
#print("--- %s seconds ---" % (end_time - start_time))

# for testing DFDC dataset

'''metafile = sample+'metadata.json'
if os.path.isfile(metafile):
    with open(metafile) as data_file:
        data = json.load(data_file)


def real_or_fake(filenames, predictions): 
    j=0
    correct = 0
    label="REAL"
    for i in filenames:
        if data[i]['label'] == 'REAL' and predictions[j]<0.5:
            correct+=1
            label="REAL"
        if data[i]['label'] == 'FAKE' and predictions[j]>=0.5:
            correct+=1
            label="FAKE"
        
        print('Filname:',i, label)
        j+=1
        
    return correct
print('Accuracy: ', (real_or_fake(filenames, predictions)/len(filenames))*100)'''


def real_or_fake(filenames, predictions): 
    j=0
    correct = 0
    label="REAL"
    for i in filenames:
        if predictions[j]<0.5:
            label="REAL"
        if predictions[j]>=0.5:
            label="FAKE"
        
        #print('Filname:',i,label)
        j+=1
        
real_or_fake(filenames, predictions)
submission_dfcvit_nov16 = pd.DataFrame({"filename": filenames, "label": predictions})
submission_dfcvit_nov16.to_csv("models/cvit_video/cvit_predictions.csv", index=False)
with open("models/cvit_video/result.txt", "w") as text_file:
        text_file.write(str(predictions[0]))
    
