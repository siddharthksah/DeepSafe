import os
import json
import numpy as np
import time
import face_recognition
import cv2
import shutil
from time import perf_counter
import helpers_face_extract_1
import helpers_read_video_1
import blazeface
import sys
import torch
from blazeface import BlazeFace

# code used to extract DFDC dataset

dir_path = "dfdc_data/" 
train_path = "dfdc_data/training_face_"
validation_path = "dfdc_data/validation_face_"
test_path = "dfdc_data/test_face_"
  
# load DFDC json
def load_metadata(dir_path):
    metafile = dir_path+'metadata.json'
     
    if os.path.isfile(metafile):
        with open(metafile) as data_file:
            data = json.load(data_file)
    else:
        return 1
    
    return data

def extract_face(dir_path):
    # iterate over DFDC dataset
    for item in sorted(os.listdir(dir_path)):
        
        file_num = int(item[16:])
        destination = train_path
        
        if (file_num > 34 and file_num<45):
            destination = validation_path 
        
        if (file_num > 45):
            destination = test_path
        
        meta_full_path = os.path.join(dir_path, item)
        
        if os.path.isdir(meta_full_path):
            data = load_metadata(meta_full_path+'/')
            
            if data != 1:
                
                if not os.path.exists(destination+str(file_num)):
                    os.makedirs(destination+str(file_num))
                
                if not os.path.isfile(destination+str(file_num)+'/metadata.json'):
                    shutil.copy2(dir_path+item+'/metadata.json', destination+str(file_num)+'/metadata.json')
                  
                filtered_files = filter_unique_files(data)
                
                for filename in filtered_files:
                    # check if the file name is found in metadata, and its label
                    if filename.endswith(".mp4") and os.path.isfile(dir_path+item+'/'+filename):
                        label = data[filename]['label'].lower()
                        # append fake video names with their corresponding real video names
                        original = ''
                        if data[filename]['label'].lower() == 'fake':
                            original = '_'+data[filename]['original'][:-4]
                        image_path = destination+str(file_num)+'/'+label
                        if not os.path.exists(image_path):
                            os.makedirs(image_path)
                            
                        process_video(dir_path+item+'/'+filename, filename, image_path, original)

# access video
def process_video(video_path, filename, image_path, original):
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    facedet = BlazeFace().to(gpu)
    facedet.load_weights("blazeface.pth")
    facedet.load_anchors("anchors.npy")
    _ = facedet.train(False)
    
    from helpers_read_video_1 import VideoReader
    from helpers_face_extract_1 import FaceExtractor

    frames_per_video = 10
     
    video_reader = VideoReader()
    video_read_fn = lambda x: video_reader.read_random_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn, facedet)
    
    faces = face_extractor.process_video(video_path)
    # Only look at one face per frame.
    face_extractor.keep_only_best_face(faces)
    n = 0
    for frame_data in faces:
        for face in frame_data["faces"]:
            face_locations = face_recognition.face_locations(face)
            for face_location in face_locations:
            
                top, right, bottom, left = face_location
                face_image = face[top:bottom, left:right]
                resized_face = cv2.resize(face_image, (224, 224), interpolation=cv2.INTER_AREA)
                resized_face = cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR)
            
                cv2.imwrite(image_path+"/"+filename[:-4]+original+"_"+str(n)+".jpg",resized_face, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                
                n += 1

def filter_unique_files(metadata):
    fake=[]
    original=[]
    
    for file_dp in metadata:
        if (('original' in metadata[file_dp]) and (metadata[file_dp]['original'] not in original) and (metadata[file_dp]['original'] is not None)):
            original.append(metadata[file_dp]['original'])
            fake.append(file_dp)
    return np.array([[i, j] for i, j in zip(fake, original)]).ravel()

start_time = perf_counter()
extract_face(dir_path)
end_time = perf_counter()
print("--- %s seconds ---" % (end_time - start_time))
