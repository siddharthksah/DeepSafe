import os
from dfdc_prediction import predictor_DFDC
import time

directory = "/home/siddharth/Desktop/DeepFake_Dataset/celeba/img_align_celeba/"

directory = directory + "/"

file_names = []
deepfake = []
time_array = []

for files in os.listdir(directory):
    start_time = time.time()
    #img = cv2.imread(directory + files)
    probab = predictor_DFDC(directory + files)
    processing_time = time.time() - start_time
    with open("img_align_celeba.txt", "a") as text_file:
        text_file.write(str(directory + files))
        text_file.write("\t")
        text_file.write(str(probab))
        text_file.write("\t")
        text_file.write(str(processing_time))
        text_file.write("\n")
