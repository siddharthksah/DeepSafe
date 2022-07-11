import os
from PIL import Image
import numpy as np
import cv2
import streamlit as st



@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img


def save_image(image_file):
    import shutil
    if image_file is not None:
        img = load_image(image_file)
        img = img.convert('RGB')
        path = 'temp/'
        img.save(os.path.join(path , 'delete.jpg'))
        if os.path.exists("temp/delete.mp4"):
            os.remove("temp/delete.mp4") # one file at a time
    else:
        st.write("Corrupted file!")