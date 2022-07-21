from PIL import Image
import os, sys

path = "/home/siddharth/Desktop/linkedin/stylegan2-ada/dataset/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((256,256), Image.ANTIALIAS)
            imResize.save(f + '.jpg', 'JPEG', quality=100)

resize()