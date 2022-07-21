from tabnanny import check
import requests,os
import streamlit as st
import urllib
import urllib.request
import validators

def check_if_url_is_valid(url):
    return validators.url(url)

def is_url_image(image_url):
   image_formats = ("image/png", "image/jpeg", "image/jpg")
   r = requests.head(image_url)
   if r.headers["content-type"] in image_formats:
      return True
   return False

def is_url_video(video_url):
    req = urllib.request.Request(video_url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
    r = urllib.request.urlopen(req)
    #st.write((r.getheader('Content-Type')))
    if (r.getheader('Content-Type')) == 'video/mp4':
        return True
    return False


def download_media(url):
    if check_if_url_is_valid(url):
        #st.write("URL OK")
        if is_url_image(url) == True:
            img_data = requests.get(url).content
            with open('temp/delete.jpg', 'wb') as handler:
                handler.write(img_data)
        elif is_url_video(url) == True:
            try:
                file_name = 'temp/delete.mp4' 
                rsp = urllib.request.urlopen(url)
                with open(file_name,'wb') as f:
                    f.write(rsp.read()) 
                #st.write("video")
            except:
                st.write("The provided URL can not be accessed at this moment!")
        else:
            try:
                file_name = 'temp/delete.mp4' 
                rsp = urllib.request.urlopen(url)
                with open(file_name,'wb') as f:
                    f.write(rsp.read())
            except:
                st.write("Error! No media found!")
    else:
        st.write("Not a valid URL!")



def download_image_from_url(url):
    if check_if_url_is_valid(url):
        #st.write("URL OK")
        if is_url_image(url) == True:
            img_data = requests.get(url).content
            with open('temp/delete.jpg', 'wb') as handler:
                handler.write(img_data)

    else:
        st.error("Not a valid URL!")

    return str(os.path.exists("temp/delete.jpg")) # Or folder, will return true or false