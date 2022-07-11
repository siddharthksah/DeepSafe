import pandas as pd
import streamlit as st
import requests
#from utils.download_youtube_video import download_video
from random import randrange

def check_video_url_if_youtube(url):

    #checker_url = "https://www.youtube.com/oembed?url=http://www.youtube.com/watch?v="
    video_url = url

    request = requests.get(video_url)

    return request.status_code == 200



def examples():
    # df = (pd.read_csv("examples/deepfake_examples_url.csv"))
    df = pd.read_csv('https://docs.google.com/spreadsheets/d/' + 
                   '1H_kMDdmw3de4BRaf8i7G8x2NgYUM0QkpyTKusb5yC3w' +
                   '/export?gid=0&format=csv'
                   # Set first column as rownames in data frame
                   #index_col=0,
                   # Parse column values to datetime
                   #parse_dates=['Quradate']
                  )
    first_column = df.iloc[:, 0]
    count = (randrange(len(first_column)))
    url = first_column[count]
    if check_video_url_if_youtube(url) == True:
        st.video(url)
    else:
        st.experimental_rerun()
    if st.button("Next"):
        st.experimental_rerun()
        # download_video(url)
        # video_file = open('./temp/delete.mp4', 'rb')
        # video_bytes = video_file.read()
        # st.video(video_bytes)

    #  if check_video_url_if_youtube(urls) == True:
    #      st.video(urls)

#examples()