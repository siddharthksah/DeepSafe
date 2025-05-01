# @author Siddharth
# @website https://siddharthksah.github.io/
import streamlit as st

# favicon and page configs
favicon = './assets/icon.png'
st.set_page_config(page_title='DeepSafe', page_icon = favicon, initial_sidebar_state = 'expanded')

import warnings
warnings.filterwarnings("ignore")

# importing the necessary packages
from audioop import add
from datetime import datetime
from distutils.util import rfc822_escape
import time, random
import matplotlib.pyplot as plt
rgb = [[random.random(), random.random(), random.random()]]
import csv
import numpy as np
from PIL import Image
from skimage import transform
import os, cv2
from PIL import Image
import pandas as pd
from statistics import mean
from bokeh.models.widgets import Div
import json
import streamlit_ext as ste
import os
import json
from datetime import datetime
import streamlit as st
import pandas as pd
from PIL import Image
import importlib
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score


# import from utils folder
from utils.save_uploaded_image_temp import save_image
from utils.save_uploaded_video_temp import save_video
from utils.download_image_video_from_url import download_image_from_url
from utils.download_from_url import download
from utils.clear_temp_folder import clean
from utils.save_deepfake_media_url import save
from utils.is_url_image import is_url_image
from utils.download_youtube_video import download_video
from utils.convert2df import convert_df
from utils.del_module import delete_module
from utils.delete_temp_on_reload import clear_temp_folder_and_reload
clear_temp_folder_and_reload()
#references
from utils.get_references_of_models import get_reference

#examples
from examples.show_sample_deepfakes_from_url import examples
import streamlit_analytics
streamlit_analytics.start_tracking()

import importlib, os
models_list_image = []
models_list_video = []
for model in os.listdir("./models"):
    if model[-6:] == "_image":
        models_list_image.append(model)
    elif model[-6:] == "_video":
        models_list_video.append(model)

# print(models_list_image)
# print(models_list_video)


# this is needed to calculate total inference time
start_time = time.time()
start_time_formatted = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

# favicon being an object of the same kind as the one you should provide st.image() with (ie. a PIL array for example) or a string (url or local file path)
st.write('<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 250px;
        max-width: 250px;
        width: 250px;
    }
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 250px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 250px;
        margin-left: 0px;
    }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] {
        display: none;
    }
    [data-testid="stSidebar"] [aria-expanded="true"] > div:first-child [data-testid="stSidebarContent"] {
        margin-top: 0px;
    }
    [data-testid="stSidebarToggleButton"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Using "with" notation
with st.sidebar:
    add_radio = st.radio(
        " ",
        ("Detector", "Examples", "Learn", "Benchmark", "About")

    )

if add_radio == "Detector":

    model_option= "NaN"
    show_real_fake_button =  False


    #introduction of the app
    st.write("""
    ## DeepSafe : A Fully Open Source DeepFake Detection Platform 
    """)

    #upload button for the input image
    uploaded_file = st.file_uploader("Choose the image/video", type=['jpg', 'png', 'jpeg', 'mp4', ".mov"])
    #print(uploaded_file)
    url = ""
    #if uploaded_file is None:
    url = st.text_input("Or paste the URL below", key="text")

    did_file_download = False

    if uploaded_file is not None:

        did_user_upload_file = True

        #getting the file extension of the uploaded file
        file_name = uploaded_file.name
        extension = file_name.split(".")[-1]

        if extension == "png" or  extension == "PNG":
            uploaded_image = Image.open(uploaded_file)
            save_image(uploaded_file)
            st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

            # cosmetic touch of names
            models_list_image_only_names = []
            for model in models_list_image:
                models_list_image_only_names.append(model[:-6].title())

            model_option = st.multiselect( 'Select a DeepFake Detection Method',
                            models_list_image_only_names)

            model_option = sorted(model_option)

            #st.write('You selected:', model_option)


        elif extension == "jpeg" or extension == "JPEG":
            uploaded_image = Image.open(uploaded_file)
            save_image(uploaded_file)
            st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

            # cosmetic touch of names
            models_list_image_only_names = []
            for model in models_list_image:
                models_list_image_only_names.append(model[:-6].title())

            model_option = st.multiselect( 'Select a DeepFake Detection Method',
                            models_list_image_only_names)

            model_option = sorted(model_option)

            #st.write('You selected:', model_option)

        elif extension == "jpg" or extension == "JPG":
            uploaded_image = Image.open(uploaded_file)
            save_image(uploaded_file)
            st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

            # cosmetic touch of names
            models_list_image_only_names = []
            for model in models_list_image:
                models_list_image_only_names.append(model[:-6].title())

            model_option = st.multiselect( 'Select a DeepFake Detection Method',
                            models_list_image_only_names)

            model_option = sorted(model_option)

            #st.write('You selected:', model_option)

        elif extension == "mp4" or extension == "MP4":
            save_video(uploaded_file)
            video_file = open('temp/delete.mp4', 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes, start_time=0)
            # cosmetic touch of names
            models_list_video_only_names = []
            for model in models_list_video:
                models_list_video_only_names.append(model[:-6].title())
            model_option = st.multiselect( 'Select a DeepFake Detection Method',
                            models_list_video_only_names)

            model_option = sorted(model_option)

            #st.write('You selected:', model_option)
        
        elif extension == "mov" or extension == "MOV":
            save_video(uploaded_file)
            video_file = open('temp/delete.mp4', 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            # cosmetic touch of names
            models_list_video_only_names = []
            for model in models_list_video:
                models_list_video_only_names.append(model[:-6].title())
            model_option = st.multiselect( 'Select a DeepFake Detection Method',
                            models_list_video_only_names)

            model_option = sorted(model_option)

            #st.write('You selected:', model_option)
        else:
            pass
            #st.error('Problem with uploaded file!')
        show_real_fake_button = True

        def clear_text():
            st.session_state["text"] = ""
        
        on_click=clear_text
        

    #st.write('<style>body { margin: 0; font-family: Arial, Helvetica, sans-serif;} .header{padding: 10px 16px; background: #555; color: #f1f1f1; position:fixed;top:0;} .sticky { position: fixed; top: 0; width: 100%;} </style><div class="header" id="myHeader">'+str("yy")+'</div>', unsafe_allow_html=True)


    #converting downloaded images to jpg
    for files in os.listdir("temp"):
        if files.split(".")[-1] == "png":
            img = Image.open('temp/delete.png')
            rgb_img = img.convert('RGB')
            rgb_img.save('temp/delete.jpg')
            os.remove("temp/delete.png")
        elif files.split(".")[-1] == "PNG":
            img = Image.open('temp/delete.PNG')
            rgb_img = img.convert('RGB')
            rgb_img.save('temp/delete.jpg')
            os.remove("temp/delete.PNG")
        elif files.split(".")[-1] == "jpeg":
            img = cv2.imread("temp/delete.jpeg")
            cv2.imwrite("temp/delete"+".jpg", img)
            os.remove("temp/delete.jpeg")
        elif files.split(".")[-1] == "JPEG":
            img = cv2.imread("temp/delete.JPEG")
            cv2.imwrite("temp/delete"+".jpg", img)
            os.remove("temp/delete.JPEG")
        elif files.split(".")[-1] == "jpg" or files.split(".")[-1] == "JPG" or files.split(".")[-1] == "mp4" or files.split(".")[-1] == "MP4" or files.split(".")[-1] == "mov" or files.split(".")[-1] == "MOV":
            pass
        else:
            st.error("Error! We do not support this file format yet!")
            clean()


    probab = 0

    if url != "" and len(os.listdir('temp/')) == 0:
        with st.spinner('Downloading media...'):
            if is_url_image(url) == True:
                download_status = (download_image_from_url(url)) 
                download_status = str(download_status)
                if (download_status != "True"):
                    st.error("Problem downloading media!")
                else:
                    pass
            else:
                download_status = download_video(url)
                if download_status == False:
                    st.error("Problem downloading media, please check the URL and filesize!")
        
        show_real_fake_button = True


    if uploaded_file is None and url!="":

        for file in os.listdir("./temp"):
                #print(file)
            if file[:6] =="delete":
                #print(file[-3:])
                if file[-3:] == "jpg" or file[-3:] == "JPG":
                    try:
                        st.image("./temp/delete.jpg", use_column_width=True)
                    except:
                        st.image("./temp/delete.JPG", use_column_width=True)
                    
                    # cosmetic touch of names
                    models_list_image_only_names = []
                    for model in models_list_image:
                        models_list_image_only_names.append(model[:-6].title())

                    model_option = st.multiselect( 'Select a DeepFake Detection Method',
                                    models_list_image_only_names)

                    model_option = sorted(model_option)

                    #st.write('You selected:', model_option)


                elif file[-3:] == "mp4" or file[-3:] == "MP4":
                    try:
                        video_file = open('./temp/delete.mp4', 'rb')
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                    except:
                        video_file = open('./temp/delete.MP4', 'rb')
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                    
                    # cosmetic touch of names
                    models_list_video_only_names = []
                    for model in models_list_video:
                        models_list_video_only_names.append(model[:-6].title())
                    model_option = st.multiselect( 'Select a DeepFake Detection Method',
                                    models_list_video_only_names)
                    print(model_option)
                    model_option = sorted(model_option)

                    #st.write('You selected:', model_option)
                else:
                    pass
            else:
                os.remove("./temp/" + file)

        show_real_fake_button = True

    if show_real_fake_button == True:
        use_gpu = st.checkbox("Use GPU if model supports (default is CPU)", value=False)
        processing_unit = "GPU" if use_gpu else "CPU"
        if st.button('Real or Fake? 🤔') == True:
            for files in os.listdir("./temp"):
                if files[:6] == "delete":
                    if files[-3:] == "mp4" or files[-3:] == "MP4":
                        st.info("Running DeepFake Video Detectors selected! You can read about them in the About tab.")
                        print("--------------------------------------------------")
                        print(model_option)
                        file_type = "video"
                        print("Filetype: Video")
                        file_size = os.path.getsize("./temp/delete.mp4") / (1024*1024)
                        print("Filesize: ", file_size , "MB")


                        # correcing the names of models - removing the cosmetics
                        model_inference_time_list = []
                        model_inference_probability_list = []
                        #print(model_option)
                        for model in model_option:
                            model = (model + "_video").lower()
                            start_time_model = time.time()
                            model_name = str("models." + model + ".demo")
                            with st.spinner('Running the {} detection model...'.format(model[:-6].title())):
                                # print(model_name)
                                var = importlib.import_module(model_name)
                                # print(want_reload_module)
                                delete_module(model_name)

                            

                            inference_time_model = round((time.time() - start_time_model), 5)

                            result_location = "models/" + model + "/result.txt"
                            f = open(result_location, 'r')
                            variable = f.readline()
                            variable = round(float(variable), 5)
                            #st.success(variable)
                            probab_model = (variable)
                            model_inference_time_list.append(inference_time_model)
                            model_inference_probability_list.append(probab_model)

                        print(model_option, model_inference_probability_list, model_inference_time_list)


                        try:
                            probab = round(float(mean(model_inference_probability_list)), 5)
                        except:
                            pass
                        #print(probab)
                        #st.write("The probability of this video being a deepfake is")
                        #st.write(probab_deepware, probab_cvit, probab_selim, probab_boken, probab)
                        #st.write(inference_time_deepware, inference_time_cvit, inference_time_selim, inference_time_boken)
                        print("--------------------------------------------------")
                        
                        st.subheader("DeepFake Detection Stats")
                    
                        video_array = np.array([model_inference_probability_list, model_inference_time_list])
                        video_array_df = pd.DataFrame(video_array, columns=model_option, index = ["DF Probability", "Inference Time in seconds"])
                        
                        
                        #st.bar_chart(video_array_df)
                        video_array_df = video_array_df.T

                        st.table(video_array_df)
                    
                        csv_1 = convert_df(video_array_df)

                        ste.download_button(
                            label="Download data as CSV ⬇️",
                            data=csv_1,
                            file_name='deepsafe_stats.csv',
                            mime='text/csv',
                        )

                        chart_data = pd.DataFrame(
                            model_inference_probability_list,
                            model_option)

                        #chart_data.rename( columns={'0':'DF_Probability'}, inplace=True )
                        chart_data.rename(columns={ chart_data.columns[0]: "DF Probability" }, inplace = True)


                        #st.dataframe(chart_data)

                        #st.bar_chart(chart_data, width=0, height=0, use_container_width=True)

                        chart_data = pd.DataFrame(
                            model_inference_time_list,
                            model_option)

                        #chart_data.rename( columns={'0':'DF_Probability'}, inplace=True )
                        chart_data.rename(columns={ chart_data.columns[0]: "Inference Time" }, inplace = True)
                        #st.dataframe(chart_data)
                        #st.bar_chart(chart_data, width=0, height=0, use_container_width=True)


                        fig, ax = plt.subplots()
                        ax.set_title('DeepFake Probability VS Detectors')

                        ax.set_xlabel('Detector')
                        ax.set_ylabel('DeepFake Probability')
                        plt.ylim(0, 1)
                        N = len(model_option)
                        bars = ax.bar(model_option, model_inference_probability_list)
                        if N > 1:
                            for i, b in enumerate(bars):
                                b.set_color(plt.cm.jet(1. * i / (N - 1)))

                        st.pyplot(fig)



                        fig, ax = plt.subplots()
                        ax.set_title('Inference Time VS Detectors')

                        ax.set_xlabel('Detector')
                        ax.set_ylabel('Inference time in seconds')
                        bars = ax.bar(model_option, model_inference_time_list)
                        if N > 1:
                            for i, b in enumerate(bars):
                                b.set_color(plt.cm.jet(1. * i / (N - 1)))

                        st.pyplot(fig)


                            
                    elif files[-3:] == "jpg" or files[-3:] == "JPG":
                        st.info("Running DeepFake Image Detectors selected! You can read about them in the About tab.")
                        print("--------------------------------------------------")
                        print(model_option)
                        file_type = "image"
                        print("Filetype: Image")
                        file_size = os.path.getsize("./temp/delete.jpg") / (1024*1024)
                        print("Filesize: ", file_size , "MB")


                        # correcing the names of models - removing the cosmetics
                        model_inference_time_list = []
                        model_inference_probability_list = []
                        #print(model_option)
                        for model in model_option:
                            model = (model + "_image").lower()
                            start_time_model = time.time()
                            model_name = str("models." + model + ".demo")
                            with st.spinner('Running the {} detection model...'.format(model[:-6].title())):
                                # var = importlib.import_module(model_name)
                                #print(model_name)
                                var = importlib.import_module(model_name)
                                #print(want_reload_module)
                                delete_module(model_name)
                                

                            inference_time_model = round((time.time() - start_time_model), 5)

                            result_location = "models/" + model + "/result.txt"
                            f = open(result_location, 'r')
                            variable = f.readline()
                            variable = round(float(variable), 5)
                            #st.success(variable)
                            probab_model = (variable)
                            model_inference_time_list.append(inference_time_model)
                            model_inference_probability_list.append(probab_model)

                        print(model_option, model_inference_probability_list, model_inference_time_list)



                        try:
                            probab = round(float(mean(model_inference_probability_list)), 5)
                        except:
                            pass
                        #print(probab)
                        #st.write("The probability of this video being a deepfake is")
                        #st.write(probab_deepware, probab_cvit, probab_selim, probab_boken, probab)
                        #st.write(inference_time_deepware, inference_time_cvit, inference_time_selim, inference_time_boken)
                        print("--------------------------------------------------")
                        
                        st.subheader("DeepFake Detection Stats")

                        image_array = np.array([model_inference_probability_list, model_inference_time_list])
                        image_array_df = pd.DataFrame(image_array, columns=model_option, index = ["DF Probability", "Inference Time in seconds"])
                        
                        #st.bar_chart(image_array_df)
                        
                        image_array_df = image_array_df.T

                        st.table(image_array_df)

                        csv_1 = convert_df(image_array_df)

                        ste.download_button(
                            label="Download data as CSV ⬇️",
                            data=csv_1,
                            file_name='deepsafe_stats.csv',
                            mime='text/csv',
                        )

                        chart_data = pd.DataFrame(
                            model_inference_probability_list,
                            model_option)

                        #chart_data.rename( columns={'0':'DF_Probability'}, inplace=True )
                        chart_data.rename(columns={ chart_data.columns[0]: "DF Probability" }, inplace = True)


                        #st.dataframe(chart_data)

                        #st.bar_chart(chart_data, width=0, height=0, use_container_width=True)

                        chart_data = pd.DataFrame(
                            model_inference_time_list,
                            model_option)

                        #chart_data.rename( columns={'0':'DF_Probability'}, inplace=True )
                        chart_data.rename(columns={ chart_data.columns[0]: "Inference Time" }, inplace = True)
                        #st.dataframe(chart_data)
                        #st.bar_chart(chart_data, width=0, height=0, use_container_width=True)


                        #st.line_chart(image_array_df)

                        #arr = np.random.normal(1, 1, size=100)

                        fig, ax = plt.subplots()
                        ax.set_title('DeepFake Probability VS Detectors')

                        ax.set_xlabel('Detector')
                        ax.set_ylabel('DeepFake Probability')
                        plt.ylim(0, 1)
                        N = len(model_option)
                        bars = ax.bar(model_option, model_inference_probability_list)
                        if N > 1:
                            for i, b in enumerate(bars):
                                b.set_color(plt.cm.jet(1. * i / (N - 1)))

                        st.pyplot(fig)



                        fig, ax = plt.subplots()
                        ax.set_title('Inference Time VS Detectors')

                        ax.set_xlabel('Detector')
                        ax.set_ylabel('Inference time in seconds')
                        bars = ax.bar(model_option, model_inference_time_list)
                        if N > 1:
                            for i, b in enumerate(bars):
                                b.set_color(plt.cm.jet(1. * i / (N - 1)))

                        st.pyplot(fig)

                            
                    
                        st.balloons()

            if probab>0.7:
                save(url)

            if file_type == "video":
                with open('stats/state.txt', 'r') as f:
                    last_line = f.readlines()[-1]
                    variable = (last_line.split("\t")[0])
                    variable = 1 + int(variable)
                total_inference_time = (time.time() - start_time)
                with open('stats/state.txt', 'a') as f:
                    f.write(str(variable))
                    f.write("\t")
                    f.write(str(start_time_formatted))
                    f.write("\t")
                    f.write(str(probab))
                    f.write("\t")
                    f.write(str(file_type))
                    f.write("\t")
                    f.write(str(file_size))
                    f.write("\t")
                    f.write(str(total_inference_time))
                    #f.close()

                with open('stats/state.txt','a') as f:
                    for i in model_option:
                        f.write('\t%s'%i)
                        #f.close()

                with open('stats/state.txt','a') as f:
                    for i in model_inference_probability_list:
                        f.write('\t%s'%i)
                        #f.close()

                with open('stats/state.txt','a') as f:
                    for i in model_inference_time_list:
                        f.write('\t%s'%i)
                    f.write("\n")
                    f.close()
                    

                # open the file in the write mode
                with open('stats/state.csv', 'a') as f:
                    # create the csv writer
                    writer = csv.writer(f)
                    row = [str(variable), str(start_time_formatted), str(probab), str(file_type), str(file_size),str(total_inference_time)]

                    for i in model_option:
                        row.append(str(i))
                    for i in model_inference_probability_list:
                        row.append(str(i))
                    for i in model_inference_time_list:
                        row.append(str(i))

                    # write a row to the csv file
                    writer.writerow(row)


            if file_type == "image":
                with open('stats/state.txt', 'r') as f:
                    last_line = f.readlines()[-1]
                    variable = (last_line.split("\t")[0])
                    variable = 1 + int(variable)
                total_inference_time = (time.time() - start_time)
                with open('stats/state.txt', 'a') as f:
                    f.write(str(variable))
                    f.write("\t")
                    f.write(str(start_time_formatted))
                    f.write("\t")
                    f.write(str(probab))
                    f.write("\t")
                    f.write(str(file_type))
                    f.write("\t")
                    f.write(str(file_size))
                    f.write("\t")
                    f.write(str(total_inference_time))
                    #f.close()

                with open('stats/state.txt','a') as f:
                    for i in model_option:
                        f.write('\t%s'%i)
                        #f.close()

                with open('stats/state.txt','a') as f:
                    for i in model_inference_probability_list:
                        f.write('\t%s'%i)
                        #f.close()

                with open('stats/state.txt','a') as f:
                    for i in model_inference_time_list:
                        f.write('\t%s'%i)
                    f.write("\n")
                    f.close()

                #import csv

                # open the file in the write mode
                with open('stats/state.csv', 'a') as f:
                    # create the csv writer
                    writer = csv.writer(f)
                    row = [str(variable), str(start_time_formatted), str(probab), str(file_type), str(file_size),str(total_inference_time)]

                    for i in model_option:
                        row.append(str(i))
                    for i in model_inference_probability_list:
                        row.append(str(i))
                    for i in model_inference_time_list:
                        row.append(str(i))

                    # write a row to the csv file
                    writer.writerow(row)


                #checking if user uploaded any file


            clean()





    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    



    with st.expander("What's DeepFake?"):
        st.write("""
            Deepfakes are altered photos and videos that use artificial intelligence to appear like the real thing. Watch this video to see it in action!
        """)
        #st.image("https://static.streamlit.io/examples/dice.jpg")
        st.video('https://youtu.be/cQ54GDm1eL0') 
        st.write("""[Read more](https://ineqe.com/2021/10/13/a-beginners-guide-to-deepfakes/)""")
        st.snow()



    if st.button("Something's wrong? Feedback"):
        js = "window.open('https://tally.so/r/wvpgAm/')"  # New tab or window
        #js = "window.location.href = 'https://forms.gle/6xiHjzmgaxG514xr5/'"  # Current tab
        html = '<img src onerror="{}">'.format(js)
        div = Div(text=html)
        st.bokeh_chart(div)

    #st.info("No image/video/audio or URL is stored/saved/further used!")

# add the feature to automatically update the references
#if add_radio == "Reference":

def read_markdown_file(markdown_file):
    with open(markdown_file, 'r', encoding='utf-8') as file:
        return file.read()

if add_radio == "Examples":
    st.header("Some of the SFW DeepFakes on the internet!")
    examples()

def show_pdf(file_path):
    with open(file_path,"rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)



import base64
if add_radio == "Learn":
    file = "./assets/Which face is real.pdf"
    show_pdf(file)

# Assuming utility functions and necessary imports are already defined

if add_radio == "Benchmark":
    st.write("## Benchmark your dataset")
    
    dataset_type = st.radio("Select the type of dataset", ["Image", "Video"])
    
    if dataset_type:
        dataset_folder = os.path.join('datasets', dataset_type.lower())
        available_datasets = [folder for folder in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, folder))]
        
        selected_dataset = st.selectbox("Select a dataset", available_datasets)
        
        if selected_dataset:
            st.write("### Select DeepFake Detection Models")
            
            # Read dataset config file
            config_file_path = os.path.join(dataset_folder, selected_dataset, '.config')
            dataset_info = ""
            if os.path.exists(config_file_path):
                with open(config_file_path, 'r') as config_file:
                    dataset_info = config_file.read()
            
            # Display dataset information as a collapsible element
            with st.expander("Read about this dataset", expanded=False):
                st.write(dataset_info)
            
            # Filter models based on dataset type
            if dataset_type == "Image":
                models_list = models_list_image
            elif dataset_type == "Video":
                models_list = models_list_video
            
            models_list_only_names = [model[:-6].title() for model in models_list]
            
            selected_models = st.multiselect('Select the models for benchmarking', models_list_only_names)
            
            if selected_models:
                selected_models = sorted(selected_models)
                
                if st.button('Benchmark Dataset'):
                    st.write("### Benchmarking in progress...")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    benchmark_results = []
                    
                    dataset_path = os.path.join(dataset_folder, selected_dataset)
                    real_files = [os.path.join(dataset_path, 'real', f) for f in os.listdir(os.path.join(dataset_path, 'real'))]
                    fake_files = [os.path.join(dataset_path, 'fake', f) for f in os.listdir(os.path.join(dataset_path, 'fake'))]
                    all_files = real_files + fake_files
                    
                    total_files = len(all_files)
                    
                    # Create leaderboard folder with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    leaderboard_folder = os.path.join('leaderboard', timestamp)
                    os.makedirs(leaderboard_folder, exist_ok=True)
                    
                    csv_file_path = os.path.join(leaderboard_folder, 'benchmark_results.csv')
                    metadata_file_path = os.path.join(leaderboard_folder, 'metadata.json')
                    
                    for index, file_path in enumerate(tqdm(all_files, desc="Benchmarking", unit="file")):
                        file_name = os.path.basename(file_path)
                        extension = file_name.split(".")[-1]
                        
                        if extension in ["png", "jpeg", "jpg"]:
                            uploaded_image = Image.open(file_path)
                            uploaded_image.save('temp/delete.jpg')
                            input_path = 'temp/delete.jpg'
                        elif extension in ["mp4", "mov"]:
                            with open(file_path, 'rb') as f:
                                with open('temp/delete.mp4', 'wb') as temp_f:
                                    temp_f.write(f.read())
                            input_path = 'temp/delete.mp4'
                        else:
                            st.error(f"Unsupported file format: {extension}")
                            continue
                        
                        # Determine if the file is real or fake based on folder structure
                        label = 1 if "fake" in file_path.lower() else 0
                        
                        file_results = {
                            "file_name": file_name,
                            "file_path": file_path,
                            "label": label
                        }
                        
                        for model in selected_models:
                            model_name = (model + "_" + dataset_type.lower()).lower()
                            start_time_model = time.time()
                            model_module_name = str("models." + model_name + ".demo")
                            
                            status_text.text(f'Processing {file_name} with model {model}...')
                            
                            with st.spinner(f'Running the {model} detection model...'):
                                var = importlib.import_module(model_module_name)
                                delete_module(model_module_name)
                            
                            inference_time_model = round((time.time() - start_time_model), 5)
                            result_location = f"models/{model_name}/result.txt"
                            
                            with open(result_location, 'r') as f:
                                variable = f.readline()
                                probab_model = round(float(variable), 5)
                            
                            file_results[model] = {
                                "probability": probab_model,
                                "inference_time": inference_time_model
                            }
                        
                        benchmark_results.append(file_results)
                        progress_bar.progress((index + 1) / total_files)
                        
                        # Save intermediate results
                        results_df = pd.DataFrame(benchmark_results)
                        csv_results = results_df.to_csv(index=False)
                        with open(csv_file_path, 'w') as f:
                            f.write(csv_results)
                        
                        metadata = {
                            "timestamp": timestamp,
                            "dataset_type": dataset_type,
                            "models": selected_models,
                            "total_files": total_files,
                            "benchmark_results": benchmark_results
                        }
                        with open(metadata_file_path, 'w') as f:
                            json.dump(metadata, f, indent=4)
                    
                    st.write("### Benchmarking Completed")
                    status_text.text("")
                    
                    ste.download_button(
                        label="Download Benchmark Results as CSV ⬇️",
                        data=csv_results,
                        file_name='benchmark_results.csv',
                        mime='text/csv'
                    )
                    
                    # Show stats and plots
                    st.write("## Model Performance Comparison")
                    model_performance = {}
                    
                    for model in selected_models:
                        model_name = (model + "_" + dataset_type.lower()).lower()
                        probabilities = [result[model]['probability'] for result in benchmark_results]
                        inference_times = [result[model]['inference_time'] for result in benchmark_results]
                        labels = [result['label'] for result in benchmark_results]
                        predictions = [1 if prob > 0.5 else 0 for prob in probabilities]
                        
                        accuracy = accuracy_score(labels, predictions)
                        precision = precision_score(labels, predictions, zero_division=0)
                        recall = recall_score(labels, predictions, zero_division=0)
                        
                        model_performance[model] = {
                            "probabilities": probabilities,
                            "inference_times": inference_times,
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall
                        }
                    
                    # Apply styles for better looking plots
                    plt.style.use('ggplot')
                    colors = plt.get_cmap('tab10').colors
                    
                    # Plot accuracy, precision, recall for each model
                    metrics = ['accuracy', 'precision', 'recall']
                    for metric in metrics:
                        fig, ax = plt.subplots()
                        values = [model_performance[model][metric] for model in selected_models]
                        ax.bar(selected_models, values, color=colors)
                        ax.set_title(f'Model {metric.capitalize()} Comparison')
                        ax.set_xlabel('Model')
                        ax.set_ylabel(metric.capitalize())
                        ax.set_ylim(0, 1)
                        for i in range(len(values)):
                            ax.text(i, values[i] + 0.01, f'{values[i]:.2f}', ha='center', va='bottom')
                        st.pyplot(fig)
                    
                    # Plot mean inference time for each model
                    fig, ax = plt.subplots()
                    mean_inference_times = [mean(model_performance[model]['inference_times']) for model in selected_models]
                    ax.bar(selected_models, mean_inference_times, color=colors)
                    ax.set_title('Mean Inference Time Comparison')
                    ax.set_xlabel('Model')
                    ax.set_ylabel('Mean Inference Time (seconds)')
                    for i in range(len(mean_inference_times)):
                        ax.text(i, mean_inference_times[i] + 0.1, f'{mean_inference_times[i]:.2f}', ha='center', va='bottom')
                    st.pyplot(fig)

                    # Plot probabilities
                    for model in selected_models:
                        fig, ax = plt.subplots()
                        probabilities = [result[model]['probability'] for result in benchmark_results]
                        ax.plot(range(len(probabilities)), probabilities, label=model)
                        ax.set_title('DeepFake Probability')
                        ax.set_xlabel('File Index')
                        ax.set_ylabel('Probability')
                        ax.legend()
                        st.pyplot(fig)

                    # Plot inference times
                    for model in selected_models:
                        fig, ax = plt.subplots()
                        inference_times = [result[model]['inference_time'] for result in benchmark_results]
                        ax.plot(range(len(inference_times)), inference_times, label=model)
                        ax.set_title('Inference Time')
                        ax.set_xlabel('File Index')
                        ax.set_ylabel('Time (seconds)')
                        ax.legend()
                        st.pyplot(fig)
                    
                    clean()
            else:
                st.info("Please select at least one model for benchmarking.")
    else:
        st.info("Please select a dataset type to proceed with benchmarking.")


if add_radio == "About":


    st.subheader("Detectors we used for this demo, credits to the original authors. These folks are amazing! 😎")
    with st.expander(("Guide on how to add your detector to this platform!")):
        markdown_file_path = "Add_Custom_Model_to_DeepSafe.md"

        # Read the contents of the Markdown file
        markdown_content = read_markdown_file(markdown_file_path)

        # Display the Markdown content in Streamlit
        st.markdown(markdown_content)


    reference_list, len_list = get_reference()
    
    reference = []

    # for i in range (int(len_list/3)):
    #     #st.write((i+1), ". ", reference_list[3*i], (reference_list[3*i+1]), (reference_list[3*i+2]))
    #     #reference.append((i+1))
    #     reference.append(reference_list[3*i])
    #     reference.append(reference_list[3*i+1])
    #     reference.append(reference_list[3*i+2])

    reference = np.resize(reference_list, len_list).reshape((int(len_list/3)), 3)
    reference = pd.DataFrame(reference, columns = ['Detector','Project Website','License'])
    reference.index = np.arange(1, len(reference) + 1)

    st.dataframe(reference)

        


    # st.write()
    # st.write("2. FaceForensics++ [Github](https://github.com/ondyari/FaceForensics)")
    # st.write("3. CNNDetection [Github](https://github.com/peterwang512/CNNDetection)")
    # st.write("4. PhotoshopFAL [Github](https://github.com/peterwang512/FALdetector)")
    # st.write("5. CViT [Github](https://github.com/erprogs/CViT)")
    # st.write("6. DeepWare [Github](https://github.com/deepware/deepfake-scanner)")
    # st.write("7. Boken [Github](https://github.com/beibuwandeluori/DeeperForensicsChallengeSolution)")


    st.success("This webapp is based on [Streamlit](https://streamlit.io/)!")

    # st.write("\n")
    # st.write("\n")
    # st.write("\n")

    # with st.expander("Can blockchain be used to combat Deepfake?"):
    #     st.write("""
    #     ELI5-- Probably, authencity of a video or a 'fingerprint' can be saved on a Blockchain.
        
    #     """)
    #     st.video('https://youtu.be/mN3cvr9aClA')
    #     #st.image("https://static.streamlit.io/examples/dice.jpg")
    #     #st.video('https://youtu.be/cQ54GDm1eL0') 
    #     st.write("""[Read more](https://www.wired.com/story/the-blockchain-solution-to-our-deepfake-problems/)""")
    #     #st.snow()







    #st.header("Contribute")
    st.title("Contribute")
    #st.subheader("Contribute")
    st.info(
        "This an open source project and you are welcome to **contribute** your "
        "comments, questions, resources and apps as "
        "[issues](https://github.com/siddharthksah/DeepSafe/issues) of or "
        "[pull requests](https://github.com/siddharthksah/DeepSafe/pulls) "
        "to the [source code](https://github.com/siddharthksah/DeepSafe). "
    )
    

    # st.info(
    #     """
    #     Currently, this app is maintained by [Siddharth](https://siddharthksah.github.io/).
    #     """)
    # st.write("\n")
    # st.write("\n")
    # st.write("\n")
    # st.write("\n")
    # st.write("\n")
    # st.write("\n")
    
    st.write("Made with :heart: by [Siddharth](https://siddharthksah.github.io/)")


    # with st.expander("TechStack"):
    #     st.markdown("![Streamlit](https://img.shields.io/badge/streamlit-%E2%9C%94-green)")
    #     st.markdown("![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)")
    #     st.markdown("![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)")
    #     st.markdown("![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)")
    #     st.markdown("![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)")
    #     st.markdown("![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)")
    #     st.markdown("![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)")
    #     st.markdown("![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)")
    #     st.markdown("![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)")
    #     st.markdown("![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)")
    #     st.markdown("![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)")
    #     st.markdown("![Dash](https://img.shields.io/badge/dash-008DE4?style=for-the-badge&logo=dash&logoColor=white)")
        

    
    if st.button("Something's wrong? Feedback"):
        js = "window.open('https://tally.so/r/wvpgAm/')"  # New tab or window
        #js = "window.location.href = 'https://forms.gle/6xiHjzmgaxG514xr5/'"  # Current tab
        html = '<img src onerror="{}">'.format(js)
        div = Div(text=html)
        st.bokeh_chart(div)

streamlit_analytics.stop_tracking()