# @author Siddharth
# @website www.siddharthsah.com

# importing the necessary packages
from audioop import add
from datetime import datetime
from distutils.util import rfc822_escape
import time, random
import matplotlib.pyplot as plt
rgb = [[random.random(), random.random(), random.random()]]
import csv
import numpy as np
import streamlit as st
from PIL import Image
from skimage import transform
import os, cv2
from PIL import Image
import pandas as pd
from statistics import mean
from bokeh.models.widgets import Div


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

#references
from utils.get_references_of_models import get_reference

#examples
from examples.show_sample_deepfakes_from_url import examples



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

# favicon and page configs
favicon = './assets/icon.png'
st.set_page_config(page_title='DeepSafe', page_icon = favicon, initial_sidebar_state = 'expanded')
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
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 140px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 140px;
        margin-left: -140px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Using "with" notation
with st.sidebar:
    add_radio = st.radio(
        " ",
        ("Detector", "Examples", "About", "Learn")

    )

if add_radio == "Detector":


    model_option= "NaN"
    show_real_fake_button =  False


    #introduction of the app
    st.write("""
    ## DeepSafe - A free DeepFake Detector
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
        if st.button('Real or Fake? 🐶') == True:
            
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

                            

                            inference_time_model = round((time.time() - start_time_model), 2)

                            result_location = "models/" + model + "/result.txt"
                            f = open(result_location, 'r')
                            variable = f.readline()
                            variable = round(float(variable),2)
                            #st.success(variable)
                            probab_model = (variable)
                            model_inference_time_list.append(inference_time_model)
                            model_inference_probability_list.append(probab_model)

                        print(model_option, model_inference_probability_list, model_inference_time_list)


                        try:
                            probab = round(float(mean(model_inference_probability_list)),2)
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

                        st.download_button(
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
                                

                            inference_time_model = round((time.time() - start_time_model), 2)

                            result_location = "models/" + model + "/result.txt"
                            f = open(result_location, 'r')
                            variable = f.readline()
                            variable = round(float(variable),2)
                            #st.success(variable)
                            probab_model = (variable)
                            model_inference_time_list.append(inference_time_model)
                            model_inference_probability_list.append(probab_model)

                        print(model_option, model_inference_probability_list, model_inference_time_list)



                        try:
                            probab = round(float(mean(model_inference_probability_list)),2)
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

                        st.download_button(
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
                with open('state.txt', 'r') as f:
                    last_line = f.readlines()[-1]
                    variable = (last_line.split("\t")[0])
                    variable = 1 + int(variable)
                total_inference_time = (time.time() - start_time)
                with open('state.txt', 'a') as f:
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

                with open('state.txt','a') as f:
                    for i in model_option:
                        f.write('\t%s'%i)
                        #f.close()

                with open('state.txt','a') as f:
                    for i in model_inference_probability_list:
                        f.write('\t%s'%i)
                        #f.close()

                with open('state.txt','a') as f:
                    for i in model_inference_time_list:
                        f.write('\t%s'%i)
                    f.write("\n")
                    f.close()
                    

                # open the file in the write mode
                with open('state.csv', 'a') as f:
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
                with open('state.txt', 'r') as f:
                    last_line = f.readlines()[-1]
                    variable = (last_line.split("\t")[0])
                    variable = 1 + int(variable)
                total_inference_time = (time.time() - start_time)
                with open('state.txt', 'a') as f:
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

                with open('state.txt','a') as f:
                    for i in model_option:
                        f.write('\t%s'%i)
                        #f.close()

                with open('state.txt','a') as f:
                    for i in model_inference_probability_list:
                        f.write('\t%s'%i)
                        #f.close()

                with open('state.txt','a') as f:
                    for i in model_inference_time_list:
                        f.write('\t%s'%i)
                    f.write("\n")
                    f.close()

                #import csv

                # open the file in the write mode
                with open('state.csv', 'a') as f:
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
    

if add_radio == "About":


    st.subheader("Detectors we used for this demo, credits to the original authors. These folks are amazing! 😎")
    with st.expander(("Guide on how to add your detector to this platform!")):
        st.info("Coming soon...")
        code = '''def hello():
            print("Hello, DeepSafe!")'''
        st.code(code, language='python')


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

    with st.expander("Can blockchain be used to combat Deepfake?"):
        st.write("""
        ELI5-- Probably, authencity of a video or a 'fingerprint' can be saved on a Blockchain.
        
        """)
        st.video('https://youtu.be/mN3cvr9aClA')
        #st.image("https://static.streamlit.io/examples/dice.jpg")
        #st.video('https://youtu.be/cQ54GDm1eL0') 
        st.write("""[Read more](https://www.wired.com/story/the-blockchain-solution-to-our-deepfake-problems/)""")
        #st.snow()







    #st.header("Contribute")
    st.title("Contribute")
    #st.subheader("Contribute")
    st.info(
        "This an open source project and you are very welcome to **contribute** your "
        "comments, questions, resources and apps as "
        "[issues](https://github.com/siddharthksah/DeepSafe/issues) of or "
        "[pull requests](https://github.com/siddharthksah/DeepSafe/pulls) "
        "to the [source code](https://github.com/siddharthksah/DeepSafe). "
    )
    

    # st.info(
    #     """
    #     Currently, this app is maintained by [Siddharth](https://www.siddharthsah.com/).
    #     """)
    # st.write("\n")
    # st.write("\n")
    # st.write("\n")
    # st.write("\n")
    # st.write("\n")
    # st.write("\n")
    
    st.write("Made with :heart: by [Siddharth](http://www.siddharthsah.com/)")


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


if add_radio == "Examples":
    st.header("Some of the SFW DeepFakes on the internet!")
    examples()


if add_radio == "Detector":
    file = "assets/Websites supported by DeepSafe.pdf"
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)