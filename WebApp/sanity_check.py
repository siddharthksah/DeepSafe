import importlib, os, time, shutil
from utils.del_module import delete_module
from utils.clear_temp_folder import clean


print("-------------------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------------------")

print("Initializing...")
time.sleep(1)
print("Loading detectors...")
time.sleep(1)

models_list_image = []
models_list_video = []

for model in os.listdir("./models"):
    if model[-6:] == "_image":
        models_list_image.append(model[:-6])
    elif model[-6:] == "_video":
        models_list_video.append(model[:-6])

print("Image based DeepFake Detectors", models_list_image)
time.sleep(1)
print("Video based DeepFake Detectors", models_list_video)

# print("-------------------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------------------")
time.sleep(2)





for files in os.listdir("./assets/Sanity_Check/"):
    if files[-3:] == "mp4" or files[-3:] == "MP4":
        #print(files)
        shutil.copy2('./assets/Sanity_Check/delete.mp4', './temp/') # target filename is /dst/dir/file.ext
        print(models_list_video)
        time.sleep(1)
        print("Running inference on a test video file against mentioned detectors...")
        time.sleep(1)
        file_type = "video"
        print("Filetype: Video")
        time.sleep(1)
        file_size = os.path.getsize("./assets/Sanity_Check/delete.mp4") / (1024*1024)
        print("Filesize: ", file_size , "MB")


        # correcing the names of models - removing the cosmetics
        model_inference_time_list = []
        model_inference_probability_list = []
        #print(model_option)
        for model in models_list_video:
            model = (model + "_video").lower()
            start_time_model = time.time()
            model_name = str("models." + model + ".demo")
            #print(model_name)

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
            
            print(model_name, "|", inference_time_model, "|", probab_model, "|", "Status : OK")
            print("-------------------------------------------------------------------------------------------")



                
    elif files[-3:] == "jpg" or files[-3:] == "JPG":
        #print(files)
        shutil.copy2('./assets/Sanity_Check/delete.jpg', './temp/') # target filename is /dst/dir/file.ext
        print(models_list_image)
        time.sleep(1)
        print("Running inference on a test video file against mentioned detectors...")
        time.sleep(1)
        file_type = "image"
        print("Filetype: Image")
        time.sleep(1)
        file_size = os.path.getsize("./assets/Sanity_Check/delete.jpg") / (1024*1024)
        print("Filesize: ", file_size , "MB")


        # correcing the names of models - removing the cosmetics
        model_inference_time_list = []
        model_inference_probability_list = []
        #print(model_option)
        for model in models_list_image:
            model = (model + "_image").lower()
            start_time_model = time.time()
            model_name = str("models." + model + ".demo")
            #print(model_name)

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
            
            print(model_name, "|", inference_time_model, "|", probab_model, "|", "Status : OK")
            print("-------------------------------------------------------------------------------------------")

clean()