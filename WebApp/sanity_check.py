import importlib
import os
import time
import shutil
from utils.del_module import delete_module
from utils.clear_temp_folder import clean

print("-------------------------------------------------------------------------------------------")
print("Initializing...")
time.sleep(1)

# Loading detectors and categorizing them based on file type
models_list_image, models_list_video = [], []
for model in os.listdir("./models"):
    if model.endswith("_image"):
        models_list_image.append(model.replace("_image", ""))
    elif model.endswith("_video"):
        models_list_video.append(model.replace("_video", ""))

print("Image based DeepFake Detectors:", models_list_image)
print("Video based DeepFake Detectors:", models_list_video)
print("-------------------------------------------------------------------------------------------")
time.sleep(2)

# Running inferences on test files
for file in os.listdir("./assets/Sanity_Check/"):
    file_type = "video" if file.lower().endswith(("mp4")) else "image"
    print(f"Running inference on a test {file_type} file against mentioned detectors...")
    test_file_path = f'./assets/Sanity_Check/delete.{file_type}'
    shutil.copy2(test_file_path, './temp/')
    time.sleep(1)

    file_size = os.path.getsize(test_file_path) / (1024 * 1024)
    print(f"Filesize: {file_size:.2f} MB")

    models_list = models_list_video if file_type == "video" else models_list_image
    model_inference_time_list, model_inference_probability_list = [], []

    for model in models_list:
        model_full_name = f"models.{model}_{file_type}.demo".lower()
        start_time_model = time.time()

        # Running the model inference
        var = importlib.import_module(model_full_name)
        delete_module(model_full_name)
        inference_time = round(time.time() - start_time_model, 2)

        # Reading inference results
        with open(f"models/{model_full_name}/result.txt", 'r') as file:
            probability = round(float(file.readline()), 2)

        model_inference_time_list.append(inference_time)
        model_inference_probability_list.append(probability)
        print(f"{model_full_name} | {inference_time} | {probability} | Status : OK")
        print("-------------------------------------------------------------------------------------------")

# Cleaning up temporary files
clean()
