import os
import shutil

def clean():
    temp_dir = './temp'
    
    # Remove the temp directory if it exists
    try:
        shutil.rmtree(temp_dir)
    except FileNotFoundError:
        pass
    
    # Create the temp directory if it does not exist
    os.makedirs(temp_dir, exist_ok=True)

    # Remove result.txt files in the models directory
    models_dir = './models'
    for model in os.listdir(models_dir):
        result_file = os.path.join(models_dir, f"{model}result.txt")
        if os.path.exists(result_file):
            os.remove(result_file)

clean()
