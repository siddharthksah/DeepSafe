import os
import shutil
def clean():
    try:
        shutil.rmtree('./temp')
    except:
        pass
    if not os.path.exists('temp'):
        os.makedirs('temp')

    for model in os.listdir("./models"):
        if os.path.exists("models/" + model + "result.txt"):
            os.remove("models/" + model + "result.txt") # one file at a time