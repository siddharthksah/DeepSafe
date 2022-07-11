from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from skimage import transform
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow




def predictor_DFDC(path):
    uploaded_image = Image.open(path)

    # preprocessing the image to use in the trained model
    np_image = uploaded_image
    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np.expand_dims(np_image, axis=0)

    # loading pre trained model called model.h5, you can find the code to export this on my GitHub https://github.com/siddharthksah
    #@st.cache
    with tensorflow.device('/cpu:0'):
        model = tensorflow.keras.models.load_model("models/dfdc_image/model.h5")
        probab = model.predict(np_image)[0][0]
    
    probab = round(probab, 4)
    with open("models/dfdc_image/result.txt", "w") as text_file:
        text_file.write(str(1-probab))

    #print(1 - probab)
    return(1 - probab)

path = "temp/delete.jpg"
predictor_DFDC(path)