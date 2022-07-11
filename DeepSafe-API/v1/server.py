from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2, os
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

from predictor import predictor_CNN

# Initialize the Flask application
app = Flask(__name__)

# route http posts to this method
@app.route('/api_v1/', methods=['POST'])

def test():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #print(img)
    img = Image.fromarray(img)
    b, g, r = img.split()
    img = Image.merge("RGB", (r, g, b))
    if not os.path.exists('tempDir'):
        os.makedirs('tempDir')
    img.save('./tempDir/image.jpg', 'JPEG')

    #run prediction
    probab = predictor_CNN()

    response = {'Probability of DeepFake': probab}

    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

# start flask app
app.run(host="0.0.0.0", port=5001)