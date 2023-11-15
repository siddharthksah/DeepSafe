from flask import Flask, request, Response
import numpy as np
import cv2
import os
from PIL import Image
import jsonpickle
from predictor import predictor_CNN

# Initialize the Flask application
app = Flask(__name__)

@app.route('/api_v1/', methods=['POST'])
def test():
    # Convert string of image data to uint8
    nparr = np.frombuffer(request.data, np.uint8)
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Convert to PIL Image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    
    # Create tempDir if it doesn't exist
    temp_dir = './tempDir'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    image_path = os.path.join(temp_dir, 'image.jpg')
    img.save(image_path, 'JPEG')

    # Run prediction
    probab = predictor_CNN()

    # Prepare and send response
    response = {'Probability of DeepFake': probab}
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")

# Start flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)
