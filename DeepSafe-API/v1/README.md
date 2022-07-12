# DeepSafe API
A RESTful Flask API. DeepSafe-API combines the powerful features of the DeepSafe-WebApp into an API. DeepSafe WebApp is an open-source platform that integrates state-of-the-art DeepFake detection methods and provide a convenient interface for the users to compare their custom detectors against SOTA along with improving the literacy of DeepFakes among common folks.
### Overview

The code consists of both client and server side of the code. The image is saved locally before doing the inference, but you can delete the save location or even use the image on the fly.

The output is a json which consists the deepfake probability, were the probability closer to 1 means the model thinks it is a deepfake.

### Installation
1. Clone the repository:
```bash
git clone https://github.com/siddharthksah/DeepSafe
cd DeepSafe/DeepSafe-API/v1
```

Python version
---
> Main supported version : <strong>3.8</strong> <br>
> Other supported versions : <strong>3.7</strong> & <strong>3.9</strong>
---
2. Creating conda environment

```bash
conda create -n deepsafe-api python==3.8 -y
conda activate deepsafe-api
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---
#### Code breakdown
Import and initialise your application:

```python
from flask_api import FlaskAPI

app = FlaskAPI(__name__)
```

## Responses

Return any valid response object as normal, or return a `list` or `dict`.

```python
@app.route('/example/')
def example():
    return {'hello': 'world'}
```

A renderer for the response data will be selected using content negotiation based on the client 'Accept' header. If you're making the API request from a regular client, this will default to a JSON response. If you're viewing the API in a browser, it'll default to the browsable API HTML.

## Requests

Access the parsed request data using `request.data`.  This will handle JSON or form data by default.

```python
@app.route('/example/')
def example():
    return {'request data': request.data}
```

## Example

The following example demonstrates a simple API for creating, listing, updating and deleting notes.

```python
from flask import request, url_for
from flask_api import FlaskAPI, status, exceptions

app = FlaskAPI(__name__)


notes = {
    0: 'do the shopping',
    1: 'build the codez',
    2: 'paint the door',
}

def note_repr(key):
    return {
        'url': request.host_url.rstrip('/') + url_for('notes_detail', key=key),
        'text': notes[key]
    }


@app.route("/", methods=['GET', 'POST'])
def notes_list():
    """
    List or create notes.
    """
    if request.method == 'POST':
        note = str(request.data.get('text', ''))
        idx = max(notes.keys()) + 1
        notes[idx] = note
        return note_repr(idx), status.HTTP_201_CREATED

    # request.method == 'GET'
    return [note_repr(idx) for idx in sorted(notes.keys())]


@app.route("/<int:key>/", methods=['GET', 'PUT', 'DELETE'])
def notes_detail(key):
    """
    Retrieve, update or delete note instances.
    """
    if request.method == 'PUT':
        note = str(request.data.get('text', ''))
        notes[key] = note
        return note_repr(key)

    elif request.method == 'DELETE':
        notes.pop(key, None)
        return '', status.HTTP_204_NO_CONTENT

    # request.method == 'GET'
    if key not in notes:
        raise exceptions.NotFound()
    return note_repr(key)


if __name__ == "__main__":
    app.run(debug=True)
```

Now run the webapp:

```shell
$ python ./example.py
 * Running on http://127.0.0.1:5000/
 * Restarting with reloader
```

You can now open a new tab and interact with the API from the command line:

```shell
$ curl -X GET http://127.0.0.1:5000/
[{"url": "http://127.0.0.1:5000/0/", "text": "do the shopping"},
 {"url": "http://127.0.0.1:5000/1/", "text": "build the codez"},
 {"url": "http://127.0.0.1:5000/2/", "text": "paint the door"}]

$ curl -X GET http://127.0.0.1:5000/1/
{"url": "http://127.0.0.1:5000/1/", "text": "build the codez"}

$ curl -X PUT http://127.0.0.1:5000/1/ -d text="flask api is teh awesomez"
{"url": "http://127.0.0.1:5000/1/", "text": "flask api is teh awesomez"}
```

You can also work on the API directly in your browser, by opening <http://127.0.0.1:5000/>.  You can then navigate between notes, and make `GET`, `PUT`, `POST` and `DELETE` API requests.

---
---
## DeepSafe API


### Server
```python
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
```
### Client
```python
import requests
import json
import cv2
import os, shutil

addr = 'http://localhost:5001'
test_url = addr + '/api_v1/'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}
img = cv2.imread('tempDir/image.jpg')

# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)

# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tobytes(), headers=headers)

print(json.loads(response.text))
```

---
---
