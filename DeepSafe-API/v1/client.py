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

