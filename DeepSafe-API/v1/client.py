import requests
import cv2
import json

# Server address and endpoint configuration
addr = 'http://localhost:5001'
api_endpoint = '/api_v1/'
test_url = addr + api_endpoint

# Read and encode the image
image_path = 'tempDir/image.jpg'
image = cv2.imread(image_path)
_, encoded_image = cv2.imencode('.jpg', image)

# HTTP request headers
headers = {
    'content-type': 'image/jpeg'
}

# Post the image to the server and get the response
response = requests.post(test_url, data=encoded_image.tobytes(), headers=headers)
response_data = json.loads(response.text)

print(response_data)
