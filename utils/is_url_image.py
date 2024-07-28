import requests

# url ="https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg?crop=1.00xw:0.669xh;0,0.190xh&resize=1200:*"
#url = "https://ggsc.s3.amazonaws.com/images/uploads/The_Science-Backed_Benefits_of_Being_a_Dog_Owner.jpg"

import requests

def is_url_image(image_url):
    image_formats = ("image/png", "image/jpeg", "image/jpg", "image/webp")
    image_extensions = (".png", ".jpeg", ".jpg", ".webp")
    try:
        r = requests.head(image_url, timeout=5)
        content_type = r.headers.get("content-type")
      #   print(f"URL: {image_url}")
      #   print(f"Content-Type: {content_type}")
        if content_type in image_formats or image_url.lower().endswith(image_extensions):
            return True
        return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

# url = "https://mockey.ai/wp-content/uploads/sites/15/2023/11/deepfake-image-generator-examples.webp"
# print(is_url_image(url))