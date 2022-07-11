import requests

#url ="https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg?crop=1.00xw:0.669xh;0,0.190xh&resize=1200:*"
#url = "https://ggsc.s3.amazonaws.com/images/uploads/The_Science-Backed_Benefits_of_Being_a_Dog_Owner.jpg"
def is_url_image(image_url):
   image_formats = ("image/png", "image/jpeg", "image/jpg")
   try:
      r = requests.head(image_url)
      if r.headers["content-type"] in image_formats:
         return True
      return False
   except:
      pass
#print(is_url_image(url))