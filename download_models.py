import gdown

#this takes a while cause the folder is quite big about 3.4G

# a folder
url = "https://drive.google.com/drive/folders/1Gan21zLaPD0wHbNF3P3a7BzgKE91BOpq?usp=sharing"

gdown.download_folder(url, quiet=True, use_cookies=False)
