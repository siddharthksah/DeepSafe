import os, io

def save_video(video_file):
    if video_file is not None:
        g = io.BytesIO(video_file.read())  ## BytesIO Object
        temporary_location = "temp/delete.mp4"
        with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
            out.write(g.read())  ## Read bytes into file
        # close file
        if os.path.exists("temp/delete.jpg"):
            os.remove("temp/delete.jpg") # one file at a time
        out.close()
    else:
        print("Corrupted file!")