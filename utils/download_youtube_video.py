import subprocess, os

#url = 'https://www.youtube.cm/watch?v=dP15zlyra3c'
#url = "https://www.youtube.com/watch?v=PdoTxtExOfY"
#url = "https://www.youtube.com/watch?v=WNCl-69POro&ab_channel=4KRelaxationChannel"
#url = "https://www.youtube.com/watch?v=mVrGcjakM70&ab_channel=TheDailyShowwithTrevorNoah"
#url = "https://www.youtube.com/watch?v=RDfjXj5EGqI&ab_channel=LullabyBaby"
#url = "https://www.youtube.com/watch?v=kOkQ4T5WO9E&list=PLDeVhw7bs_K5_jTmR5AxhKHM5we_x7WKM"
#url = "https://www.youtube.com/watch?v=kOkQ4T5WO9E&list=PLDeVhw7bs_K5_jTmR5AxhKHM5we_x7WKM&ab_channel=CalvinHarrisVEVO"

def download_video(url):
    download_status = str(subprocess.run(["yt-dlp", "-o", "./temp/delete.mp4", url, "--max-filesize", "50m", "--no-playlist"]))
    #print(download_status)
    #print((download_status.split(",")[-1][-2]))
    if (download_status.split(",")[-1][-2]) == "1":
            #print("wth")
            return False

    else:
        if len(os.listdir('./temp')) == 0:
            return False
        else:
            for files in os.listdir("./temp"):
                if files == "delete.mp4":
                    return True
        

#print(download_video(url))
