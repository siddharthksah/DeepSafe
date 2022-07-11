import subprocess
import os
def demo_selim():
    subprocess.run(["bash","models/selim_video/predict_submission.sh", "./temp", "models/selim_video/result.csv"])
demo_selim()