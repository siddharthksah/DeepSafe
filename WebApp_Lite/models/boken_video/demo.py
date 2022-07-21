import warnings
warnings.filterwarnings("ignore")
import subprocess

def demo_boken():
    subprocess.run(["python3", "models/boken_video/inference.py"])
demo_boken()
