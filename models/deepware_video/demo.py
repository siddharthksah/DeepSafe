import subprocess
def demo_deepware():
	subprocess.run(["python3", "models/deepware_video/scan.py", "./temp/", "models/deepware_video/weights", "models/deepware_video/config.json", "cuda:0"])
demo_deepware()