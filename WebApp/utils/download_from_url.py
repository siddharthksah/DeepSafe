import subprocess, os

import shutil
def download(url):
    return subprocess.run(["you-get", url, "--output-dir", "temp/", "--output-filename", "delete"])
     
