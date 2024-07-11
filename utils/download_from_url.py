import subprocess
import os
import shutil

def download(url):
    """
    Downloads a video from the specified URL using the `you-get` command-line utility.
    
    Args:
        url (str): The URL of the video to be downloaded.
        
    Returns:
        CompletedProcess: The result of the `subprocess.run` call.
    """
    output_dir = "temp/"
    output_filename = "delete"
    command = ["you-get", url, "--output-dir", output_dir, "--output-filename", output_filename]
    
    return subprocess.run(command)
