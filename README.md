# DeepSafe
WebBased DeepFake Detection

<h1 align="center">
  gdown
</h1>

<h4 align="center">
  Download a large file from Google Drive.
</h4>

<div align="center">
  <a href="https://pypi.python.org/pypi/gdown"><img src="https://img.shields.io/pypi/v/gdown.svg"></a>
  <a href="https://pypi.org/project/gdown"><img src="https://img.shields.io/pypi/pyversions/gdown.svg"></a>
  <a href="https://github.com/wkentaro/gdown/actions"><img src="https://github.com/wkentaro/gdown/workflows/ci/badge.svg"></a>
</div>

<div align="center">
  <img src=".readme/cli.png" width="90%">
  <img src=".readme/python.png" width="90%">
</div>

<br/>

Mega Drive
https://mega.nz/folder/faxWBbID#B84oIE9VEw2FvV8dEVW1XQ

Download Weights

'''
import gdown

#this takes a while cause the folder is quite big about 3.4G

# a folder
url = "https://drive.google.com/drive/folders/1Gan21zLaPD0wHbNF3P3a7BzgKE91BOpq?usp=sharing"
gdown.download_folder(url, quiet=True, use_cookies=False)


'''

## Description

Download a large file from Google Drive.  
If you use curl/wget, it fails with a large file because of
the security warning from Google Drive.
Supports downloading from Google Drive folders (max 50 files per folder).


## Installation

```bash
pip install gdown

# to upgrade
pip install --upgrade gdown
```


## Usage

### From Command Line

```bash
$ gdown --help
usage: gdown [-h] [-V] [-O OUTPUT] [-q] [--fuzzy] [--id] [--proxy PROXY]
             [--speed SPEED] [--no-cookies] [--no-check-certificate]
             [--continue] [--folder] [--remaining-ok]
             url_or_id
...

### From Python

```python
import gdown

# a folder
url = "https://drive.google.com/drive/folders/15uNXeRBIhVvZJIhL4yTw4IsStMhUaaxl"
gdown.download_folder(url, quiet=True, use_cookies=False)

# same as the above, but with the folder ID
id = "15uNXeRBIhVvZJIhL4yTw4IsStMhUaaxl"
gdown.download_folder(id=id, quiet=True, use_cookies=False)
```

# streamlit-multiapps
A simple framework in python to create multi page web application using streamlit.

# How to Run

1. Clone the repository:
```
$ git clone git@github.com:upraneelnihar/streamlit-multiapps
$ cd streamlit-multiapps
```

2. Install dependencies:
```
$ pip install -r requirements.txt
```

3. Start the application:
```
streamlit run app.py
```

Below are instructions to implement in in your local system using a separate development environment using the [Conda](http://conda.pydata.org/docs/index.html) package management system which comes bundled with the Anaconda Python distribution provided by Continuum Analytics.

### Step 1:
[Fork and clone](https://github.com/siddharthksah/Pose-Estimation-with-MediaPipe) a copy of this repository on to your local machine.

### Step 2:
Create a `conda` environment called `pose-estimation` and install all the necessary dependencies, the environment.yml file is uploaded in the repo for ease:

    $ conda env create --file environment.yml
    
### Step 3:
Install the extra dependencies required to run the webapp smoother:

    $ pip install watchdog

### Step 4:
Activate the `pose-estimation` environment:

    $ source activate pose-estimation

To confirm that everything has installed correctly, type

    $ which pip

at the terminal prompt. You should see something like the following:

    $ ~/anaconda/envs/pose-estimation/bin/pip

which indicates that you are using the version of `pip` that is installed inside the `pose-estimation` Conda environment and not the system-wide version of `pip` that you would normally use to install Python packages.

### Step 5:
Change into your local copy of the this repo:

    $ cd Pose-Estimation-with-MediaPipe

### Step 6:
Run the webapp:

    $ streamlit run main.py


Hosting on Google Cloud Run
Streamlit
Dockerized

