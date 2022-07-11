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

<div align="center">

![Streamlit Prophet](streamlit_prophet/references/logo.png)

[![CI status](https://github.com/artefactory-global/streamlit_prophet/actions/workflows/ci.yml/badge.svg?branch%3Amain&event%3Apush)](https://github.com/artefactory-global/streamlit_prophet/actions/workflows/ci.yml?query=branch%3Amain)
[![Python Version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue.svg)](#supported-python-versions)
[![Dependencies Status](https://img.shields.io/badge/dependabots-active-informational.svg)](https://github.com/artefactory-global/streamlit_prophet/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/maximelutel/streamlit_prophet/main/streamlit_prophet/app/dashboard.py)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-informational.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory-global/streamlit_prophet/blob/main/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%F0%9F%9A%80-semantic%20versions-informational.svg)](https://github.com/artefactory-global/streamlit_prophet/releases)
[![License](https://img.shields.io/badge/License-MIT-informational.svg)](https://github.com/artefactory-global/streamlit_prophet/blob/main/LICENSE)

Deploy a [Streamlit](https://streamlit.io/) app to train, evaluate and optimize a [Prophet](https://facebook.github.io/prophet/) forecasting model visually

## ⭐  Quick Start  ⭐

[Test the app online](https://share.streamlit.io/maximelutel/streamlit_prophet/main/streamlit_prophet/app/dashboard.py) with shared computing resources & [read introductory article](https://medium.com/artefact-engineering-and-data-science/visual-time-series-forecasting-with-streamlit-prophet-71d86a769928?source=friends_link&sk=590cca0d24f53f73a9fdb0490a9a47a7)

If you plan to use the app regularly, you should install the package and run it locally:
```bash
pip install -U streamlit_prophet
streamlit_prophet deploy dashboard
```

</div>

https://user-images.githubusercontent.com/56996548/126762714-f2d3f3a1-7098-4a86-8c60-0a69d0f913a7.mp4

## 💻 Requirements

### Python version
* Main supported version : <strong>3.7</strong> <br>
* Other supported versions : <strong>3.8</strong> & <strong>3.9</strong>

Please make sure you have one of these versions installed to be able to run the app on your machine.

### Operating System
Windows users have to install [WSL2](https://docs.microsoft.com/en-us/windows/wsl/) to download the package. 
This is due to an incompatibility between Windows and Prophet's main dependency (pystan). 
Other operating systems should work fine.

## ⚙️ Installation


### Create a virtual environment (optional)
We strongly advise to create and activate a new virtual environment, to avoid any dependency issue.

For example with conda:
```bash
pip install conda; conda create -n streamlit_prophet python=3.7; conda activate streamlit_prophet
```

Or with virtualenv:
```bash
pip install virtualenv; python3.7 -m virtualenv streamlit_prophet --python=python3.7; source streamlit_prophet/bin/activate
```


### Install package
Install the package from PyPi (it should take a few minutes):
```bash
pip install -U streamlit_prophet
```

Or from the main branch of this repository:
```bash
pip install git+https://github.com/artefactory-global/streamlit_prophet.git@main
```


## 📈 Usage

Once installed, run the following command from CLI to open the app in your default web browser:

```bash
streamlit_prophet deploy dashboard
```

Now you can train, evaluate and optimize forecasting models in a few clicks.
All you have to do is to upload a time series dataset. 
This dataset should be a csv file that contains a date column, a target column and optionally some features, like on the example below:

![](streamlit_prophet/references/input_format.png)

Then, follow the guidelines in the sidebar to:

* <strong>Prepare data</strong>: Filter, aggregate, resample and/or clean your dataset.
* <strong>Choose model parameters</strong>: Default parameters are available but you can tune them.
Look at the tooltips to understand how each parameter is impacting forecasts.
* <strong>Select evaluation method</strong>: Define the evaluation process, the metrics and the granularity to
assess your model performance.
* <strong>Make a forecast</strong>: Make a forecast on future dates that are not included in your dataset,
with the model previously trained.

Once you are satisfied, click on "save experiment" to download all plots and data locally.


## 🛠️ How to contribute ?

All contributions, ideas and bug reports are welcome! 
We encourage you to open an [issue](https://github.com/artefactory-global/streamlit_prophet/issues) for any change you would like to make on this project.


For more information, see [`CONTRIBUTING`](https://github.com/artefactory-global/streamlit_prophet/blob/main/CONTRIBUTING.md) instructions.
If you wish to containerize the app, see [`DOCKER`](https://github.com/artefactory-global/streamlit_prophet/blob/main/DOCKER.md) instructions.

# ml_project_template

## Tutorial for deployment website in AWS instance [can be found here](./docs/tutorial.md)

# Tutorial
This is tutorial for local development on local machine
# Install 
```
sudo -s
apt update
apt install -y npm
apt install -y python3.6 python3-pip
apt install -y nginx
apt update && apt install -y libsm6 libxext6
apt install libxrender1
apt install -y redis-server
```

### Clone the repo
```
apt install git
git clone https://github.com/thanhhau097/ml_project_template.git 
```

### Install requirements
```
cd ml_project_template
alias python=python3
alias pip=pip3
```

#### Requirements for Flask app
```
cd api
pip install -r requirements.txt
npm install
```
#### Requirements for web app
```
cd ../web
npm install
```

### Change your custom path
#### api/app.py
You need to config your AWS account in local machine: (optional, it is useful when you want to save uploaded data to AWS s3)
```
apt install awscli
aws configure
```

Change your bucket in AWS S3 (it is optional, when you need to upload user data to s3 bucket)
```
async_data = {
    'data': {'image': image_utils.encode(image), 'result': result},
    'bucket': 'your-bucket',
    'object_name': 'file-path-in-bucket/{}.pkl'.format(file_name)
}
```

#### nginx/nginx.conf
Comment out the SSL config in this file because you don't need domain name for local development (line 4-6, 29.41).
If you want to deploy into production, please see [this link](./tutorial.md).

### Write your code
There are 3 tasks that you need to do for your project:

1. Write prediction for you model in `model/predictor.py` and update your weights in model/weights folder
2. Write your API in `api/app.py` using Flask framework (you can use the template that was written for image)
3. Write your web app using ReactJS (you can use the demo template that I wrote in `web/`)

# How to run the service
Open multiple terminal windows, each process should be handle in one window.

1. Web
```
cd web/
npm run build
npm run start
```

2. Flask
```
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHONPATH=$PYTHONPATH:./ 
python api/app.py 
```

3. Redis

Change redis conf in /etc/redis/redis.conf 
```
bind 0.0.0.0
```

```
systemctl stop redis
systemctl start redis
```

4. Celery

Change 'redis://redis:6379/0' to 'redis://localhost:6379/0' because you are running redis in local machine
```
celery worker -A api.app.celery_app --loglevel=info
```

5. Nginx

Change the app and web in nginx/nginx.conf file to 0.0.0.0
```
sudo -s
rm /etc/nginx/sites-enabled/default
cp nginx/nginx.conf /etc/nginx/sites-enabled/
systemctl reload nginx
```

Now you can go to your browser and see what is happening: [0.0.0.0](0.0.0.0)


### Tools are used in this template
1. Flask
2. Redis
3. ReactJS
4. Nginx
5. Certbot (optional, when you have a domain name)
6. Celery
7. Docker
8. Jenkins (optional)

# PyTorch Project Template
A simple and well designed structure is essential for any Deep Learning project, so after a lot practice and contributing in pytorch projects here's a pytorch project template that combines **simplicity, best practice for folder structure** and **good OOP design**. 
The main idea is that there's much same stuff you do every time when you start your pytorch project, so wrapping all this shared stuff will help you to change just the core idea every time you start a new pytorch project. 

**So, here’s a simple pytorch template that help you get into your main project faster and just focus on your core (Model Architecture, Training Flow, etc)**

In order to decrease repeated stuff, we recommend to use a high-level library. You can write your own high-level library or you can just use some third-part libraries such as [ignite](https://github.com/pytorch/ignite), [fastai](https://github.com/fastai/fastai), [mmcv](https://github.com/open-mmlab/mmcv) … etc. This can help you write compact but full-featured training loops in a few lines of code. Here we use ignite to train mnist as an example.

# Requirements
- [yacs](https://github.com/rbgirshick/yacs) (Yet Another Configuration System)
- [PyTorch](https://pytorch.org/) (An open source deep learning platform) 
- [ignite](https://github.com/pytorch/ignite) (High-level library to help with training neural networks in PyTorch)

# Table Of Contents
-  [In a Nutshell](#in-a-nutshell)
-  [In Details](#in-details)
-  [Future Work](#future-work)
-  [Contributing](#contributing)
-  [Acknowledgments](#acknowledgments)

# In a Nutshell   
In a nutshell here's how to use this template, so **for example** assume you want to implement ResNet-18 to train mnist, so you should do the following:
- In `modeling`  folder create a python file named whatever you like, here we named it `example_model.py` . In `modeling/__init__.py` file, you can build a function named `build_model` to call your model

```python
from .example_model import ResNet18

def build_model(cfg):
    model = ResNet18(cfg.MODEL.NUM_CLASSES)
    return model
``` 

   
- In `engine`  folder create a model trainer function and inference function. In trainer function, you need to write the logic of the training process, you can use some third-party library to decrease the repeated stuff.

```python
# trainer
def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn):
 """
 implement the logic of epoch:
 -loop on the number of iterations in the config and call the train step
 -add any summaries you want using the summary
 """
pass

# inference
def inference(cfg, model, val_loader):
"""
implement the logic of the train step
- run the tensorflow session
- return any metrics you need to summarize
 """
pass
```

- In `tools`  folder, you create the `train.py` .  In this file, you need to get the instances of the following objects "Model",  "DataLoader”, “Optimizer”, and config
```python
# create instance of the model you want
model = build_model(cfg)

# create your data generator
train_loader = make_data_loader(cfg, is_train=True)
val_loader = make_data_loader(cfg, is_train=False)

# create your model optimizer
optimizer = make_optimizer(cfg, model)
```

- Pass the all these objects to the function `do_train` , and start your training
```python
# here you train your model
do_train(cfg, model, train_loader, val_loader, optimizer, None, F.cross_entropy)
```

**You will find a template file and a simple example in the model and trainer folder that shows you how to try your first model simply.**


# In Details
```
├──  config
│    └── defaults.py  - here's the default config file.
│
│
├──  configs  
│    └── train_mnist_softmax.yml  - here's the specific config file for specific model or dataset.
│ 
│
├──  data  
│    └── datasets  - here's the datasets folder that is responsible for all data handling.
│    └── transforms  - here's the data preprocess folder that is responsible for all data augmentation.
│    └── build.py  		   - here's the file to make dataloader.
│    └── collate_batch.py   - here's the file that is responsible for merges a list of samples to form a mini-batch.
│
│
├──  engine
│   ├── trainer.py     - this file contains the train loops.
│   └── inference.py   - this file contains the inference process.
│
│
├── layers              - this folder contains any customed layers of your project.
│   └── conv_layer.py
│
│
├── modeling            - this folder contains any model of your project.
│   └── example_model.py
│
│
├── solver             - this folder contains optimizer of your project.
│   └── build.py
│   └── lr_scheduler.py
│   
│ 
├──  tools                - here's the train/test model of your project.
│    └── train_net.py  - here's an example of train model that is responsible for the whole pipeline.
│ 
│ 
└── utils
│    ├── logger.py
│    └── any_other_utils_you_need
│ 
│ 
└── tests					- this foler contains unit test of your project.
     ├── test_data_sampler.py
```


# Future Work

# Contributing
Any kind of enhancement or contribution is welcomed.


# Acknowledgments




[![maintained by dataroots](https://img.shields.io/badge/maintained%20by-dataroots-%2300b189)](https://dataroots.io)
[![PythonVersion](https://img.shields.io/pypi/pyversions/gino_admin)](https://img.shields.io/pypi/pyversions/gino_admin)
[![tests](https://github.com/datarootsio/ml-skeleton-py/workflows/tests/badge.svg?branch=master)](https://github.com/datarootsio/ml-skeleton-py/actions)
[![Codecov](https://codecov.io/github/datarootsio/ml-skeleton-py/badge.svg?branch=master&service=github)](https://github.com/datarootsio/ml-skeleton-py/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
![](https://scontent.fbru1-1.fna.fbcdn.net/v/t1.0-9/94305647_112517570431823_3318660558911176704_o.png?_nc_cat=111&_nc_sid=e3f864&_nc_ohc=-spbrtnzSpQAX_qi7iI&_nc_ht=scontent.fbru1-1.fna&oh=483d147a29972c72dfb588b91d57ac3c&oe=5F99368A "Logo")

**NOTE:** This is a best-practices first project template that allows you to get started on a new machine learning project. For more info on how to use it check out [HOWTO.md](HOWTO.md). Feel free to use it how it suits you best 🚀

# `PROJECT NAME`

> project for: `client name`  

## Objective

`ADD OBJECTIVE OF CASE`

## Explorative results

`SHORT SUMMARY AND LINK TO REPORT`

## Modelling results

`SHORT SUMMARY AND LINK TO REPORT`

## Usage

`ADD EXPLANATION`

## Configuration

`RELEVANT INFO ON CONFIGURATION`

## Deploy

`RELEVANT INFO ON DEPLOYMENT`

> copyright by `your company`
> main developer `developer_name` (`developer email`)

