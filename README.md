# DeepSafe
DeepSafe - DeepFake Detection made easy

## DeepSafe consiste of 3 tools.
1. WebApp
2. DeepSafe API
3. Chrome Extension

# WebApp - [Live here](https://deepsafe-75twwvyl5q-de.a.run.app)
This is a limited access app, for the full access [contact me](mailto:siddharth123sk@gmail.com).

1. Clone the repository:
```
git clone https://github.com/siddharthksah/DeepSafe
cd DeepSafe
```
2. Creating conda environment

### Python version
* Main supported version : <strong>3.8</strong> <br>
* Other supported versions : <strong>3.7</strong> & <strong>3.9</strong>

```
conda create -n deepsafe python==3.8 -y
conda activate deepsafe
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Download Model Weights

| Service | Google Drive  | Mega Drive  |
| :---:   | :-: | :-: |
  | Link | [Google Drive](https://drive.google.com/drive/folders/1Gan21zLaPD0wHbNF3P3a7BzgKE91BOpq?usp=sharing) | [Mega Drive](https://mega.nz/folder/faxWBbID#B84oIE9VEw2FvV8dEVW1XQ) |
  
Or you can use [Gdown](https://github.com/wkentaro/gdown)

  <a href="https://pypi.python.org/pypi/gdown"><img src="https://img.shields.io/pypi/v/gdown.svg"></a>
  
### Installation

```bash
pip install gdown

# to upgrade
pip install --upgrade gdown
```
  
```
import gdown

#this takes a while cause the folder is quite big about 3.4G

url = "https://drive.google.com/drive/folders/1Gan21zLaPD0wHbNF3P3a7BzgKE91BOpq?usp=sharing

gdown.download_folder(url, quiet=True, use_cookies=False)
```
  
  


5. Start the application:
```
streamlit run main.py
```

### Dockerize the webapp

```
#Base Image to use
FROM python:3.7.9-slim

#Expose port 8080
EXPOSE 8080

#Optional - install git to fetch packages directly from github
RUN apt-get update && apt-get install -y git
RUN apt-get install ffmpeg libsm6 libxext6  -y


#Copy Requirements.txt file into app directory
COPY requirements.txt app/requirements.txt

#install all requirements in requirements.txt
RUN pip install -r app/requirements.txt

#Copy all files in current directory into app directory
COPY . /app

#Change Working Directory to app directory
WORKDIR /app

#Run the application on port 8080
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]
```
Building the Docker Image
```
docker build -f Dockerfile -t app:latest .

```
Running the docker image and creating the container
```
docker run -p 8501:8501 app:latest
```



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

# Python ML project skeleton

This is an opinionated project skeleton for a Python based machine learning projects.  
This skeleton is to be used as the default project start for any Python ML project (unless arguments support otherwise).  
While the project is heavily opinionated, opinions are welcomed to be discussed: feel free to open an issue or comment.

## Installing the package

1. Clone the repo

    ```bash
    git clone git@github.com:datarootsio/ml-skeleton-py.git
    cd ml-skeleton-py
    ```

2. Install dependencies using [pip](https://pip.pypa.io/en/stable/installing/). The following command
will install the dependencies from `setup.py`. In the backend it will run `pip install -e ".[test, serve]"`. Note that installing dependencies with `-e` 
editable mode is needed to properly run unit tests. `[test, serve]` is optional. `test` refers to
unit test dependencies and `serve` refers to deployment dependencies.

    ```bash
    make install
    ```

## Running the project

Preferably, you can use make commands (from `Makefile`) or directly run scripts from `scripts`.  
Refer to section below for the descriptions of make commands. Before running it, consider creating  
a virtual environment.  

**Makefile and test example**

Try out the `make` commands on the example `creditcard.csv` dataset model (see `make help`).

```
clean                          clean artifacts
coverage                       create coverage report
generate-dataset               run ETL pipeline
help                           show help on available commands
lint                           flake8 linting and black code style
run-pipeline                   clean artifacts -> generate dataset -> train -> serve
serve                          serve trained model with a REST API using dploy-kickstart
test-docker                    run unit tests in docker environment
test                           run unit tests in the current virtual environment
train                          train the model, you can pass arguments as follows: make ARGS="--foo 10 --bar 20" train
```

Note the dependency: `generate-dataset` > `train` > `serve`.

## Docker

Currently, you can find the following docker files:  
1. `jupyter.Dockerfile` builds an image for running notebooks.  
2. `test.Dockerfile` builds an image to run all tests in (`make test-docker`).
3. `serve.Dockerfile` build an image to serve the trained model via a REST api.
To ease the serving it uses open source `dploy-kickstart` module. To find more info
about `dploy-kickstart` click [here](https://github.com/dploy-ai/dploy-kickstart/).

Finally, you can start all services using `docker-compose`:  
for example `docker-compose up jupyter` or `docker-compose up serve`.  

Do you need a notebook for development? Just run `docker-compose up jupyter`. It will launch a Jupyter Notebook 
with access to your local development files.

## Deploying the API

Calling `make serve` will start a Flask based API using `dploy-kickstart`
wrapper. 

In `ml_skeleton_py/model/predict.py` file, there is `# @dploy endpoint predict`
annotation above the `predict` method. 

From `# @dploy endpoint predict` annotation, we are telling `dploy-kickstart` 
that the url that we need to do the post request is `http://localhost:8080/predict`.
As another example, if the annotation would be `# @dploy endpoint score` then the url
would change to `http://localhost:8080/score`.  

Going back to our case, the posted data to `http://localhost:8080/predict` url will be
the argument of the exposed method which is `def predict(body)`. 

As a concrete example;

After calling `make serve`, we can do our predictions with the following curl command.
In this case, `def predict(body)` method will be triggered and the value of the `--data`
will be the argument of `def predict(body)` function, i.e. `body`.

```sh
 curl --request POST \
  --url http://localhost:8080/predict \
  --header 'content-type: application/json' \
  --data '{"model_f_name": "lr.joblib",
           "features": [28692.0,-29.200328590574397,16.1557014298057,-30.013712485724803,6.47673117996833,-21.2258096535165,-4.90299739658728,
                        -19.791248405247,19.168327389730102,-3.6172417860425496,-7.87012194292549,4.06625507293473,-5.66149242261771,1.2929501445424199,
                        -5.07984568135779,-0.126522740416921,-5.24447151974264,-11.274972585125198,-4.67843652929376,0.650807370688892,1.7158618242835801,1.8093709332883998,
                        -2.1758152034214198,-1.3651041075509,0.174286359566544,2.10386807204715,-0.20994399913056697,1.27868097084218,0.37239271433854104,
                        99.99]
           }'
```

To test the health of the deployed model, you can make a get request as shown below;

```sh
    curl --request GET \
      --url http://localhost:8080/healthz
```



## Project Structure Overview 
The project structure tree is shown below. This structure is designed
in a way to easily develop ML projects. Feedback / PRs are always welcome
about the structure.

```
.
├── .github             # Github actions CI pipeline
|
├── data                
│   ├── predictions     # predictions data, calculated using the model
│   ├── raw             # immutable original data
│   ├── staging         # data obtained after preprocessing, i.e. cleaning, merging, filtering etc.
│   └── transformed     # data ready for modeling (dataset containing features and label)
|
├── docker              # Store all dockerfiles
|
├── ml_skeleton_py      # Logic of the model
│   ├── etl             # Logic for cleaning the data and preparing train / test set 
│   └── model           # Logic for ML model including CV, parameter tuning, model evaluation
|
├── models              # Store serialized fitted models
|
├── notebooks           # Store prototype or exploration related .ipynb notebooks
|
├── reports             # Store textual or visualisation content, i.e. pdf, latex, .doc, .txt 
|
├── scripts             # Call ml_skeleton_py module from here e.g. cli for training
|
└── tests               # Unit tests
```

## Best practices for development

- Make sure that `docker-compose up test` runs properly.  
- In need for a Notebook? Use the docker image: `docker-compose up jupyter`.
- Commit often, perfect later.
- Integrate `make test` with your CI pipeline.
- Capture `stdout` when deployed.

# Containerized [Streamlit](https://www.streamlit.io/) web app

This repository is featured in a 3-part series on [Deploying web apps with Streamlit, Docker, and AWS](https://collinprather.github.io/blog/docker/aws/2020/03/10/streamlit-docker-pt1.html). Checkout the blog posts!

![app](images/app.gif)

---

## Setup instructions

If you are interested in the web app backed by a Postgres database, checkout out the [`docker-compose+postgres` branch](https://github.com/collinprather/streamlit-docker/tree/docker-compose+postgres). 

### Getting up and running locally

```shell
$ git clone https://github.com/collinprather/streamlit-docker.git
$ cd streamlit-docker/
$ docker image build -t streamlit:app .
$ docker container run -p 8501:8501 -d streamlit:app
```

Then, the web app will be available at `http://localhost:8501/`

To shut down the web app when you're done, you can find the process running your container with

```shell
$ docker ps | grep 'streamlit:app'
6d1871137019        streamlit:app       "/bin/sh -c 'streaml…"   8 minutes ago       Up 8 minutes        0.0.0.0:8501->8501/tcp   <weird_name>
```

Then stop that process with the following command.

```shell
$ docker kill <weird_name>
<weird_name>
$
```

### Deploying to the cloud

Refer to my [blog post](https://collinprather.github.io/blog/docker/aws/2020/03/11/streamlit-docker-pt2.html)!




![Docker Automated build](https://img.shields.io/docker/automated/aminehy/docker-streamlit-app)
![Docker Pulls](https://img.shields.io/docker/pulls/aminehy/docker-streamlit-app)

# Description
This repository contains a docker image that allows running streamlit web application. It can be used to test the application and/or deploy to a cloud service like Google Cloud, Heroku, Amazon AWS
 

# Run the docker container
Simply enter the following command to run your application
```
docker run -ti --rm aminehy/docker-streamlit-app:latest
```

**Local development**
 - Mount your working folder in the container 
  ```
  docker run -ti --rm -v $(pwd):/app aminehy/docker-streamlit-app:latest
  ```

 - If your main file name is different to  `main.py` (e.g. `app.py`)
 ```
 docker run -ti --rm -v $(pwd):/app aminehy/docker-streamlit-app:latest streamlit run name_main_file.py
 ```

- To access the docker container in the bash mode
```
docker run -ti --rm aminehy/deploy_streamlit_app:latest bash
```

# Build docker image
You can build this docker image from a dockerfile using this command
```
docker build -t aminehy/docker-streamlit-app:latest .
```





---
title: "Build your Python image"
keywords: python, build, images, dockerfile
description: Learn how to build your first Docker image by writing a Dockerfile
---

{% include_relative nav.html selected="1" %}

## Prerequisites

Work through the orientation and setup in Get started [Part 1](../../get-started/index.md) to understand Docker concepts.

{% include guides/enable-buildkit.md %}

## Overview

Now that we have a good overview of containers and the Docker platform, let’s take a look at building our first image. An image includes everything needed to run an application - the code or binary, runtime, dependencies, and any other file system objects required.

To complete this tutorial, you need the following:

- Python version 3.8 or later. [Download Python](https://www.python.org/downloads/){: target="_blank" rel="noopener" class="_"}
- Docker running locally. Follow the instructions to [download and install Docker](../../desktop/index.md)
- An IDE or a text editor to edit files. We recommend using [Visual Studio Code](https://code.visualstudio.com/Download){: target="_blank" rel="noopener" class="_"}.

## Sample application

Let’s create a simple Python application using the Flask framework that we’ll use as our example. Create a directory in your local machine named `python-docker` and follow the steps below to create a simple web server.

```console
$ cd /path/to/python-docker
$ pip3 install Flask
$ pip3 freeze | grep Flask >> requirements.txt
$ touch app.py
```

Now, let’s add some code to handle simple web requests. Open this working directory in your favorite IDE and enter the following code into the `app.py` file.

```python
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, Docker!'
```

## Test the application

Let’s start our application and make sure it’s running properly. Open your terminal and navigate to the working directory you created.

```console
$ python3 -m flask run
```

To test that the application is working properly, open a new browser and navigate to `http://localhost:5000`.

Switch back to the terminal where our server is running and you should see the following requests in the server logs. The data and timestamp will be different on your machine.

```shell
127.0.0.1 - - [22/Sep/2020 11:07:41] "GET / HTTP/1.1" 200 -
```

## Create a Dockerfile for Python

Now that our application is running properly, let’s take a look at creating a Dockerfile.

{% include guides/create-dockerfile.md %}

Next, we need to add a line in our Dockerfile that tells Docker what base image
we would like to use for our application.

```dockerfile
# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster
```

Docker images can be inherited from other images. Therefore, instead of creating our own base image, we’ll use the official Python image that already has all the tools and packages that we need to run a Python application.

> **Note**
>
> To learn more about creating your own base images, see [Creating base images](https://docs.docker.com/develop/develop-images/baseimages/).

To make things easier when running the rest of our commands, let’s create a working directory. This instructs Docker to use this path as the default location for all subsequent commands. By doing this, we do not have to type out full file paths but can use relative paths based on the working directory.

```dockerfile
WORKDIR /app
```

Usually, the very first thing you do once you’ve downloaded a project written in Python is to install `pip` packages. This ensures that your application has all its dependencies installed.

Before we can run `pip3 install`, we need to get our `requirements.txt` file into our image. We’ll use the `COPY` command to do this. The `COPY` command takes two parameters. The first parameter tells Docker what file(s) you would like to copy into the image. The second parameter tells Docker where you want that file(s) to be copied to. We’ll copy the `requirements.txt` file into our working directory `/app`.

```dockerfile
COPY requirements.txt requirements.txt
```

Once we have our `requirements.txt` file inside the image, we can use the `RUN` command to execute the command `pip3 install`. This works exactly the same as if we were running `pip3 install` locally on our machine, but this time the modules are installed into the image.

```dockerfile
RUN pip3 install -r requirements.txt
```

At this point, we have an image that is based on Python version 3.8 and we have installed our dependencies. The next step is to add our source code into the image. We’ll use the `COPY` command just like we did with our `requirements.txt` file above.

```dockerfile
COPY . .
```

This `COPY` command takes all the files located in the current directory and copies them into the image. Now, all we have to do is to tell Docker what command we want to run when our image is executed inside a container. We do this using the `CMD` command. Note that we need to make the application externally visible (i.e. from outside the container) by specifying `--host=0.0.0.0`.

```dockerfile
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
```

Here's the complete Dockerfile.

```dockerfile
# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
```

### Directory structure

Just to recap, we created a directory in our local machine called `python-docker` and created a simple Python application using the Flask framework. We also used the `requirements.txt` file to gather our requirements, and created a Dockerfile containing the commands to build an image. The Python application directory structure would now look like:

```shell
python-docker
|____ app.py
|____ requirements.txt
|____ Dockerfile
```

## Build an image

Now that we’ve created our Dockerfile, let’s build our image. To do this, we use the `docker build` command. The `docker build` command builds Docker images from a Dockerfile and a “context”. A build’s context is the set of files located in the specified PATH or URL. The Docker build process can access any of the files located in this context.

The build command optionally takes a `--tag` flag. The tag is used to set the name of the image and an optional tag in the format `name:tag`. We’ll leave off the optional `tag` for now to help simplify things. If you do not pass a tag, Docker uses “latest” as its default tag.

Let’s build our first Docker image.

```console
$ docker build --tag python-docker .
[+] Building 2.7s (10/10) FINISHED
 => [internal] load build definition from Dockerfile
 => => transferring dockerfile: 203B
 => [internal] load .dockerignore
 => => transferring context: 2B
 => [internal] load metadata for docker.io/library/python:3.8-slim-buster
 => [1/6] FROM docker.io/library/python:3.8-slim-buster
 => [internal] load build context
 => => transferring context: 953B
 => CACHED [2/6] WORKDIR /app
 => [3/6] COPY requirements.txt requirements.txt
 => [4/6] RUN pip3 install -r requirements.txt
 => [5/6] COPY . .
 => [6/6] CMD [ "python3", "-m", "flask", "run", "--host=0.0.0.0"]
 => exporting to image
 => => exporting layers
 => => writing image sha256:8cae92a8fbd6d091ce687b71b31252056944b09760438905b726625831564c4c
 => => naming to docker.io/library/python-docker
```

## View local images

To see a list of images we have on our local machine, we have two options. One is to use the CLI and the other is to use [Docker Desktop](../../desktop/dashboard.md#explore-your-images). As we are currently working in the terminal let’s take a look at listing images using the CLI.

To list images, simply run the `docker images` command.

```console
$ docker images
REPOSITORY      TAG               IMAGE ID       CREATED         SIZE
python-docker   latest            8cae92a8fbd6   3 minutes ago   123MB
python          3.8-slim-buster   be5d294735c6   9 days ago      113MB
```

You should see at least two images listed. One for the base image `3.8-slim-buster` and the other for the image we just built `python-docker:latest`.

## Tag images

As mentioned earlier, an image name is made up of slash-separated name components. Name components may contain lowercase letters, digits and separators. A separator is defined as a period, one or two underscores, or one or more dashes. A name component may not start or end with a separator.

An image is made up of a manifest and a list of layers. Do not worry too much about manifests and layers at this point other than a “tag” points to a combination of these artifacts. You can have multiple tags for an image. Let’s create a second tag for the image we built and take a look at its layers.

To create a new tag for the image we’ve built above, run the following command.

```console
$ docker tag python-docker:latest python-docker:v1.0.0
```

The `docker tag` command creates a new tag for an image. It does not create a new image. The tag points to the same image and is just another way to reference the image.

Now, run the `docker images` command to see a list of our local images.

```console
$ docker images
REPOSITORY      TAG               IMAGE ID       CREATED         SIZE
python-docker   latest            8cae92a8fbd6   4 minutes ago   123MB
python-docker   v1.0.0            8cae92a8fbd6   4 minutes ago   123MB
python          3.8-slim-buster   be5d294735c6   9 days ago      113MB
```

You can see that we have two images that start with `python-docker`. We know they are the same image because if you take a look at the `IMAGE ID` column, you can see that the values are the same for the two images.

Let’s remove the tag that we just created. To do this, we’ll use the `rmi` command. The `rmi` command stands for remove image.

```console
$ docker rmi python-docker:v1.0.0
Untagged: python-docker:v1.0.0
```

Note that the response from Docker tells us that the image has not been removed but only “untagged”. You can check this by running the `docker images` command.

```console
$ docker images
REPOSITORY      TAG               IMAGE ID       CREATED         SIZE
python-docker   latest            8cae92a8fbd6   6 minutes ago   123MB
python          3.8-slim-buster   be5d294735c6   9 days ago      113MB
```

Our image that was tagged with `:v1.0.0` has been removed, but we still have the `python-docker:latest` tag available on our machine.

## Next steps

In this module, we took a look at setting up our example Python application that we will use for the rest of the tutorial. We also created a Dockerfile that we used to build our Docker image. Then, we took a look at tagging our images and removing images. In the next module we’ll take a look at how to:

[Run your image as a container](run-containers.md){: .button .primary-btn}

## Feedback

Help us improve this topic by providing your feedback. Let us know what you think by creating an issue in the [Docker Docs](https://github.com/docker/docker.github.io/issues/new?title=[Python%20docs%20feedback]){:target="_blank" rel="noopener" class="_"} GitHub repository. Alternatively, [create a PR](https://github.com/docker/docker.github.io/pulls){:target="_blank" rel="noopener" class="_"} to suggest updates.

<br />
