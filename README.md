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

[Container on Docher Hub](https://hub.docker.com/repository/docker/siddharth123sk/deepsafe)

```dockerfile
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

Getting the image to Docker Hub

After you made your own Docker image, you can sign up for an account on https://hub.docker.com/. After verifying your email you are ready to go and upload your first docker image.

1. Log in on https://hub.docker.com/
2. Click on Create Repository.
3. Choose a name (e.g. verse_gapminder) and a description for your repository and click Create.
4. Log into the Docker Hub from the command line

```
docker login --username=yourhubusername --email=youremail@company.com
```

just with your own user name and email that you used for the account. Enter your password when prompted. If everything worked you will get a message similar to
```
WARNING: login credentials saved in /home/username/.docker/config.json
Login Succeeded
```
Check the image ID using
```
docker images
```
and what you will see will be similar to

```
REPOSITORY              TAG       IMAGE ID         CREATED           SIZE
verse_gapminder_gsl     latest    023ab91c6291     3 minutes ago     1.975 GB
verse_gapminder         latest    bb38976d03cf     13 minutes ago    1.955 GB
rocker/verse            latest    0168d115f220     3 days ago        1.954 GB
```
and tag your image
```
docker tag bb38976d03cf yourhubusername/verse_gapminder:firsttry
```

The number must match the image ID and :firsttry is the tag. In general, a good choice for a tag is something that will help you understand what this container should be used in conjunction with, or what it represents. If this container contains the analysis for a paper, consider using that paper’s DOI or journal-issued serial number; if it’s meant for use with a particular version of a code or data version control repo, that’s a good choice too - whatever will help you understand what this particular image is intended for.

Push your image to the repository you created
```
docker push yourhubusername/verse_gapminder
```

## Deploy WebApp on Google Cloud

### Build and deploy

Command to build the application. Please remeber to change the project name and application name
```
gcloud builds submit --tag gcr.io/<ProjectName>/<AppName>  --project=<ProjectName>
```

Command to deploy the application
```
gcloud run deploy --image gcr.io/<ProjectName>/<AppName> --platform managed  --project=<ProjectName> --allow-unauthenticated
```
### If you are new to GCP - Follow the steps below -

#### Activate gcloud Command Line Tool and Push Local Image to GCP
To install the app on  Google Cloud, need to have account and gcloud tool installed in the system. 
Initiate GCloud
```
gcloud init
```
Set Project,Billing,  Service Account and Region and Zone
exmaple to set Region as Mumbai India...
```
gcloud config set compute/region asia-south1
gcloud config set compute/zone asia-south1-b
```
Enable Container Registry and Cloud Run Api
run the following command in glocud terminal
```
gcloud services enable run.googleapis.com containerregistry.googleapis.com
```

Push Local Image to GCP Cloud Container Registry
Following command will allow local docker engine tobe used by gcloud tool
```
gcloud auth configure-docker
```
Following step will create a tag of the local image as per gcp requirment.
```
docker  tag st_demo:v1.0  gcr.io/< GCP PROJECT ID > /st_demo:v1.0
```
Push Local Image to GCP Registry
```
docker push gcr.io/< GCP PROJECT ID > /st_demo:v1.0
```

Finally ! Deploy on Serverless Cloud Run
Run the following Single Line command to deploy / host the app.  
```
gcloud run deploy < service name >  --image < gcp image name>   --platform managed --allow-unauthenticated --region < your region > --memory 2Gi --timeout=3600
```
<pre>
< service name >          : Service Name User Supplied 
< gcp image name>         : Image Pushed into GCP 
< your region >           : Region was set at the Gcloud Init.
< platform managed >      : GCP Specific Parameter, consult GCP Manual for further details.
< allow-unauthenticated > : GCP Specific Parameter, consult GCP Manual for further details.
< memory >                : Memory to be allocated for the container deployment.
< timeout >               : GCP Specific Parameter, consult GCP Manual for further details. For streamlit deployment, this value should be set to a high value to avoid a timeout / connection error. 
</pre>



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

