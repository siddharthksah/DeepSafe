FROM nvidia/cuda:11.0-base
RUN apt-get update && apt-get install -y python3-pip  wget
WORKDIR /
COPY . /

RUN python3 -m pip install -U pip && python3 -m pip install -r requirements.txt
ENV PYTHONUNBUFFERED 1

EXPOSE 5000
CMD ["python3","app.py"]
