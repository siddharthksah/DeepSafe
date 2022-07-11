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
