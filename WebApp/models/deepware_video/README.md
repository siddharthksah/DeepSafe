## Deepware Scanner (CLI)

This repository contains the command-line deepfake scanner tool with the pre-trained models that are currently used at [deepware.ai](https://deepware.ai).

To get an in-depth review about the scanner and the training process please refer to [deepware.md](deepware.md). Here we will discuss the usage of the command-line scanner.

### Installation

Make sure you have a nvidia gpu with cuda support. Scanner can run both on Windows and Linux.

##### On Linux

- Clone the repo or download it as a zip and extract to a directory.
- Install the dependencies listed in [requirements](requirements.txt) file with `pip install -r requirements.txt`
- Download the [pre-trained model](https://download.deepware.ai/weights.zip) and place it in the weights directory.

##### On Windows

We packed all the requirements in a portable zip file including the model. You can [download](https://download.deepware.ai/dw-offline.zip) the zip file and start scanning right away.

### Usage

The scanning script is [scan.py](scan.py) and it has four command line arguments. Here's the usage printed by the script.

```scan.py <scan_dir> <models_dir> <cfg_file> <device>```

Let's dive into the arguments.

- `scan_dir` is the directory of videos, alternatively a file with list of video paths is supported.
- `models_dir` is the directory pre-trained models are stored. There can be multiple models and they will be ensembled automatically.
- `cfg_file` is the [config file](config.json). You shouldn't worry about this unless you want to train a new model.
- `device` is the cuda device that will be used for scanning. `cpu` is not supported as of now.

Here's an example command line.

```scan.py /home/user/videos weights config.json cuda:0```

On windows you can use `scan.bat` file with just the video folder as input.

### Training

Training scripts with a subset of [DFDC dataset](https://ai.facebook.com/datasets/dfdc/) will be published soon. Stay tuned! :bell:
