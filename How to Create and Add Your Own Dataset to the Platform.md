# How to Create and Add Your Own Dataset to the Platform

## Introduction

This guide will help you create and add your own dataset for DeepFake detection to the platform. Follow the steps below to ensure your dataset is properly formatted and can be used for benchmarking.

## Dataset Structure

Your dataset should be structured as follows:

```
dataset_name/
├── fake/
│   ├── fake_image_1.jpg
│   ├── fake_image_2.jpg
│   └── ...
├── real/
│   ├── real_image_1.jpg
│   ├── real_image_2.jpg
│   └── ...
└── .config
```


### Folders

- `fake/`: This folder should contain all the fake images or videos. Ensure the files are named appropriately (e.g., `fake_image_1.jpg`, `fake_video_1.mp4`).
- `real/`: This folder should contain all the real images or videos. Ensure the files are named appropriately (e.g., `real_image_1.jpg`, `real_video_1.mp4`).

### `.config` File

The `.config` file should be a text file containing information about your dataset. This information will be displayed on the platform to provide context about the dataset. Here is an example of the `.config` file content:

Dataset Name: My Custom Dataset
Description: This dataset contains images and videos for DeepFake detection. The 'fake' folder contains manipulated media, while the 'real' folder contains authentic media.
Source: Created for research purposes.
Number of Files:
- Real: 100 images/videos
- Fake: 100 images/videos


## Adding Your Dataset to the Platform

1. **Create the Dataset Structure**: Follow the structure outlined above. Ensure that the `fake/` and `real/` folders contain the appropriate files and that the `.config` file is created and populated with information about your dataset.

2. **Name Your Dataset Folder**: Give your dataset folder a unique and descriptive name (e.g., `my_custom_dataset`).

3. **Place the Dataset in the Correct Directory**:
    - For image datasets, place your dataset folder inside the `datasets/image/` directory.
    - For video datasets, place your dataset folder inside the `datasets/video/` directory.

Here is an example of the directory structure for an image dataset:

```
datasets/
├── image/
│   ├── my_custom_dataset/
│   │   ├── fake/
│   │   │   ├── fake_image_1.jpg
│   │   │   ├── fake_image_2.jpg
│   │   │   └── ...
│   │   ├── real/
│   │   │   ├── real_image_1.jpg
│   │   │   ├── real_image_2.jpg
│   │   │   └── ...
│   │   └── .config
│   └── ...
└── video/
    └── ...
```

4. **Verify the Dataset**: Ensure that the dataset is correctly placed in the directory and that all files are in the appropriate folders.

5. **Load the Dataset on the Platform**: Once your dataset is added, you should be able to select and use it for benchmarking within the platform.

## Additional Tips

- **Consistent Naming**: Ensure that file names are consistent and descriptive. This will help in easily identifying and managing files.
- **Quality Check**: Verify that all files are correctly labeled as either `real` or `fake` and that the `.config` file accurately describes the dataset.
