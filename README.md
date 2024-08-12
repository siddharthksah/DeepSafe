# DeepSafe 🐶 - Open Source DeepFake Detection

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)

DeepSafe is a Streamlit-based web application for DeepFake detection, offering an easy-to-use interface for analyzing images and videos. Users can add their own deepfake detection models and compare them with existing models out of the box.

## WebApp

[Live here](https://deepsafe.disconinja.duckdns.org/) (Limited access, for full access please [contact me](mailto:siddharth123sk@gmail.com)).

## Table of Contents

- [Features](#features)
- [Usage](#usage)
- [Installation](#installation)
- [Additional Sections](#additional-sections)
- [WebApp](#webapp)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Contact Information](#contact-information)

## Features

✨ **Multi-model Support**: Users can select from multiple DeepFake detection models for both images and videos.  
📁 **File Upload**: Supports uploading images (jpg, png, jpeg) and videos (mp4, mov).  
🌐 **URL Input**: Allows users to input URLs for image or video analysis.  
⚙️ **Processing Unit Selection**: Option to use GPU for supported models (default is CPU).  
📊 **Result Visualization**: 
- Displays DeepFake detection stats in a table format.
- Provides downloadable CSV of detection results.
- Visualizes results with bar charts for DeepFake probability and inference time.

## Usage

1. Select the "Detector" option from the sidebar.
2. Upload an image/video or provide a URL.
3. Choose the DeepFake detection model(s) you want to use.
4. Optionally select GPU processing if available.
5. Click "Real or Fake? 🤔" to start the analysis.
6. View the results in the displayed charts and tables.

## Additional Sections

- **Examples**: View sample DeepFakes by selecting "Examples" from the sidebar.
- **About**: Learn about the detectors used in the app and their original authors.
- **Learn**: Access educational resources about DeepFakes.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/siddharthksah/DeepSafe
    cd DeepSafe
    ```
2. Create a conda environment:
    ```bash
    conda create -n deepsafe python==3.8 -y
    conda activate deepsafe
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Download Model Weights from Google Drive:

    <a href="https://drive.google.com/drive/folders/1bdzx0TJELfHQ4dDtzeTFtTFMpO4U7I-O?usp=drive_link" target="_blank">
        <img src="https://img.shields.io/badge/Download%20Models-Google%20Drive-blue?style=for-the-badge&logo=googledrive" alt="Google Drive Link">
    </a>

    ```python
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    import os

    # Authenticate and create the PyDrive client.
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    # Specify the folder ID (the part after 'folders/' in the URL)
    folder_id = '1UmMTuXPmu-eYfskbrGgZ1uNXceznPQ6o'

    # Create 'models' directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # List all files in the folder
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

    # Download each file to the 'models' directory
    for file in file_list:
        file.GetContentFile(os.path.join('models', file['title']))

    print("Download complete.")
    ```
5. Start the application:
    ```bash
    streamlit run main.py
    ```

## Benchmarking

DeepSafe includes a powerful benchmarking feature that allows users to benchmark their datasets against selected deepfake detection models. The results include accuracy, precision, and recall metrics, along with detailed visualizations.

### Benchmark Your Datasets

1. **Select Dataset Type**: Choose between Image or Video datasets.
2. **Choose Dataset**: Select an available dataset for benchmarking.
3. **Model Selection**: Pick the deepfake detection models you want to benchmark your dataset against.
4. **Start Benchmarking**: Click the "Benchmark Dataset" button to initiate the benchmarking process.

The benchmarking results are displayed in detailed bar charts for DeepFake probability and inference time, and a downloadable CSV of the detection results is provided.

### Adding Your Own Custom Dataset

1. **Dataset Structure**: Ensure your dataset follows this folder structure:
    ```
    datasets/
    └── image/ (or video/)
        └── your_dataset_name/
            ├── real/
            │   ├── image1.jpg
            │   └── ...
            └── fake/
                ├── image1.jpg
                └── ...
    ```
2. **Config File**: Create a `.config` file within your dataset folder to provide metadata about your dataset. Here’s an example of a `.config` file:
    ```ini
    [Dataset]
    name = Your Dataset Name
    description = A brief description of your dataset.
    source = URL or source information.
    ```
3. **Upload Dataset**: Place your dataset in the appropriate folder (`datasets/image/` or `datasets/video/`).

4. **Run Benchmark**: Follow the steps in the benchmarking section to benchmark your custom dataset.

## Future Work

DeepSafe acts as a platform where newer models can be incorporated into the app.

## Contributing

Any kind of enhancement or contribution is welcomed. You can contribute your comments, questions, resources, and apps as [issues](https://github.com/siddharthksah/DeepSafe/issues) or [pull requests](https://github.com/siddharthksah/DeepSafe/pulls) to the [source code](https://github.com/siddharthksah/DeepSafe).

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

## Acknowledgments

| Methods       | Repositories                                                                                                     | Release Date |
| ------------- | ---------------------------------------------------------------------------------------------------------------- | ------------ |
| MesoNet       | https://github.com/DariusAf/MesoNet                                                                              | 2018.09      |
| FWA           | [https://github.com/danmohaha/CVPRW2019_Face_Artifacts](https://github.com/danmohaha/CVPRW2019_Face_Artifacts)  | 2018.11      |
| VA            | https://github.com/FalkoMatern/Exploiting-Visual-Artifacts                                                       | 2019.01      |
| Xception      | https://github.com/ondyari/FaceForensics                                                                         | 2019.01      |
| ClassNSeg     | https://github.com/nii-yamagishilab/ClassNSeg                                                                    | 2019.06      |
| Capsule       | https://github.com/nii-yamagishilab/Capsule-Forensics-v2                                                         | 2019.1       |
| CNNDetection  | https://github.com/peterwang512/CNNDetection                                                                     | 2019.12      |
| DSP-FWA       | https://github.com/danmohaha/DSP-FWA                                                                             | 2019.11      |
| Upconv        | https://github.com/cc-hpc-itwm/UpConv                                                                            | 2020.03      |
| WM            | https://github.com/cuihaoleo/kaggle-dfdc                                                                         | 2020.07      |
| Selim         | [https://github.com/selimsef/dfdc_deepfake_challenge](https://github.com/selimsef/dfdc_deepfake_challenge)       | 2020.07      |
| Photoshop FAL | https://peterwang512.github.io/FALdetector/                                                                      | 2019         |
| FaceForensics | https://github.com/ondyari/FaceForensics                                                                         | 2018.03      |
| CViT          | https://github.com/erprogs/CViT                                                                                  | 2021         |
| Boken         | https://github.com/beibuwandeluori/DeeperForensicsChallengeSolution                                              | 2020         |
| GANimageDetection | [GANimageDetection](https://github.com/grip-unina/GANimageDetection) | [License](https://github.com/grip-unina/GANimageDetection/blob/main/LICENSE.md) |
## License

This project is licensed under a dual license:

1. **Open Source License (MIT License)**: For personal and non-commercial use.
2. **Commercial License**: For any commercial use, please contact [Siddharth](mailto:siddharth123sk@gmail.com) to obtain a commercial license.

## Contact Information

For questions, please contact me at siddharth123sk[@]gmail.com

Made with ❤️ by [Siddharth](https://siddharthksah.github.io/)
