## Team: BokingChen 
This folder contains the neceseray code to run inference using the [DeeperForensics Challenge](https://competitions.codalab.org/competitions/25228) solution by team BokingChen.  For more details on the project please visit the GitHub [repository](https://github.com/beibuwandeluori/DeeperForensicsChallengeSolution).

### Requirements

- numpy==1.18.1
- opencv_python==4.2.0.34
- torch==1.4.0
- torchvision==0.5.0
- boto3==1.17.5
- albumentations==0.5.2
- efficientnet_pytorch==0.7.0
- facenet_pytorch==2.5.1
- Pillow==8.1.1
- pretrainedmodels==0.7.4
- Flask==1.1.2
- pandas==1.0.3

### Usage instructions

``` python
from ensemble import Ensemble
submit = Ensemble()
prediction = submit.inference(video_path)
