import warnings
warnings.filterwarnings("ignore")
import gc
import os
import torch

from eval_kit.detector import DeeperForensicsDetector
from eval_kit.client_dev import get_local_frames_iter
from models import model_selection, get_efficientnet, DetectionPipeline
from facenet_pytorch import MTCNN, extract_face
from utils import *


class Ensemble(DeeperForensicsDetector):
    def __init__(self):
        super(Ensemble, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        self.is_ensamble = True
        if not self.is_ensamble:
            # model, _, *_ = model_selection('se_resnext101_32x4d', num_out_classes=2, dropout=0.5)
            model = get_efficientnet(
                model_name='efficientnet-b0', num_classes=2, pretrained=False)
            model_path = 'models/boken_video/weights/boken/efn-b0_LS_27_loss_0.2205.pth'
            model.load_state_dict(torch.load(
                model_path, map_location=self.device))
            #print('Load model in:', model_path)
            self.model = model.to(self.device)
        else:
            self.models = self.load_models(model_names=['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2'],
                                           model_paths=['models/boken_video/weights/boken/efn-b0_LS_27_loss_0.2205.pth',
                                                        'models/boken_video/weights/boken/efn-b1_LS_6_loss_0.1756.pth',
                                                        'models/boken_video/weights/boken/efn-b2_LS_12_loss_0.1728.pth'])
        self.mtcnn = MTCNN(margin=14, keep_all=True,
                           factor=0.6, device=self.device).eval()

    def load_models(self, model_names, model_paths):
        models = []
        for i in range(len(model_names)):
            model = get_efficientnet(
                model_name=model_names[i], num_classes=2, pretrained=False)
            model_path = model_paths[i]
            model.load_state_dict(torch.load(
                model_path, map_location=self.device))
            #print('Load model ', i, 'in:', model_path)
            model.to(self.device)
            models.append(model)

        return models

    def inference(self, video_path):
        # Here, we just simply return possibility of 0.5
        # Define face detection pipeline
        #print(video_path)
        video_frames = get_local_frames_iter(video_path)
        detection_pipeline = DetectionPipeline(detector=self.mtcnn)
        # Load frames and find faces
        faces = detection_pipeline(video_frames)

        if faces.shape == (0,):  # 排除没有提取到人脸的视频
            #print('No face detect in video!')
            pred = 0.5
        else:
            if not self.is_ensamble:
                prediction, output = predict_with_model(
                    faces, self.model, device=self.device, is_tta=False)
                pred = output[:, 1]
                pred = sum(pred) / len(pred)
                pred = clip_pred(pred, threshold=0.01)
            else:
                pred = []
                for model in self.models:
                    prediction, output = predict_with_model(
                        faces, model, device=self.device, is_tta=False)
                    pred_i = output[:, 1]
                    pred_i = sum(pred_i) / len(pred_i)
                    pred.append(pred_i)

                pred = sum(pred) / len(pred)
                pred = clip_pred(pred, threshold=0.01)
        
        #print(pred)
        with open("models/boken_video/result.txt", "w") as text_file:
            text_file.write(str(pred))
        return pred

    def __del__(self):
        del self.models
        torch.cuda.empty_cache()
        gc.collect()
