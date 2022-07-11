##https://github.com/polimi-ispl/icpr2020dfdc

import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.model_zoo import load_url
from PIL import Image
import matplotlib.pyplot as plt

import sys
# sys.path.append('..')
from blazeface import FaceExtractor, BlazeFace
from architectures import fornet,weights
from isplutils import utils


def predictor_Faceforensics(path):


    """
    Choose an architecture between
    - EfficientNetB4
    - EfficientNetB4ST
    - EfficientNetAutoAttB4
    - EfficientNetAutoAttB4ST
    - Xception
    """
    net_model = 'EfficientNetAutoAttB4'

    """
    Choose a training dataset between
    - DFDC
    - FFPP
    """
    train_db = 'DFDC'

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    face_policy = 'scale'
    face_size = 224


    model_url = weights.weight_url['{:s}_{:s}'.format(net_model,train_db)]
    net = getattr(fornet,net_model)().eval().to(device)
    net.load_state_dict(load_url(model_url,map_location=device,check_hash=True))

    transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)

    facedet = BlazeFace().to(device)
    facedet.load_weights("models/faceforensics_image/blazeface/blazeface.pth")
    facedet.load_anchors("models/faceforensics_image/blazeface/anchors.npy")
    face_extractor = FaceExtractor(facedet=facedet)


    #im_real = Image.open('FaceForensics++/samples/lynaeydofd_fr0.jpg')
    #im_fake = Image.open('FaceForensics++/samples/mqzvfufzoq_fr0.jpg')

    im_original = Image.open(path)

    #fig,ax = plt.subplots(1,2,figsize=(12,4))

    #ax[0].imshow(im_real)
    #ax[0].set_title('REAL')

    #ax[1].imshow(im_fake)
    #ax[1].set_title('FAKE')


    #im_real_faces = face_extractor.process_image(img=im_real)
    #im_fake_faces = face_extractor.process_image(img=im_fake)

    im_original_face = face_extractor.process_image(img=im_original)

    im_original_face = im_original_face['faces'][0] # take the face with the highest confidence score found by BlazeFace




    #im_real_face = im_real_faces['faces'][0] # take the face with the highest confidence score found by BlazeFace
    #im_fake_face = im_fake_faces['faces'][0] # take the face with the highest confidence score found by BlazeFace


    #fig,ax = plt.subplots(1,2,figsize=(8,4))

    #ax[0].imshow(im_real_face)
    #ax[0].set_title('REAL')

    #ax[1].imshow(im_fake_face)
    #ax[1].set_title('FAKE');



    faces_t = torch.stack( [ transf(image=im)['image'] for im in [im_original_face, im_original_face] ] )

    with torch.no_grad():
        faces_pred = torch.sigmoid(net(faces_t.to(device))).cpu().numpy().flatten()


        """
    Print scores.
    A score close to 0 predicts REAL. A score close to 1 predicts FAKE.
    """
    #print('Score for FAKE face: {:.4f}'.format(faces_pred[1]))
    #print(faces_pred)
    probab = (1 - float('{:.4f}'.format(faces_pred[0])))
    with open("models/faceforensics_image/result.txt", "w") as text_file:
        text_file.write(str(probab))
    #print(probab)
    return(probab)

    
path = "temp/delete.jpg"
(predictor_Faceforensics(path))