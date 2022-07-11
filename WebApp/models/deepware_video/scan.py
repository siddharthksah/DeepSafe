import os, sys, glob, time, json
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from dface import MTCNN, FaceNet
from concurrent.futures import ThreadPoolExecutor
import timm.models.efficientnet as effnet
from sklearn.cluster import DBSCAN
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.cuda.amp import autocast


device = 'cpu'

margin = 0
scan_fps = 1
batch_size = 32
face_size = None

mtcnn = None
facenet = None
deepware = None


class EffNet(nn.Module):
	def __init__(self, arch='b3'):
		super(EffNet, self).__init__()
		fc_size = {'b1':1280, 'b2':1408, 'b3':1536, 'b4':1792,
				   'b5':2048, 'b6':2304, 'b7':2560}
		assert arch in fc_size.keys()
		effnet_model = getattr(effnet, 'tf_efficientnet_%s_ns'%arch)
		self.encoder = effnet_model()
		self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
		self.dropout = nn.Dropout(0.2)
		self.fc = nn.Linear(fc_size[arch], 1)

	def forward(self, x):
		x = self.encoder.forward_features(x)
		x = self.avg_pool(x).flatten(1)
		x = self.dropout(x)
		x = self.fc(x)
		return x


class Ensemble(nn.Module):
	def __init__(self, models):
		super(Ensemble, self).__init__()
		self.models = nn.ModuleList(models)

	def forward(self, x):
		preds = []
		for i, model in enumerate(self.models):
			y = model(x)
			preds.append(y)
		final = torch.mean(torch.stack(preds), dim=0)
		return final


def get_frames(video, batch_size=10, target_fps=1):
	vid = cv2.VideoCapture(video)
	total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
	if total <= 0:
		return None
	fps = vid.get(cv2.CAP_PROP_FPS)
	if target_fps > fps:
		target_fps = fps
	nfrm = int(total/fps*target_fps)
	idx = np.linspace(0, total, nfrm, endpoint=False, dtype=int)
	batch = []
	for i in range(total):
		ok = vid.grab()
		if i not in idx:
			continue
		ok, frm = vid.retrieve()
		if not ok:
			continue
		h, w = frm.shape[:2]
		if w*h > 1920*1080:
			scale = 1920/max(w, h)
			frm = cv2.resize(frm, (int(w*scale), int(h*scale)))
		frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
		batch.append(frm)
		if len(batch) == batch_size:
			yield batch
			batch = []
	if len(batch) > 0:
		yield batch
	vid.release()


def crop_face(img, box, margin=1):
	x1, y1, x2, y2 = box
	size = int(max(x2-x1, y2-y1) * margin)
	center_x, center_y = (x1 + x2)//2, (y1 + y2)//2
	x1, x2 = center_x-size//2, center_x+size//2
	y1, y2 = center_y-size//2, center_y+size//2
	face = Image.fromarray(img).crop([x1, y1, x2, y2])
	return np.asarray(face)


def fix_margins(faces, new_margin):
	fixed = []
	for face in faces:
		img = Image.fromarray(face)
		w, h = img.size
		sz = int(w/margin*new_margin)
		img = TF.center_crop(img, (sz, sz))
		fixed.append(np.asarray(img))
	return fixed


def cluster(faces):
	if margin != 1.2:
		faces = fix_margins(faces, 1.2)

	embeds = facenet.embedding(faces)

	dbscan = DBSCAN(eps=0.35, metric='cosine', min_samples=scan_fps*5)
	labels = dbscan.fit_predict(embeds)

	clusters = defaultdict(list)
	for idx, label in enumerate(labels):
		clusters[label].append(idx)
	bad = {0: clusters.pop(-1, [])}
	if len(clusters) == 0 and len(bad[0]) >= scan_fps*5:
		return bad
	return clusters


def id_strategy(pred, t=0.8):
	pred = np.array(pred)
	fake = pred[pred >= t]
	real = pred[pred <= (1-t)]
	if len(fake) >= int(len(pred)*0.9):
		return np.mean(fake)
	if len(real) >= int(len(pred)*0.9):
		return np.mean(real)
	return np.mean(pred)


confident = lambda p: np.mean(np.abs(p-0.5)*2) >= 0.7
label_spread = lambda x: x-np.log10(x) if x >= 0.8 else x


def strategy(preds):
	#
	# If there is a fake id and we're confident,
	# return spreaded fake score, otherwise return
	# the original fake score.
	# If everyone is real and we're confident return
	# the minimum real score, otherwise return the
	# mean of all predictions.
	#
	preds = np.array(preds)
	p_max = np.max(preds)
	if p_max >= 0.8:
		if confident(preds):
			return label_spread(p_max)
		return p_max
	if confident(preds):
		return np.min(preds)
	return np.mean(preds)


preprocess = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225]),
])


def scan(file):
	frames = get_frames(file, batch_size, scan_fps)
	faces, preds = [], []

	for batch in frames:
		results = mtcnn.detect(batch)
		for i, res in enumerate(results):
			if res is None:
				continue
			boxes, probs, lands = res
			for j, box in enumerate(boxes):
				if probs[j] > 0.98:
					face = crop_face(batch[i], box, margin)
					face = cv2.normalize(face, None, 0, 255, cv2.NORM_MINMAX)
					face = cv2.resize(face, face_size)
					faces.append(face)

	if len(faces) == 0:
		return None, []

	with torch.no_grad():
		n = batch_size
		splitted_faces = int(np.ceil(len(faces)/n))

		for i in range(splitted_faces):
			faces_proc = []
			for face in faces[i*n:(i+1)*n]:
				face = preprocess(face)
				faces_proc.append(face)

			x = torch.stack(faces_proc)
			with autocast():
				y = deepware(x.to(device))
			preds.append(y)

	preds = torch.sigmoid(torch.cat(preds, dim=0))[:,0].cpu().numpy()
	return list(preds), faces


def process(file):
	try:
		preds, faces = scan(file)
		if preds is None:
			return 0.5

		clust = cluster(faces)
		if len(clust) == 0:
			return 0.5

		id_preds = defaultdict(list)

		for label, indices in clust.items():
			for idx in indices:
				id_preds[label].append(preds[idx])

		preds = [id_strategy(preds) for preds in id_preds.values()]
		if len(preds) == 0:
			return 0.5

		score = strategy(preds)
		return np.clip(score, 0.01, 0.99)

	except Exception as e:
		print(e, file, file=sys.stderr)
		return 0.5


def init(models_dir, cfg_file, dev):
	global device, mtcnn, facenet, deepware, margin, face_size

	with open(cfg_file) as f:
		cfg = json.loads(f.read())

	arch = cfg['arch']
	margin = cfg['margin']
	face_size = (cfg['size'], cfg['size'])

	#print(f'margin: {margin}, size: {face_size}, device: {dev}, arch: {arch}')

	device = dev
	mtcnn = MTCNN(device)
	facenet = FaceNet(device)

	if os.path.isdir(models_dir):
		model_paths = glob.glob('%s/*.pt'%models_dir)
	else:
		model_paths = [models_dir]
	model_list = []
	assert len(model_paths) >= 1
	#print('loading %d models...'%len(model_paths))

	for model_path in model_paths:
		b3_model = EffNet(arch)
		checkpoint = torch.load(model_path, map_location="cpu")
		b3_model.load_state_dict(checkpoint)
		del checkpoint
		model_list.append(b3_model)

	deepware = Ensemble(model_list).eval().to(device)


def main():
	if len(sys.argv) != 5:
		print('usage: scan.py <scan_dir> <models_dir> <cfg_file> <device>')
		exit(1)

	init(sys.argv[2], sys.argv[3], sys.argv[4])

	if os.path.isdir(sys.argv[1]):
		files = glob.glob(sys.argv[1]+'/*')
	else:
		with open(sys.argv[1], 'r') as f:
			files = [l.strip() for l in f.readlines()]

	with ThreadPoolExecutor(max_workers=4) as ex:
		preds = list(tqdm(ex.map(process, files), total=len(files)))

	# with open('deepware/result.csv', 'w') as f:
	# 	print('filename,label', file=f)
	# 	for i, file in enumerate(files):
	# 		print('%s,%.4f'%(file, preds[i]), file=f)

	with open("models/deepware_video/result.txt", "w") as text_file:
            text_file.write(str(preds[0]))

if __name__ == '__main__':
	np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
	main()
