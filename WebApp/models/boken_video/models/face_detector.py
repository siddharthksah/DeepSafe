import cv2
from PIL import Image
import numpy as np


__all__ = ['DetectionPipeline']


def get_boundingbox(box, width, height, scale=1.2, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
    x2 = x1 + size_bb
    y2 = y1 + size_bb

    return [x1, y1, x2, y2]


def crop_face(frames, boxes, scale=1.2):
    faces = []
    for frame, bs in zip(frames, boxes):
        # frame 是 Image RGB格式(h, w, c)
        height, width = np.array(frame).shape[:2]

        if bs is None:
            continue
        box = bs[0]  # 每个视频只截取一张图片, 有些帧可能空脸
        box = get_boundingbox(box, width, height, scale=scale)  # box放大1.2倍数
        face = frame.crop(box)
        face = np.array(face)
        # face = cv2.resize(face, (256, 256))
        faces.append(face)
    return np.array(faces, dtype=object)


class DetectionPipeline:
    """Pipeline class for detecting faces in the frames of a video file."""

    def __init__(self, detector, change2Image=True):
        self.detector = detector
        self.change2Image = change2Image

    def __call__(self, video_frames):
        frames = []
        for frame in video_frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)

        boxes, _ = self.detector.detect(frames, landmarks=False)
        faces = crop_face(frames, boxes)

        return np.array(faces)
