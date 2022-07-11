import numpy as np
import cv2

def extract_frames(video_path, n_frames=15):
    """
    Extract frames from a video. You can use either provided method here or implement your own method.

    params:
        - video_local_path (str): the path of video.
    return:
        - frames (list): a list containing frames extracted from the video.
    """
    ########################################################################################################
    # You can change the lines below to implement your own frame extracting method (and possibly other preprocessing),
    # or just use the provided codes.
    vid = cv2.VideoCapture(video_path)
    frames = []

    # while True:
    #     success, frame = vid.read()
    #     if not success:
    #         break
    #     if frame is not None:
    #         frames.append(frame)
    #     # Here, we extract one frame only without other preprocessing
    #     if len(frames) >= n_frames:
    #         break

    v_len = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    sample = np.linspace(0, v_len - 1, n_frames).astype(int)
    for j in range(v_len):
        success = vid.grab()
        if j in sample:
            # Load frame
            success, frame = vid.retrieve()
            if not success:
                continue
            if frame is not None:
                frames.append(frame)

    vid.release()
    return frames
    ########################################################################################################
