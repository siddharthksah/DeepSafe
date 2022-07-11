import boto3
import json
import os
import time
import sys

import logging
import zipfile
try:
    import zlib
    compression = zipfile.ZIP_DEFLATED
except:
    compression = zipfile.ZIP_STORED

from io import BytesIO

from eval_kit.extract_frames import extract_frames

# EVALUATION SYSTEM SETTINGS
# DON'T CHANGE ANY CODE HERE, OR YOU MAY ENCOUNTER SOME UNEXPECTED PROBLEMS!

WORKSPACE_BUCKET = 'deeperforensics-eval-workspace'
VIDEO_LIST_PATH = 'test-data-3k/deeperforensicis_runtime_eval_video_list.txt'
VIDEO_PREFIX = 'test-data-3k/'
UPLOAD_PREFIX = 'test-output/'
TMP_PATH = '/tmp'


def _get_s3_video_list(s3_bucket, s3_path):
    s3_client = boto3.client('s3', region_name='us-west-2')
    f = BytesIO()
    s3_client.download_fileobj(s3_bucket, s3_path, f)
    lines = f.getvalue().decode('utf-8').split('\n')
    return [x.strip() for x in lines if x != '']


def _download_s3_video(s3_bucket, s3_path, filename):
    s3_client = boto3.client('s3', region_name='us-west-2')
    local_path = os.path.join(TMP_PATH, filename)
    s3_client.download_file(s3_bucket, s3_path, local_path)

def _upload_output_to_s3(data, filename, s3_bucket, s3_prefix):
    s3_client = boto3.client('s3', region_name='us-west-2')
    local_path = os.path.join(TMP_PATH, filename)
    s3_path = os.path.join(s3_prefix, filename)

    # Put data into binary file
    data_str = json.dumps(data)
    encode_data = data_str.encode()
    with open(local_path, 'wb') as f:
        f.write(encode_data)
    s3_client.upload_file(local_path, s3_bucket, s3_path)


def get_job_name():
    return os.environ['DEEPERFORENSICS_EVAL_JOB_NAME']


def upload_eval_output(output_probs, output_times, num_frames, job_name, total_time):
    """
    This function uploads the testing output to S3 to trigger evaluation.
    params:
    - output_probs (dict): dict of probability of every video
    - output_times (dict): dict of processing time of every video
    - num_frames (dict): dict of number of frames extracting from every video
    - job_name (str)
    """
    upload_data = {
        i: {
            "prob": output_probs[i],
            "runtime": output_times[i],
            "num_frames": num_frames[i]
        } for i in output_probs
    }

    upload_data["total_time"] = total_time

    filename = '{}.bin'.format(job_name)

    _upload_output_to_s3(upload_data, filename, WORKSPACE_BUCKET, UPLOAD_PREFIX)

    logging.info("output uploaded to {}{}".format(UPLOAD_PREFIX, filename))

def get_frames_iter():
    """
    This function returns a iterator of frames of test videos.
    Each iteration provides a tuple of (video_id, frames), each frame will be in RGB color format with array shape of (height, width, 3).
    return: tuple(video_id: str, frames: list)
    """
    video_list = _get_s3_video_list(WORKSPACE_BUCKET, VIDEO_LIST_PATH)
    logging.info("got video list, {} videos".format(len(video_list)))

    for video_id in video_list:
        # get video from s3
        st = time.time()
        try:
            _download_s3_video(WORKSPACE_BUCKET, os.path.join(VIDEO_PREFIX, video_id), video_id)
        except:
            logging.info("Failed to download video: {}".format(os.path.join(VIDEO_PREFIX, video_id)))
            raise
        video_local_path = os.path.join(TMP_PATH, video_id) # local path of the video named video_id
        frames = extract_frames(video_local_path)
        elapsed = time.time() - st
        logging.info("video downloading & frames extracting time: {}".format(elapsed))
        yield video_id, frames
        try:
            os.remove(video_local_path) # remove the video named video_id
        except:
            logging.info("Failed to delete this video, error: {}".format(sys.exc_info()[0]))

def get_local_frames_iter(max_number=None):
    """
    This function returns a iterator of frames of test videos.
    It is used for local test of participating algorithms.
    Each iteration provides a tuple of (video_id, frames), each frame will be in RGB color format with array shape of (height, width, 3)
    return: tuple(video_id: str, frames: list)
    """
    video_list = [x.strip() for x in open(VIDEO_LIST_PATH)]
    logging.info("got local video list, {} videos".format(len(video_list)))

    for video_id in video_list:
        # get video from local file
        try:
            frames = extract_frames(os.path.join(VIDEO_PREFIX, video_id))
        except:
            logging.info("Failed to read image: {}".format(os.path.join(VIDEO_PREFIX, video_id)))
            raise
        yield video_id, frames


def verify_local_output(output_probs, output_times, num_frames):
    """
    This function prints the ground truth and prediction for the participant to verify, calculates average FPS.
    params:
    - output_probs (dict): dict of probability of every video
    - output_times (dict): dict of processing time of every video
    - num_frames (dict): dict of number of frames extracting from every video
    """
    # gts = json.load(open('test-data/local_test_groundtruth.json'), parse_int=float)
    gts = json.load(open('test-data/local_test_groundtruth.json'))

    all_time = 0
    all_num_frames = 0
    for k in gts:

        assert k in output_probs and k in output_times, ValueError("The detector doesn't work on video {}".format(k))

        all_time += output_times[k]
        all_num_frames += num_frames[k]

        logging.info("Video ID: {}, Runtime: {}".format(k, output_times[k]))
        logging.info("\tgt: {}".format(gts[k]))
        logging.info("\toutput probability: {}".format(output_probs[k]))
        logging.info("\tnumber of frame: {}".format(num_frames[k]))
        logging.info("\toutput time: {}".format(output_times[k]))

        logging.info(" ")

    average_fps = all_num_frames / all_time
    logging.info("Done. Average FPS: {:.03f}".format(average_fps))
