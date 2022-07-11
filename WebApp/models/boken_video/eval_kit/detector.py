from abc import ABC, abstractmethod


class DeeperForensicsDetector(ABC):
    def __init__(self):
        """
        Participants may define their own initialization process.
        During this process you can set up your network.
        """

    @abstractmethod
    def inference(self, video_frames):
        """
        Process a list of video frames, the evaluation toolkit will measure the runtime of every call to this method.

        params:
            - video_frames (list): may be a list of numpy arrays with dtype=np.uint8 representing frames of **one** video
        return:
            - probability (float)
        """
        pass
