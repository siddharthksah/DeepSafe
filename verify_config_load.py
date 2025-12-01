import sys
import os

# Add current directory to path so we can import deepsafe_utils
sys.path.append(os.getcwd())

from deepsafe_utils.config_manager import ConfigManager


def verify_config():
    cm = ConfigManager()
    if not cm.is_config_loaded_successfully():
        print("Failed to load config.")
        sys.exit(1)

    video_models = cm.get_model_endpoints("video")
    print(f"Configured Video Models: {list(video_models.keys())}")

    if "fake_stormer" in video_models:
        print("SUCCESS: 'fake_stormer' is found in video models configuration.")
    else:
        print("FAILURE: 'fake_stormer' NOT found in video models configuration.")


if __name__ == "__main__":
    verify_config()
