import pytest
import sys
import os
from pathlib import Path

# Add project root to sys.path so we can import deepsafe_utils and api
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deepsafe_utils.config_manager import ConfigManager
from deepsafe_utils.api_client import APIClient

@pytest.fixture(scope="session")
def config_manager():
    return ConfigManager()

@pytest.fixture(scope="session")
def api_client(config_manager):
    # Assumes Docker containers are running for integration tests
    return APIClient(config_manager, "image", run_from_host=True)

@pytest.fixture
def sample_image_path():
    # Return absolute path to a sample image
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test_samples', 'sample_image.jpg'))
    if not os.path.exists(path):
        pytest.skip(f"Sample image not found at {path}")
    return path
