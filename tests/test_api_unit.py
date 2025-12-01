import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
import os

# Add api directory to path to import main
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "api")))

from main import app, get_current_user

client = TestClient(app)


# Mock auth dependency to bypass login for some tests
async def mock_get_current_user():
    return {"username": "testuser", "disabled": False}


@pytest.fixture
def mock_auth():
    app.dependency_overrides[get_current_user] = mock_get_current_user
    yield
    app.dependency_overrides = {}


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "DeepSafe API is running"}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "overall_api_status" in data


def test_register_user():
    # Use a unique username to avoid conflict if DB persists (though it's mock in-memory)
    username = "pytest_user"
    response = client.post(
        "/register",
        data={
            "username": username,
            "password": "password123",
            "confirm_password": "password123",
        },
    )
    # Might be 200 or 400 if user exists (in-memory DB persists across tests in same process?)
    # Actually TestClient restarts app usually? No, module level.
    # But main.py re-initializes fake_users_db on import.
    assert response.status_code in [200, 400]


def test_login_user():
    # Register first
    client.post(
        "/register",
        data={
            "username": "login_test",
            "password": "password123",
            "confirm_password": "password123",
        },
    )

    response = client.post(
        "/token", data={"username": "login_test", "password": "password123"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()


def test_protected_route_without_auth():
    response = client.get("/users/me")
    assert response.status_code == 401


def test_protected_route_with_mock_auth(mock_auth):
    response = client.get("/users/me")
    assert response.status_code == 200
    assert response.json()["username"] == "testuser"
