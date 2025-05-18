import os
import pytest
import tempfile
from fastapi.testclient import TestClient

from app.main import app
from app.utils.faiss_manager import FaissManager, _faiss_manager_instance, get_faiss_manager
from app.utils import auth


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory for tests: {temp_dir}")
        # Make sure the directory exists
        os.makedirs(temp_dir, exist_ok=True)
        yield temp_dir


@pytest.fixture(scope="function", autouse=True)
def reset_faiss_manager_singleton(test_data_dir):
    """Reset the FaissManager singleton for each test to use test_data_dir"""
    global _faiss_manager_instance

    # Set the environment variable for the data dir
    os.environ["FAISS_DATA_DIR"] = test_data_dir
    print(f"Set FAISS_DATA_DIR to: {test_data_dir}")

    # Reset the singleton
    _faiss_manager_instance = None

    # Get a fresh instance for the test
    manager = get_faiss_manager()
    print(f"Reset FaissManager singleton with data_dir: {manager.data_dir}")

    yield manager


@pytest.fixture(scope="function")
def test_tenant():
    """Get a test tenant ID"""
    return "test-tenant-001"


@pytest.fixture(scope="function")
def test_api_key():
    """Get a test API key"""
    return "test-api-key-001"


@pytest.fixture(scope="function", autouse=True)
def override_auth_dependencies(monkeypatch, test_tenant, test_api_key):
    """Override auth dependencies for testing"""
    # Add our test API key to the API_KEYS dictionary
    auth.API_KEYS[test_api_key] = test_tenant
    print(f"Added test API key {test_api_key} for tenant {test_tenant}")

    # Mock the get_tenant_id dependency to avoid API key validation
    async def mock_get_tenant_id():
        return test_tenant

    # Mock the validate_tenant_access function
    def mock_validate_tenant_access(requesting_tenant, resource_tenant):
        if requesting_tenant != resource_tenant:
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this resource"
            )

    # Apply the monkeypatches
    monkeypatch.setattr(auth, "get_tenant_id", mock_get_tenant_id)
    monkeypatch.setattr(auth, "validate_tenant_access", mock_validate_tenant_access)


@pytest.fixture(scope="function")
def api_client(test_api_key):
    """Create a test client for the FastAPI app with authentication headers"""
    with TestClient(app, headers={"X-API-Key": test_api_key}) as client:
        yield client
