import os
import pytest
import tempfile
from fastapi.testclient import TestClient

from app.main import app
from app.utils.faiss_manager import FaissManager


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(scope="function")
def faiss_manager(test_data_dir):
    """Create a FAISS manager for testing"""
    # Override the data directory for testing
    os.environ["FAISS_DATA_DIR"] = test_data_dir

    # Create a fresh manager for each test
    manager = FaissManager(data_dir=test_data_dir)

    yield manager


@pytest.fixture(scope="function")
def api_client():
    """Create a test client for the FastAPI app"""
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="function")
def test_tenant():
    """Get a test tenant ID"""
    return "test-tenant-001"


@pytest.fixture(scope="function")
def test_api_key():
    """Get a test API key"""
    return "test-api-key-001"


@pytest.fixture(scope="function", autouse=True)
def override_auth_dependencies(monkeypatch, test_tenant):
    """Override auth dependencies for testing"""
    # Import the auth module
    from app.utils import auth

    # Mock the get_tenant_id dependency
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
